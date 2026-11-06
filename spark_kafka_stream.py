"""REAL Spark Structured Streaming source from three-partition Apache Kafka.

Stages: Kafka -> JSON schema -> distributed pandas UDF scoring -> Parquet sink
and independent stateful event-time Spark window aggregation -> Parquet sink.
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from config import BASE_DIR, ML_FEATURES, ANOMALY_FEATURES

TOPIC = os.getenv("ISS_KAFKA_TOPIC", "iss_telemetry")
BOOTSTRAP = os.getenv("ISS_SPARK_BOOTSTRAP", "localhost:9092")
RUNTIME = BASE_DIR / "runtime" / "kafka_spark"

SCHEMA = T.StructType([
    T.StructField("event_id", T.LongType()),
    T.StructField("timestamp", T.StringType()),
    T.StructField("producer_sent_at", T.StringType()),
    T.StructField("latitude", T.DoubleType()),
    T.StructField("longitude", T.DoubleType()),
    T.StructField("altitude_km", T.DoubleType()),
    T.StructField("speed_kmph", T.DoubleType()),
    T.StructField("rolling_avg_altitude", T.DoubleType()),
    T.StructField("rolling_avg_speed", T.DoubleType()),
    T.StructField("rolling_altitude_std", T.DoubleType()),
    T.StructField("altitude_rate", T.DoubleType()),
    T.StructField("anomaly", T.BooleanType()),
])


_MODELS = {}

def _cached_model(path):
    import joblib
    if path not in _MODELS:
        _MODELS[path] = joblib.load(path)
    return _MODELS[path]

def register_scoring_udfs(model_dir: Path):
    """Load joblib artifacts lazily on each Spark Python worker."""
    from pyspark.sql.functions import pandas_udf
    path_ml = str(model_dir / "next_direction_rf.joblib")
    path_iso = str(model_dir / "isolation_forest.joblib")
    # Worker processes are reused across microbatches where possible.
    @pandas_udf("double")
    def probability_up(*cols: pd.Series) -> pd.Series:
        import joblib
        frame = pd.concat(cols, axis=1)
        frame.columns = ML_FEATURES
        return pd.Series(_cached_model(path_ml).predict_proba(frame)[:, 1])

    @pandas_udf("boolean")
    def anomaly_flag(*cols: pd.Series) -> pd.Series:
        import joblib
        frame = pd.concat(cols, axis=1)
        frame.columns = ANOMALY_FEATURES
        return pd.Series(_cached_model(path_iso).predict(frame) == -1)

    return probability_up, anomaly_flag


def kafka_backlog(processed_offsets):
    """Estimate remaining records after latest processed offset in each partition."""
    from kafka import KafkaConsumer, TopicPartition
    consumer = KafkaConsumer(bootstrap_servers=BOOTSTRAP, consumer_timeout_ms=1000)
    try:
        tps = [TopicPartition(TOPIC, int(part)) for part in processed_offsets]
        ends = consumer.end_offsets(tps)
        return int(sum(max(0, ends[tp] - processed_offsets[tp.partition] - 1) for tp in tps))
    finally:
        consumer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset-output", action="store_true", help="Delete previous Spark outputs AND checkpoints (replay from earliest)")
    ap.add_argument("--no-windows", action="store_true", help="Disable second, stateful event-time window query")
    ap.add_argument("--trigger-seconds", type=int, default=3)
    args = ap.parse_args()
    if args.reset_output and RUNTIME.exists():
        import shutil
        shutil.rmtree(RUNTIME)
    RUNTIME.mkdir(parents=True, exist_ok=True)

    spark = (SparkSession.builder.appName("ISS-Kafka-Spark-Stream-Intelligence")
             .config("spark.sql.shuffle.partitions", "3")
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    kafka = (spark.readStream.format("kafka")
             .option("kafka.bootstrap.servers", BOOTSTRAP)
             .option("subscribe", TOPIC)
             .option("startingOffsets", "earliest")
             .option("maxOffsetsPerTrigger", 500)
             .option("failOnDataLoss", "true")
             .load())
    parsed = (kafka.select(F.col("partition").alias("kafka_partition"),
                           F.col("offset").alias("kafka_offset"),
                           F.from_json(F.col("value").cast("string"), SCHEMA).alias("event"))
              .select("kafka_partition", "kafka_offset", "event.*")
              .filter(F.col("event_id").isNotNull())
              .withColumn("event_time", F.to_timestamp("timestamp"))
              .withColumn("producer_time", F.to_timestamp("producer_sent_at")))

    # This query uses a stateful event-time window and a 2-min watermark;
    # unlike per-microbatch sums, window state persists across Spark batches.
    window_query = None
    if not args.no_windows:
        windowed = (parsed.withWatermark("event_time", "2 minutes")
                    .groupBy(F.window("event_time", "1 minute"))
                    .agg(F.count("*").alias("events"),
                         F.avg("altitude_km").alias("avg_altitude_km"),
                         F.avg("speed_kmph").alias("avg_speed_kmph"),
                         F.sum(F.col("anomaly").cast("int")).alias("reference_anomalies"))
                    .select(F.col("window.start").alias("window_start"),
                            F.col("window.end").alias("window_end"),
                            "events", "avg_altitude_km", "avg_speed_kmph", "reference_anomalies"))
        window_query = (windowed.writeStream.outputMode("append").format("parquet")
                        .option("path", str(RUNTIME / "minute_windows"))
                        .option("checkpointLocation", str(RUNTIME / "checkpoint_windows"))
                        .trigger(processingTime=f"{args.trigger_seconds} seconds")
                        .start())

    score_prob, flag_anomaly = register_scoring_udfs(BASE_DIR / "models")
    scored = (parsed
              .withColumn("probability_up", score_prob(*[F.col(f) for f in ML_FEATURES]))
              .withColumn("predicted_up", F.col("probability_up") >= F.lit(0.5))
              .withColumn("predicted_anomaly", flag_anomaly(*[F.col(f) for f in ANOMALY_FEATURES]))
              .withColumn("spark_processed_at", F.current_timestamp()))

    total_count = [0]
    started = time.perf_counter()
    metrics_path = RUNTIME / "operational_metrics.jsonl"

    def consume_batch(batch, batch_id):
        # materialize exactly once: Spark applies both models on worker processes.
        batch.persist()
        n = batch.count()
        if n == 0:
            batch.unpersist()
            return
        # Distributed storage: Spark writes batch partitions in parallel.
        (batch.write.mode("append").parquet(str(RUNTIME / "scored_events")))
        summary = batch.agg(
            F.count("*").alias("n"),
            F.sum(F.col("predicted_anomaly").cast("int")).alias("anomalies"),
            F.avg((F.unix_millis("spark_processed_at") - F.unix_millis("producer_time")) / 1000.0).alias("latency_seconds_avg"),
            F.max((F.unix_millis("spark_processed_at") - F.unix_millis("producer_time")) / 1000.0).alias("latency_seconds_max"),
        ).first()
        offsets = {int(r["kafka_partition"]): int(r["last_offset"]) for r in
                   batch.groupBy("kafka_partition").agg(F.max("kafka_offset").alias("last_offset")).collect()}
        total_count[0] += n
        elapsed = max(0.001, time.perf_counter() - started)
        record = {
            "batch_id": int(batch_id), "recorded_at": datetime.now(timezone.utc).isoformat(),
            "events_in_batch": int(n), "total_processed": int(total_count[0]),
            "mean_throughput_events_per_second": round(total_count[0] / elapsed, 2),
            "predicted_anomalies_in_batch": int(summary["anomalies"] or 0),
            "mean_processing_lag_seconds": summary["latency_seconds_avg"],
            "max_processing_lag_seconds": summary["latency_seconds_max"],
            "last_offsets_by_partition": offsets,
        }
        # Backlog can fail if broker is restarted; don't fail scoring for it.
        try:
            record["estimated_kafka_backlog"] = kafka_backlog(offsets)
        except Exception as exc:
            record["estimated_kafka_backlog"] = None
            record["backlog_error"] = str(exc)
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        batch.unpersist()
        print("SPARK MICRO-BATCH", json.dumps(record), flush=True)

    scored_query = (scored.writeStream.foreachBatch(consume_batch)
                    .option("checkpointLocation", str(RUNTIME / "checkpoint_scoring"))
                    .trigger(processingTime=f"{args.trigger_seconds} seconds")
                    .start())
    print("Kafka -> Spark active; topic:", TOPIC, "windows:", bool(window_query), flush=True)
    try:
        scored_query.awaitTermination()
    finally:
        if window_query is not None:
            window_query.stop()
        spark.stop()


if __name__ == "__main__":
    main()

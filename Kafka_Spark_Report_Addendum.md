# Technical Report Addendum — Kafka + Apache Spark (Endterm Upgrade)

## Motivation and implementation

The initial endterm prototype used a Python queue-based distributed-systems simulation, which was allowed by the assignment. The upgraded implementation adds a **real Kafka broker** and **Spark Structured Streaming**. This addendum supersedes the original report's description of the simulated distribution layer. The original report remains valid for the dataset characterization, offline ML evaluation, anomaly evaluation, and time-series results.

## Architecture

The `kafka_producer.py` reads all raw ISS observations, computes rolling features with trailing windows, attaches the available retrospective anomaly label for evaluation (not as a model input), and publishes JSON events to the `iss_telemetry` Kafka topic. A partition key/explicit index distributes events across three Kafka partitions. Each event includes its historical observation timestamp, sequence ID, and replay publication timestamp. This demonstrates a real producer/broker/partition model with reproducible event replay.

The `spark_kafka_stream.py` application connects using the Kafka Structured Streaming source, parses an explicit JSON schema and starts two streaming queries. The first applies the existing trained Random Forest (probability of next-sample altitude increasing) and Isolation Forest (anomaly flag) via pandas UDFs on Spark executors, writing scored events to Parquet. The second performs stateful **one-minute event-time windows** with a **two-minute watermark** and saves aggregated event counts, mean altitude, mean speed, and retrospective reference anomaly counts. Checkpoints record offsets and state for restarts.

`dashboard_kafka_spark.py` reads Spark-written Parquet and operational metrics, refreshing automatically. It displays two changing telemetry charts plus counts, anomaly totals, next-altitude probability, throughput, processing lag and estimated backlog.

## Integrity and temporal leakage

Prediction features depend on the current event or previously observed events, never on the subsequent altitude outcome. Historical next-step labels support offline fitting/evaluation. Offline Random Forest metrics and anomaly confusion-matrix metrics originate from the earlier chronological 70/30 split; they must not be represented as results of a new Kafka+Spark run. The retrospective anomaly field is not used by the inference models.

## Operational measurements

The application saves, for each microbatch: processed event count, cumulative throughput, mean/maximum delay between producer send and Spark processing time, Kafka per-partition offsets, and an estimated remaining Kafka backlog. The replay can be operated slowly for visualization or rapidly to stress the queue. Kafka retention and Spark microbatch checkpoint handling permit repeated streaming demonstrations. A crash during Parquet writing can cause duplicated output files; the dashboard deduplicates by (Kafka partition, offset), but the output sink is not transactional end-to-end exactly-once.

## Scope and limitations

The Spark master is configured as `local[3]`: it schedules multiple parallel tasks but does **not** demonstrate a physically multi-node deployment. Because the sample data cover only one synthetic day, broad claims about generalization to real station operations would be unjustified. The Kafka/Spark implementation has code-level and event-schema checks; its end-to-end runtime results are **pending a live environment run**. Capture the terminal and dashboard during defence as operational evidence.

## Suggested demo sequence

1. `docker compose up -d` and `python kafka_setup.py` — demonstrate the real Kafka topic with three partitions.
2. `bash run_spark.sh` — show the Structured Streaming query running.
3. `python kafka_producer.py --limit 600 --delay 0.05` — replay the actual ISS observations into Kafka.
4. `streamlit run dashboard_kafka_spark.py` — show the continuously refreshed graphs and model outputs.
5. Inspect `runtime/kafka_spark/operational_metrics.jsonl` and `minute_windows/` — explain throughput, latency, backlog, event-time windows, checkpoints and watermark.

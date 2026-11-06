"""Publish actual ISS telemetry to Apache Kafka, preserving event-time information.

The trailing features are computed incrementally from the raw CSV, not from
future observations. The known anomaly label is used ONLY for post-hoc scoring.
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import RAW_DATA
from features import add_reproducible_stream_features

TOPIC = os.getenv("ISS_KAFKA_TOPIC", "iss_telemetry")
BOOTSTRAP = os.getenv("ISS_KAFKA_BOOTSTRAP", "localhost:9092")


def build_events(raw_path=RAW_DATA, limit=None):
    raw = pd.read_csv(raw_path, parse_dates=["timestamp"])
    enriched = add_reproducible_stream_features(raw)
    if limit is not None:
        enriched = enriched.iloc[:limit]
    # anomaly flag is a truth label stored in the supplied processed data.
    # Match by timestamp; this flag never enters either trained model.
    from config import PROCESSED_DATA
    truth = pd.read_csv(PROCESSED_DATA, usecols=["timestamp", "anomaly"])
    truth["timestamp"] = pd.to_datetime(truth["timestamp"], utc=True)
    enriched = enriched.merge(truth, on="timestamp", how="left", validate="one_to_one")
    enriched["anomaly"] = enriched["anomaly"].fillna(False).astype(bool)
    return enriched


def main():
    parser = argparse.ArgumentParser(description="Send ISS samples to Kafka")
    parser.add_argument("--limit", type=int, default=600, help="Number of events (0 = all 8640)")
    parser.add_argument("--delay", type=float, default=0.05, help="Seconds between Kafka sends; 10 = original pace")
    args = parser.parse_args()
    if args.delay < 0 or args.limit < 0:
        parser.error("--delay and --limit must be nonnegative")
    df = build_events(limit=None if args.limit == 0 else args.limit)
    from kafka import KafkaProducer
    producer = KafkaProducer(bootstrap_servers=BOOTSTRAP, acks="all", retries=5,
                             value_serializer=lambda v: json.dumps(v, allow_nan=False).encode("utf-8"),
                             key_serializer=lambda k: k.encode("utf-8"))
    start = time.perf_counter()
    for seq, row in enumerate(df.itertuples(index=False, name=None)):
        event = dict(zip(df.columns, row))
        event["timestamp"] = pd.Timestamp(event["timestamp"]).isoformat()
        event["event_id"] = seq
        event["producer_sent_at"] = datetime.now(timezone.utc).isoformat()
        event = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in event.items()}
        # Explicit partition assignment proves partition distribution in defence.
        # Each partition's message order is preserved by Kafka.
        producer.send(TOPIC, key=f"partition-{seq % 3}", value=event, partition=seq % 3)
        if seq % 300 == 0:
            print(f"Sent {seq+1}/{len(df)} events", flush=True)
        if args.delay:
            time.sleep(args.delay)
    producer.flush(timeout=120)
    producer.close()
    elapsed = time.perf_counter() - start
    print(f"DONE: {len(df)} events published in {elapsed:.2f}s ({len(df)/max(elapsed, 1e-6):.1f} events/s)")


if __name__ == "__main__":
    main()

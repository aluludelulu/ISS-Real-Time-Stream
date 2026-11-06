import json
import queue
import threading
import time
from dataclasses import dataclass, asdict

import joblib
import numpy as np
import pandas as pd

from config import (
    PROCESSED_DATA, MODEL_DIR, OUTPUT_DIR, RUNTIME_DIR,
    N_WORKERS, ML_FEATURES, ANOMALY_FEATURES,
)
from features import load_processed


@dataclass
class StreamMetrics:
    event_count: int
    throughput_events_per_sec: float
    mean_latency_ms: float
    p95_latency_ms: float
    max_backlog: int
    worker_count: int


def run_stream(worker_count=N_WORKERS, producer_delay=0.0, worker_delay=0.0, limit=None, batch_size=64):
    """Replay the ISS table through one producer and multiple partition workers.

    Each partition is FIFO, so event order is preserved inside that partition.
    Workers use micro-batches only for efficient model inference; the architecture
    still represents independent consumer stages processing partitioned streams.
    """
    df = load_processed(PROCESSED_DATA)
    if limit:
        df = df.iloc[:limit].copy()

    classifier = joblib.load(MODEL_DIR / "next_direction_rf.joblib")
    anomaly_model = joblib.load(MODEL_DIR / "isolation_forest.joblib")

    queues = [queue.Queue() for _ in range(worker_count)]
    results, latencies = [], []
    max_backlog = 0
    lock = threading.Lock()
    stop = object()

    def process_batch(worker_id, batch):
        rows = [item[2] for item in batch]
        x_ml = pd.DataFrame(rows)[ML_FEATURES]
        x_an = pd.DataFrame(rows)[ANOMALY_FEATURES]
        probs = classifier.predict_proba(x_ml)[:, 1]
        flags = anomaly_model.predict(x_an) == -1
        finished = time.perf_counter()
        local_results, local_latencies = [], []
        for item, up_prob, iso_flag in zip(batch, probs, flags):
            seq, enqueued_at, row = item
            latency_ms = (finished - enqueued_at) * 1000.0
            local_latencies.append(latency_ms)
            local_results.append({
                "sequence": seq,
                "partition": worker_id,
                "timestamp": row["timestamp"],
                "altitude_km": row["altitude_km"],
                "speed_kmph": row["speed_kmph"],
                "predicted_next_up_probability": float(up_prob),
                "isolation_forest_anomaly": bool(iso_flag),
                "ground_truth_anomaly": bool(row["anomaly"]),
                "latency_ms": latency_ms,
            })
        if worker_delay:
            time.sleep(worker_delay)
        with lock:
            results.extend(local_results)
            latencies.extend(local_latencies)

    def worker(worker_id):
        q = queues[worker_id]
        while True:
            first = q.get()
            if first is stop:
                q.task_done()
                break
            batch = [first]
            saw_stop = False
            while len(batch) < batch_size:
                try:
                    nxt = q.get_nowait()
                except queue.Empty:
                    break
                if nxt is stop:
                    q.task_done()
                    saw_stop = True
                    break
                batch.append(nxt)
            process_batch(worker_id, batch)
            for _ in batch:
                q.task_done()
            if saw_stop:
                break

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(worker_count)]
    for t in threads:
        t.start()

    start = time.perf_counter()
    for seq, row in df.iterrows():
        partition = int(seq % worker_count)
        queues[partition].put((int(seq), time.perf_counter(), row.to_dict()))
        max_backlog = max(max_backlog, sum(q.qsize() for q in queues))
        if producer_delay:
            time.sleep(producer_delay)

    for q in queues:
        q.put(stop)
    for q in queues:
        q.join()
    for t in threads:
        t.join()

    elapsed = time.perf_counter() - start
    results = sorted(results, key=lambda x: x["sequence"])
    out = pd.DataFrame(results)
    RUNTIME_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    out.to_csv(RUNTIME_DIR / "stream_output.csv", index=False)

    metric = StreamMetrics(
        event_count=len(results),
        throughput_events_per_sec=len(results) / elapsed,
        mean_latency_ms=float(np.mean(latencies)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        max_backlog=int(max_backlog),
        worker_count=worker_count,
    )
    with open(OUTPUT_DIR / "stream_metrics.json", "w", encoding="utf-8") as f:
        json.dump(asdict(metric), f, indent=2)
    print(json.dumps(asdict(metric), indent=2))
    return out, metric


if __name__ == "__main__":
    run_stream()

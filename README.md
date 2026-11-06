# IMPORTANT: Kafka + Spark upgrade

**For the REAL Kafka + Apache Spark implementation, start with [`README_KAFKA_SPARK.md`](README_KAFKA_SPARK.md).** The older guide below describes the retained simulation prototype and should not be used as the primary distributed-processing demonstration.

---

# ISS Real-Time Monitoring, Time-Series Analysis and Stream Intelligence

This project implements every Endterm requirement using the supplied ISS telemetry data (8,640 events, 10-second interval, 24 hours).

## Requirement mapping

1. **Continuously updated visualization** — `dashboard.py` replays the stream and updates altitude and speed charts plus live KPIs.
2. **Time-series analysis** — `time_series_analysis.py` resamples to 1-minute intervals, computes 5-minute rolling trends and estimates a repeating altitude period using FFT.
3. **Anomaly detection** — `train_models.py` fits an unsupervised Isolation Forest on current altitude-rate and speed features and evaluates it against the supplied `anomaly` labels.
4. **Real-time ML** — Random Forest classification predicts whether the *next* altitude sample will move up or not.
5. **Prediction-time features only** — the classifier uses current position, altitude, speed and trailing rolling statistics; no future feature is used.
6. **ML metrics** — accuracy, F1 and ROC-AUC are saved in `outputs/model_metrics.json`.
7. **Distributed stream processing** — `stream_pipeline.py` uses one producer, three partition queues and three consumer workers. Order is preserved inside each partition.
8. **Operational characteristics** — event count, throughput, mean latency, p95 latency and maximum backlog are saved in `outputs/stream_metrics.json`.

## Files

- `data/iss_data.csv` — supplied raw data.
- `data/processed_iss_data.csv` — supplied feature-enriched data with anomaly label.
- `data/detected_anomalies.csv` — supplied anomaly subset.
- `stream_generator.py` — reproducible event-by-event replay/data generator for the supplied dataset.
- `features.py` — online-safe feature utilities.
- `train_models.py` — ML + anomaly model training/evaluation.
- `time_series_analysis.py` — resampling, rolling trends and periodicity analysis.
- `stream_pipeline.py` — multi-worker partitioned replay and operational metrics.
- `dashboard.py` — live Streamlit dashboard with at least two continuously updated charts.
- `run_all.py` — one-command analysis pipeline.
- `ISS_Endterm_Project.ipynb` — defense-friendly notebook version.
- `outputs/` — evaluation metrics and figures.
- `models/` — trained model artifacts.

## Reproducibility

### Local

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
python stream_generator.py --delay 0.1 --limit 20
streamlit run dashboard.py
```

### Google Colab

1. Upload/extract this project folder to Colab or Google Drive.
2. Install dependencies: `!pip install -q pandas numpy matplotlib scikit-learn joblib streamlit`.
3. Open `ISS_Endterm_Project.ipynb` and run cells from top to bottom.
4. The notebook performs the full analysis and a compact live replay demonstration without needing Kafka.

## ML design

At event time `t`, the model sees only values already known at `t`: latitude, longitude, altitude, speed, trailing rolling averages, trailing altitude standard deviation and altitude rate from the previous observation. The target is defined from `t+1` only for training/evaluation: whether the next altitude is greater than the current altitude. A chronological 70/30 split prevents random future observations from leaking into training.

## Distributed architecture

`Producer -> Partition 0/1/2 -> Worker 0/1/2 -> ML + Anomaly Stage -> Aggregated output + metrics`

The deterministic round-robin partitioner demonstrates partitioned streams and parallel consumers while preserving sequence order inside each partition. This is an allowed simulation of distributed stream-processing concepts under the assignment specification.

## Suggested defense demo

1. Run `python train_models.py` and show F1/ROC-AUC plus anomaly precision/recall/F1.
2. Run `python stream_pipeline.py` and show throughput, p95 latency, event count and max backlog.
3. Run `streamlit run dashboard.py`, press **Start / restart replay**, and point out the two continuously updating charts and current ML/anomaly status.
4. Explain that all rolling features are trailing/current-only, which avoids look-ahead leakage.

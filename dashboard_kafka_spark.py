"""Monitor live Spark output (NOT a simulated local replay)."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent / "runtime" / "kafka_spark"
st.set_page_config(page_title="Live Kafka + Spark ISS", layout="wide")
st.title("ISS Real-Time Intelligence — Kafka + Apache Spark")
st.caption("Charts and counters read Spark's actual scored Parquet output and metrics, not a preloaded CSV replay.")

@st.fragment(run_every="3s")
def live():
    files = list((ROOT / "scored_events").glob("*.parquet"))
    if not files:
        st.info("No Spark output yet. Start Kafka, run Spark, then start the producer.")
        return
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    if df.empty:
        return
    # exactly-once presentation, even if Spark retry wrote duplicate parquet after a failure.
    df = df.drop_duplicates(subset=["kafka_partition", "kafka_offset"]).sort_values("event_time")
    last = df.iloc[-1]
    a, b, c, d = st.columns(4)
    a.metric("Scored Kafka events", f"{len(df):,}")
    b.metric("Detected anomalies", int(df["predicted_anomaly"].sum()))
    c.metric("Last UP probability", f"{last['probability_up']:.1%}")
    d.metric("Kafka partitions active", int(df["kafka_partition"].nunique()))
    recent = df.tail(600).set_index("event_time")
    st.subheader("Altitude and trailing mean")
    st.line_chart(recent[["altitude_km", "rolling_avg_altitude"]])
    st.subheader("Speed and trailing mean")
    st.line_chart(recent[["speed_kmph", "rolling_avg_speed"]])
    mpath = ROOT / "operational_metrics.jsonl"
    if mpath.exists():
        try:
            records = [json.loads(line) for line in mpath.read_text().splitlines() if line.strip()]
            if records:
                metrics = records[-1]
                x, y, z = st.columns(3)
                x.metric("Measured mean throughput", f"{metrics['mean_throughput_events_per_second']:.1f} events/s")
                y.metric("Processing lag", f"{(metrics['mean_processing_lag_seconds'] or 0):.2f} s")
                z.metric("Estimated Kafka backlog", str(metrics.get("estimated_kafka_backlog", "N/A")))
                st.dataframe(pd.DataFrame(records).tail(10), hide_index=True)
        except (ValueError, OSError) as exc:
            st.warning(f"Operational metrics are being written: {exc}")
    with st.expander("Latest scored events"):
        st.dataframe(df.tail(20)[["timestamp", "kafka_partition", "altitude_km", "predicted_up", "predicted_anomaly", "probability_up"]])

live()

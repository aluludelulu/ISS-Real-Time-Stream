import time
import joblib
import pandas as pd
import streamlit as st

from config import PROCESSED_DATA, MODEL_DIR, ML_FEATURES, ANOMALY_FEATURES
from features import load_processed

st.set_page_config(page_title="ISS Stream Intelligence", layout="wide")
st.title("ISS Real-Time Monitoring and Stream Intelligence")
st.caption("Replay of the supplied 10-second ISS stream with live visualization, anomaly detection and ML prediction.")

@st.cache_data
def data():
    return load_processed(PROCESSED_DATA)

@st.cache_resource
def models():
    return (
        joblib.load(MODEL_DIR / "next_direction_rf.joblib"),
        joblib.load(MODEL_DIR / "isolation_forest.joblib"),
    )

df = data()
clf, iso = models()

speed = st.sidebar.slider("Replay delay per batch (seconds)", 0.00, 0.50, 0.05, 0.01)
batch_size = st.sidebar.slider("Events per refresh", 10, 250, 60, 10)
max_events = st.sidebar.slider("Events to replay", 300, len(df), 1800, 300)

c1, c2, c3, c4 = st.columns(4)
event_metric = c1.empty()
anomaly_metric = c2.empty()
prob_metric = c3.empty()
lat_metric = c4.empty()
alt_chart = st.empty()
speed_chart = st.empty()
status = st.empty()

if st.button("Start / restart replay"):
    shown = []
    for end in range(batch_size, max_events + batch_size, batch_size):
        batch = df.iloc[:min(end, max_events)].copy()
        latest = batch.iloc[-1]
        up_prob = float(clf.predict_proba(pd.DataFrame([{f: latest[f] for f in ML_FEATURES}]))[0, 1])
        iso_flag = bool(iso.predict(pd.DataFrame([{f: latest[f] for f in ANOMALY_FEATURES}]))[0] == -1)
        event_metric.metric("Events processed", len(batch))
        anomaly_metric.metric("Anomalies in replay", int(batch["anomaly"].sum()))
        prob_metric.metric("Next-altitude UP probability", f"{up_prob:.1%}")
        lat_metric.metric("Current altitude", f"{latest['altitude_km']:.2f} km")

        chart_data = batch.set_index("timestamp")
        alt_chart.line_chart(chart_data[["altitude_km", "rolling_avg_altitude"]], height=300)
        speed_chart.line_chart(chart_data[["speed_kmph", "rolling_avg_speed"]], height=300)
        status.info(f"Latest event: {latest['timestamp']} | Isolation Forest anomaly: {iso_flag}")
        time.sleep(speed)
else:
    st.info("Press Start / restart replay to demonstrate continuously updated charts.")

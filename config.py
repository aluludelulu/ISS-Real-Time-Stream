from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
RUNTIME_DIR = BASE_DIR / "runtime"

RAW_DATA = DATA_DIR / "iss_data.csv"
PROCESSED_DATA = DATA_DIR / "processed_iss_data.csv"
ANOMALY_DATA = DATA_DIR / "detected_anomalies.csv"

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
N_WORKERS = 3
ROLLING_WINDOW = 30
ANOMALY_CONTAMINATION = 0.03

ML_FEATURES = [
    "latitude", "longitude", "altitude_km", "speed_kmph",
    "rolling_avg_altitude", "rolling_avg_speed",
    "rolling_altitude_std", "altitude_rate",
]
ANOMALY_FEATURES = ["altitude_rate", "speed_kmph"]

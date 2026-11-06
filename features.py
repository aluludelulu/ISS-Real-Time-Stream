import numpy as np
import pandas as pd


def load_processed(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def build_next_direction_target(df: pd.DataFrame) -> pd.DataFrame:
    """Target = 1 when next altitude sample is higher than current sample.
    All input features are computed from information available at the current event.
    """
    out = df.copy()
    next_altitude = out["altitude_km"].shift(-1)
    out["target_up"] = (next_altitude > out["altitude_km"]).astype(int)
    return out.iloc[:-1].copy()


def add_reproducible_stream_features(raw: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Create online-safe rolling features from raw ISS observations.
    Rolling windows are trailing windows and therefore do not use future values.
    """
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["speed_mps"] = df["speed_kmph"] / 3.6
    df["rolling_avg_altitude"] = df["altitude_km"].rolling(window, min_periods=1).mean()
    df["rolling_avg_speed"] = df["speed_kmph"].rolling(window, min_periods=1).mean()
    df["rolling_altitude_std"] = df["altitude_km"].rolling(window, min_periods=1).std(ddof=0).fillna(0)
    df["minimum_altitude"] = df["altitude_km"].expanding().min()
    df["maximum_altitude"] = df["altitude_km"].expanding().max()
    dt = df["timestamp"].diff().dt.total_seconds().replace(0, np.nan)
    df["altitude_rate"] = (df["altitude_km"].diff() / dt).fillna(0)
    return df

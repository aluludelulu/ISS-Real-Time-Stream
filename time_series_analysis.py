import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import PROCESSED_DATA, OUTPUT_DIR
from features import load_processed


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_processed(PROCESSED_DATA).set_index("timestamp")

    one_min = df[["altitude_km", "speed_kmph"]].resample("1min").mean()
    one_min["altitude_5min_mean"] = one_min["altitude_km"].rolling(5, min_periods=1).mean()
    one_min["speed_5min_mean"] = one_min["speed_kmph"].rolling(5, min_periods=1).mean()

    # Estimate strongest repeating altitude period in the 60-120 minute band.
    x = one_min["altitude_km"].to_numpy()
    x = x - np.nanmean(x)
    freq = np.fft.rfftfreq(len(x), d=1.0)  # cycles per minute
    power = np.abs(np.fft.rfft(x)) ** 2
    valid = freq > 0
    periods = np.full_like(freq, np.inf, dtype=float)
    periods[valid] = 1.0 / freq[valid]
    band = valid & (periods >= 60) & (periods <= 120)
    dominant_period = float(periods[band][np.argmax(power[band])])

    metrics = {
        "raw_event_count": int(len(df)),
        "one_minute_points": int(len(one_min)),
        "sampling_interval_seconds": 10,
        "altitude_mean_km": float(df["altitude_km"].mean()),
        "altitude_min_km": float(df["altitude_km"].min()),
        "altitude_max_km": float(df["altitude_km"].max()),
        "speed_mean_kmph": float(df["speed_kmph"].mean()),
        "dominant_altitude_period_minutes_60_120_band": dominant_period,
    }
    with open(OUTPUT_DIR / "time_series_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(one_min.index, one_min["altitude_km"], linewidth=0.9, label="1-min altitude")
    ax.plot(one_min.index, one_min["altitude_5min_mean"], linewidth=1.8, label="5-min rolling mean")
    ax.set_title("ISS altitude: resampled series and rolling trend")
    ax.set_xlabel("Time")
    ax.set_ylabel("Altitude (km)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "altitude_time_series.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(one_min.index, one_min["speed_kmph"], linewidth=0.9, label="1-min speed")
    ax.plot(one_min.index, one_min["speed_5min_mean"], linewidth=1.8, label="5-min rolling mean")
    ax.set_title("ISS speed: resampled series and rolling trend")
    ax.set_xlabel("Time")
    ax.set_ylabel("Speed (km/h)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "speed_time_series.png", dpi=180)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

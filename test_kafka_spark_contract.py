"""Checks which do NOT require a running broker or PySpark."""
import unittest
from pathlib import Path

from kafka_producer import build_events
from config import ML_FEATURES, ANOMALY_FEATURES

class ContractTests(unittest.TestCase):
    def test_replay_event_shapes(self):
        df = build_events(limit=100)
        self.assertEqual(len(df), 100)
        self.assertTrue(set(ML_FEATURES + ANOMALY_FEATURES).issubset(df.columns))
        self.assertEqual(df["timestamp"].diff().dropna().dt.total_seconds().unique().tolist(), [10.0])
        self.assertEqual(df["rolling_avg_altitude"].iloc[0], df["altitude_km"].iloc[0])
        self.assertAlmostEqual(df["rolling_avg_altitude"].iloc[1], df["altitude_km"].iloc[:2].mean())
        self.assertFalse(df[list(set(ML_FEATURES + ANOMALY_FEATURES))].isna().any().any())

    def test_model_artifacts(self):
        p = Path(__file__).parent / "models"
        self.assertTrue((p / "next_direction_rf.joblib").exists())
        self.assertTrue((p / "isolation_forest.joblib").exists())

if __name__ == "__main__":
    unittest.main()

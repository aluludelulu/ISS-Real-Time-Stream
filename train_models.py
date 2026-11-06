import json
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    PROCESSED_DATA, MODEL_DIR, OUTPUT_DIR, RANDOM_STATE, TRAIN_RATIO,
    ML_FEATURES, ANOMALY_FEATURES, ANOMALY_CONTAMINATION,
)
from features import load_processed, build_next_direction_target


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_processed(PROCESSED_DATA)

    # -------- Real-time ML classification --------
    ml = build_next_direction_target(df)
    split = int(len(ml) * TRAIN_RATIO)
    train, test = ml.iloc[:split], ml.iloc[split:]

    classifier = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    classifier.fit(train[ML_FEATURES], train["target_up"])
    pred = classifier.predict(test[ML_FEATURES])
    prob = classifier.predict_proba(test[ML_FEATURES])[:, 1]

    ml_metrics = {
        "train_events": int(len(train)),
        "test_events": int(len(test)),
        "accuracy": float(accuracy_score(test["target_up"], pred)),
        "f1": float(f1_score(test["target_up"], pred)),
        "roc_auc": float(roc_auc_score(test["target_up"], prob)),
        "confusion_matrix": confusion_matrix(test["target_up"], pred).tolist(),
        "target": "next altitude direction: 1=up, 0=not up",
        "split": "chronological 70/30",
    }
    joblib.dump(classifier, MODEL_DIR / "next_direction_rf.joblib")

    # -------- Unsupervised anomaly detection --------
    split_a = int(len(df) * TRAIN_RATIO)
    train_a, test_a = df.iloc[:split_a], df.iloc[split_a:]
    anomaly_model = make_pipeline(
        StandardScaler(),
        IsolationForest(
            n_estimators=300,
            contamination=ANOMALY_CONTAMINATION,
            random_state=RANDOM_STATE,
        ),
    )
    anomaly_model.fit(train_a[ANOMALY_FEATURES])
    a_pred = (anomaly_model.predict(test_a[ANOMALY_FEATURES]) == -1).astype(int)
    y_true = test_a["anomaly"].astype(int).to_numpy()
    anomaly_metrics = {
        "train_events": int(len(train_a)),
        "test_events": int(len(test_a)),
        "precision": float(precision_score(y_true, a_pred, zero_division=0)),
        "recall": float(recall_score(y_true, a_pred, zero_division=0)),
        "f1": float(f1_score(y_true, a_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, a_pred).tolist(),
        "features": ANOMALY_FEATURES,
        "contamination": ANOMALY_CONTAMINATION,
        "split": "fit on first 70%; evaluated on last 30%",
    }
    joblib.dump(anomaly_model, MODEL_DIR / "isolation_forest.joblib")

    metrics = {"machine_learning": ml_metrics, "anomaly_detection": anomaly_metrics}
    with open(OUTPUT_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("ML metrics")
    print(json.dumps(ml_metrics, indent=2))
    print("\nAnomaly metrics")
    print(json.dumps(anomaly_metrics, indent=2))
    print("\nClassification report")
    print(classification_report(test["target_up"], pred, digits=4))


if __name__ == "__main__":
    main()

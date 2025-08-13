from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import load_csv, prepare_dataset, save_metadata, train_test_split_time
from .features import FeatureParams, LabelParams


ARTIFACTS_DIR = Path("artifacts/model")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def log_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = sorted(pd.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("market-master-actions")

    feature_params = FeatureParams()
    label_params = LabelParams()

    with mlflow.start_run(run_name="rf_baseline"):
        ds, feature_cols = prepare_dataset("data/raw/sample_ohlcv.csv", feature_params, label_params)
        X_train, y_train, X_test, y_test = train_test_split_time(ds, feature_cols, test_ratio=0.2)

        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

        mlflow.log_params({
            "n_estimators": clf.n_estimators,
            "max_depth": clf.max_depth,
            "min_samples_leaf": clf.min_samples_leaf,
            "random_state": clf.random_state,
            "class_weight": str(clf.class_weight),
            "features": ",".join(feature_cols),
        })

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        mlflow.log_metrics({"accuracy": acc, "f1_macro": f1_macro})

        # Detailed report
        report: Dict[str, Dict[str, float]] = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_text(json.dumps(report, indent=2), "classification_report.json")
        log_confusion_matrix(y_test, y_pred)

        # Save artifacts
        model_path = ARTIFACTS_DIR / "model.pkl"
        joblib.dump(clf, model_path)
        mlflow.log_artifact(model_path.as_posix(), artifact_path="model")

        meta_path = save_metadata(
            ARTIFACTS_DIR,
            feature_columns=feature_cols,
            feature_params=feature_params,
            label_params=label_params,
            dataset_info={
                "train_rows": str(len(X_train)),
                "test_rows": str(len(X_test)),
            },
        )
        mlflow.log_artifact(meta_path.as_posix(), artifact_path="model")

        print(f"Saved model to {model_path}")
        print(f"Metrics: accuracy={acc:.4f}, f1_macro={f1_macro:.4f}")


if __name__ == "__main__":
    main()



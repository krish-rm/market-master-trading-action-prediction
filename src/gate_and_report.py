from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import mlflow
import pandas as pd


MON_DIR = Path("data/monitoring")
OUT_DIR = Path("artifacts/index")


def load_drift_count() -> int:
    drift_csv = MON_DIR / "drift_summary.csv"
    if not drift_csv.exists():
        return 0
    df = pd.read_csv(drift_csv)
    if "drift_flag" not in df.columns:
        return 0
    return int(df["drift_flag"].fillna(False).astype(bool).sum())


def load_macro_f1() -> float:
    qual_json = MON_DIR / "classification_quality.json"
    if not qual_json.exists():
        return 0.0
    rep = json.loads(qual_json.read_text(encoding="utf-8"))
    # Prefer macro avg; fallback to weighted
    macro = rep.get("macro avg", {})
    return float(macro.get("f1-score", 0.0))


def gate(drift_threshold: int, f1_threshold: float) -> Dict[str, Any]:
    drift_features = load_drift_count()
    macro_f1 = load_macro_f1()
    passed = (drift_features <= drift_threshold) and (macro_f1 >= f1_threshold)
    return {
        "drift_features": drift_features,
        "macro_f1": macro_f1,
        "drift_threshold": drift_threshold,
        "f1_threshold": f1_threshold,
        "passed": passed,
    }


def main() -> None:
    # Simple default thresholds for MVP
    summary = gate(drift_threshold=10, f1_threshold=0.20)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "gate_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("market-master-actions")
    with mlflow.start_run(run_name="promotion_gate"):
        mlflow.log_metrics({
            "drift_features": summary["drift_features"],
            "macro_f1": summary["macro_f1"],
        })
        mlflow.log_param("gate_passed", str(summary["passed"]))
        mlflow.log_artifact(out_path.as_posix(), artifact_path="gating")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()



from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import mlflow
import pandas as pd

from .features import FeatureParams, LabelParams, build_dataset


ARTIFACTS_DIR = Path("artifacts/model")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
META_PATH = ARTIFACTS_DIR / "metadata.json"
HISTORY_CSV = Path("data/raw/sample_ohlcv.csv")


def run_inference(open_: float, high: float, low: float, close: float, volume: int) -> Dict[str, Any]:
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    fparams = FeatureParams(**meta.get("feature_params", {}))
    lparams = LabelParams(**meta.get("label_params", {}))
    feature_cols: List[str] = meta["feature_columns"]

    hist = pd.read_csv(HISTORY_CSV, parse_dates=["timestamp"])  # existing training schema
    last_ts = pd.to_datetime(hist["timestamp"].max())
    next_ts = last_ts + pd.Timedelta(minutes=5)
    new = pd.DataFrame(
        [{
            "timestamp": next_ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }]
    )
    df = pd.concat([hist, new], ignore_index=True)

    ds, fc = build_dataset(df, fparams, lparams)
    x = ds.iloc[[-1]][feature_cols]
    proba = model.predict_proba(x)[0]
    classes = list(model.classes_)
    pred_idx = int(proba.argmax())
    action = classes[pred_idx]
    probs = {str(c): float(p) for c, p in zip(classes, proba)}
    return {"action": action, "probabilities": probs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Log a single inference to MLflow")
    parser.add_argument("--open", type=float, required=True)
    parser.add_argument("--high", type=float, required=True)
    parser.add_argument("--low", type=float, required=True)
    parser.add_argument("--close", type=float, required=True)
    parser.add_argument("--volume", type=int, required=True)
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("market-master-actions")
    with mlflow.start_run(run_name="inference"):
        mlflow.log_params({
            "in_open": args.open,
            "in_high": args.high,
            "in_low": args.low,
            "in_close": args.close,
            "in_volume": args.volume,
        })
        out = run_inference(args.open, args.high, args.low, args.close, args.volume)
        # log probabilities as metrics
        for cls, p in out["probabilities"].items():
            mlflow.log_metric(f"proba_{cls}", p)
        mlflow.log_param("pred_action", out["action"])
        # write full output
        out_path = Path("artifacts") / "inference.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        mlflow.log_artifact(out_path.as_posix(), artifact_path="inference")
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()



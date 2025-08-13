from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import mlflow
import pandas as pd

from .features import FeatureParams, LabelParams, build_dataset


MODELS_DIR = Path("artifacts/models")
SELECTED_DIR = Path("artifacts/model")
HISTORY_CSV = Path("data/raw/sample_ohlcv.csv")


def build_latest_row(open_: float, high: float, low: float, close: float, volume: int) -> pd.DataFrame:
    hist = pd.read_csv(HISTORY_CSV, parse_dates=["timestamp"])  # schema: timestamp, ohlcv
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
    return pd.concat([hist, new], ignore_index=True)


def compare_predictions(open_: float, high: float, low: float, close: float, volume: int) -> List[Dict[str, Any]]:
    meta = json.loads((SELECTED_DIR / "metadata.json").read_text(encoding="utf-8"))
    fparams = FeatureParams(**meta.get("feature_params", {}))
    lparams = LabelParams(**meta.get("label_params", {}))
    feature_cols: List[str] = meta["feature_columns"]

    df = build_latest_row(open_, high, low, close, volume)
    ds, fc = build_dataset(df, fparams, lparams)
    x = ds.iloc[[-1]][feature_cols]

    rows: List[Dict[str, Any]] = []
    for pkl in sorted(glob.glob(str(MODELS_DIR / "*.pkl"))):
        name = Path(pkl).stem
        model = joblib.load(pkl)
        proba = model.predict_proba(x)[0]
        classes = list(model.classes_)
        pred_idx = int(proba.argmax())
        action = classes[pred_idx]
        row: Dict[str, Any] = {"model": name, "action": action}
        for j, c in enumerate(classes):
            row[f"p_{c}"] = float(proba[j])
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare predictions from all saved models")
    parser.add_argument("--open", type=float, required=True)
    parser.add_argument("--high", type=float, required=True)
    parser.add_argument("--low", type=float, required=True)
    parser.add_argument("--close", type=float, required=True)
    parser.add_argument("--volume", type=int, required=True)
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("market-master-actions")
    with mlflow.start_run(run_name="predictions_compare"):
        mlflow.log_params({
            "open": args.open,
            "high": args.high,
            "low": args.low,
            "close": args.close,
            "volume": args.volume,
        })
        rows = compare_predictions(args.open, args.high, args.low, args.close, args.volume)
        # save JSON and CSV
        out_dir = MODELS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "predictions_summary.json"
        csv_path = out_dir / "predictions_summary.csv"
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        mlflow.log_artifact(json_path.as_posix(), artifact_path="predictions_compare")
        mlflow.log_artifact(csv_path.as_posix(), artifact_path="predictions_compare")
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()



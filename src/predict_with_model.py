from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

from .features import FeatureParams, LabelParams, build_dataset


def predict_with_model(model_path: str | Path, open_: float, high: float, low: float, close: float, volume: int) -> Dict[str, Any]:
    model = joblib.load(model_path)
    meta_path = Path("artifacts/model/metadata.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fparams = FeatureParams(**meta.get("feature_params", {}))
    lparams = LabelParams(**meta.get("label_params", {}))
    feature_cols: List[str] = meta["feature_columns"]

    hist = pd.read_csv("data/raw/sample_ohlcv.csv", parse_dates=["timestamp"])  # existing training schema
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
    return {"action": action, "probabilities": {str(c): float(proba[i]) for i, c in enumerate(classes)}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with a specific saved model file")
    parser.add_argument("--model", type=str, required=True, help="Path to model .pkl (e.g., artifacts/models/rf.pkl)")
    parser.add_argument("--open", type=float, required=True)
    parser.add_argument("--high", type=float, required=True)
    parser.add_argument("--low", type=float, required=True)
    parser.add_argument("--close", type=float, required=True)
    parser.add_argument("--volume", type=int, required=True)
    args = parser.parse_args()

    out = predict_with_model(args.model, args.open, args.high, args.low, args.close, args.volume)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()



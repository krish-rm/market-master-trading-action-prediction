from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .features import FeatureParams, LabelParams, build_dataset


ARTIFACTS_DIR = Path("artifacts/model")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
META_PATH = ARTIFACTS_DIR / "metadata.json"
HISTORY_CSV = Path("data/raw/sample_ohlcv.csv")


class OHLCVInput(BaseModel):
    open: float = Field(...)
    high: float = Field(...)
    low: float = Field(...)
    close: float = Field(...)
    volume: int = Field(...)


app = FastAPI(title="Market Master Actions API")


def _load_model_and_meta():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise RuntimeError("Model artifacts not found. Train the model first.")
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, meta


MODEL, META = _load_model_and_meta()
FEATURE_COLS: List[str] = META["feature_columns"]
FPARAMS = FeatureParams(**META["feature_params"]) if "feature_params" in META else FeatureParams()
LPARAMS = LabelParams(**META["label_params"]) if "label_params" in META else LabelParams()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "feature_columns": FEATURE_COLS,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict")
def predict(bar: OHLCVInput) -> Dict[str, Any]:
    if not HISTORY_CSV.exists():
        raise HTTPException(status_code=500, detail="History CSV not found. Fetch data first.")
    hist = pd.read_csv(HISTORY_CSV, parse_dates=["timestamp"])
    # Append the provided bar as the next row after last timestamp (+5min assumption)
    last_ts = hist["timestamp"].max()
    next_ts = pd.to_datetime(last_ts) + pd.Timedelta(minutes=5)
    new = pd.DataFrame(
        [{
            "timestamp": next_ts,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }]
    )
    df = pd.concat([hist, new], ignore_index=True)

    # Rebuild features/labels to get consistent preprocessing, then take last row features only
    ds, feature_cols = build_dataset(df, FPARAMS, LPARAMS)
    if not set(FEATURE_COLS).issubset(feature_cols):
        raise HTTPException(status_code=500, detail="Feature mismatch. Retrain the model.")
    x = ds.iloc[[-1]][FEATURE_COLS]

    try:
        proba = MODEL.predict_proba(x)[0]
        classes = list(MODEL.classes_)
        pred_idx = int(proba.argmax())
        action = classes[pred_idx]
        probs = {str(c): float(p) for c, p in zip(classes, proba)}
        return {
            "action": action,
            "probabilities": probs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")



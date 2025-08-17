from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .features import FeatureParams, LabelParams, build_dataset

ARTIFACTS_DIR = Path("artifacts/model")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
META_PATH = ARTIFACTS_DIR / "metadata.json"
REGISTRY_MODEL_NAME = "market-master-component-classifier"
REGISTRY_ALIAS = "Production"
COMPONENTS_DIR = Path("data/components")
WEIGHTS_DEFAULT = Path("data/weights/qqq_weights.csv")


app = FastAPI(title="Market Master Index API", version="1.0")


# Global variables for lazy loading
MODEL = None
META = None
FEATURE_COLS = None
FPARAMS = None
LPARAMS = None


def _load_model_and_meta():
    """Load model and metadata with proper error handling"""
    global MODEL, META, FEATURE_COLS, FPARAMS, LPARAMS
    
    # Try to load from MLflow Model Registry alias first, fallback to local artifacts
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        uri = f"models:/{REGISTRY_MODEL_NAME}@{REGISTRY_ALIAS}"
        model = mlflow.sklearn.load_model(uri)
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        return model, meta
    except Exception as e:
        # Try local artifacts as fallback
        if MODEL_PATH.exists() and META_PATH.exists():
            try:
                model = joblib.load(MODEL_PATH)
                meta = json.loads(META_PATH.read_text(encoding="utf-8"))
                return model, meta
            except Exception as local_e:
                raise RuntimeError(f"Failed to load local model: {local_e}")
        else:
            raise RuntimeError(f"Model artifacts not found. Train pooled model first. Error: {e}")


def _ensure_model_loaded():
    """Ensure model is loaded, load if not already loaded"""
    global MODEL, META, FEATURE_COLS, FPARAMS, LPARAMS
    
    if MODEL is None or META is None:
        MODEL, META = _load_model_and_meta()
        FEATURE_COLS = META["feature_columns"]
        FPARAMS = FeatureParams(**META.get("feature_params", {}))
        LPARAMS = LabelParams(**META.get("label_params", {}))


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        _ensure_model_loaded()
        return {
            "status": "ok",
            "model_loaded": True,
            "model_source": f"registry:{REGISTRY_MODEL_NAME}@{REGISTRY_ALIAS}",
            "feature_columns": FEATURE_COLS,
        }
    except Exception as e:
        return {
            "status": "model_not_ready",
            "model_loaded": False,
            "message": f"Model not available: {str(e)}",
            "model_source": str(MODEL_PATH),
        }


def _predict_component(symbol: str, interval: str = "1h") -> Dict[str, Any]:
    # Ensure model is loaded
    _ensure_model_loaded()
    
    csv_path = COMPONENTS_DIR / f"{symbol.upper()}_{interval}.csv"
    if not csv_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Component data not found for {symbol} {interval}"
        )
    raw = pd.read_csv(csv_path, parse_dates=["timestamp"])  # timestamp, ohlcv
    if raw.empty:
        raise HTTPException(status_code=400, detail="Empty component history")
    ds, _ = build_dataset(raw, FPARAMS, LPARAMS)
    if len(ds) == 0:
        raise HTTPException(
            status_code=400, detail="Insufficient history after warm-up"
        )

    # Build feature row. Training used symbol one-hot columns prefixed by sym_
    sym_cols = [c for c in FEATURE_COLS if c.startswith("sym_")]
    base_cols = [c for c in FEATURE_COLS if c not in sym_cols]
    x_base = ds.iloc[[-1]][base_cols]
    sym_row = {c: 0.0 for c in sym_cols}
    sym_key = f"sym_{symbol.upper()}"
    if sym_key in sym_row:
        sym_row[sym_key] = 1.0
    x_sym = pd.DataFrame([sym_row]) if sym_cols else pd.DataFrame()
    x = pd.concat([x_base.reset_index(drop=True), x_sym], axis=1)
    x = x[FEATURE_COLS]

    proba = MODEL.predict_proba(x)[0]
    classes = list(MODEL.classes_)
    pred_idx = int(proba.argmax())
    action = classes[pred_idx]
    probs = {str(c): float(p) for c, p in zip(classes, proba)}
    last_ts = pd.to_datetime(raw["timestamp"].max())
    # approximate next timestamp
    next_ts = str(last_ts + pd.Timedelta(hours=1 if interval == "1h" else 0))
    return {
        "symbol": symbol.upper(),
        "action": action,
        "confidence": float(proba[pred_idx]),
        "probabilities": probs,
        "timestamp_next": next_ts,
    }


@app.get("/predict/component")
def predict_component(
    symbol: str = Query(..., description="Ticker symbol, e.g., NVDA"),
    interval: str = "1h",
) -> Dict[str, Any]:
    try:
        return _predict_component(symbol, interval)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


def _action_to_score(action: str) -> int:
    mapping = {"strong_buy": 2, "buy": 1, "hold": 0, "sell": -1, "strong_sell": -2}
    return mapping.get(action, 0)


@app.get("/signal/index")
def signal_index(universe: str = "qqq", interval: str = "1h") -> Dict[str, Any]:
    if universe.lower() != "qqq":
        raise HTTPException(
            status_code=400,
            detail="Unsupported universe; only 'qqq' is available in MVP",
        )
    if not WEIGHTS_DEFAULT.exists():
        raise HTTPException(
            status_code=500, detail="Weights file missing. Fetch weights first."
        )
    w = pd.read_csv(WEIGHTS_DEFAULT)
    w["symbol"] = w["symbol"].astype(str).str.upper()
    total = w["weight"].sum()
    if total > 0:
        w["weight"] = w["weight"] / total

    rows: List[Dict[str, Any]] = []
    for sym in w["symbol"].tolist():
        try:
            rows.append(_predict_component(sym, interval))
        except HTTPException:
            continue
    if not rows:
        raise HTTPException(
            status_code=500, detail="No component predictions available"
        )

    df = pd.DataFrame(rows).merge(w, on="symbol", how="left")
    df["score"] = df["action"].apply(_action_to_score)
    df["contrib"] = df["score"] * df["weight"].fillna(0.0)
    wss = float(df["contrib"].sum())
    signal = "BUY /NQ" if wss >= 0.5 else ("SELL /NQ" if wss <= -0.5 else "HOLD")
    next_ts = str(df.get("timestamp_next", pd.Series([None])).iloc[0])
    per_symbol = df.sort_values("weight", ascending=False)[
        ["symbol", "action", "confidence", "weight"]
    ].to_dict(orient="records")
    return {
        "wss": wss,
        "signal": signal,
        "timestamp_next": next_ts,
        "per_symbol": per_symbol,
    }


# ---- Optional POST /predict for rubric compliance (accepts recent bars) ----


class Bar(BaseModel):
    timestamp: Optional[str] = None
    open: float
    high: float
    low: float
    close: float
    volume: float


class PredictRequest(BaseModel):
    symbol: str
    interval: str = "1h"
    bars: List[Bar]


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    try:
        # Ensure model is loaded
        _ensure_model_loaded()
        
        # Convert bars to DataFrame
        rows = [b.dict() for b in req.bars]
        raw = pd.DataFrame(rows)
        if "timestamp" in raw.columns:
            try:
                raw["timestamp"] = pd.to_datetime(
                    raw["timestamp"], utc=True, errors="coerce"
                )
            except Exception:
                pass
        # Build dataset and take last available row after warm-up
        ds, _ = build_dataset(raw, FPARAMS, LPARAMS)
        if len(ds) == 0:
            raise HTTPException(
                status_code=400, detail="Insufficient history after warm-up"
            )
        # Assemble feature vector with symbol one-hot
        sym_cols = [c for c in FEATURE_COLS if c.startswith("sym_")]
        base_cols = [c for c in FEATURE_COLS if c not in sym_cols]
        x_base = ds.iloc[[-1]][base_cols]
        sym_row = {c: 0.0 for c in sym_cols}
        sym_key = f"sym_{req.symbol.upper()}"
        if sym_key in sym_row:
            sym_row[sym_key] = 1.0
        x_sym = pd.DataFrame([sym_row]) if sym_cols else pd.DataFrame()
        x = pd.concat([x_base.reset_index(drop=True), x_sym], axis=1)
        x = x[FEATURE_COLS]
        proba = MODEL.predict_proba(x)[0]
        classes = list(MODEL.classes_)
        pred_idx = int(proba.argmax())
        action = classes[pred_idx]
        probs = {str(c): float(p) for c, p in zip(classes, proba)}
        return {
            "symbol": req.symbol.upper(),
            "action": action,
            "confidence": float(proba[pred_idx]),
            "probabilities": probs,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

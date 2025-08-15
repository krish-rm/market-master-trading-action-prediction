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
META_PATH = ARTIFACTS_DIR / "metadata.json"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
COMPONENTS_DIR = Path("data/components")


def load_meta() -> Dict[str, Any]:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return meta


def load_weights(weights_path: Path) -> pd.DataFrame:
    w = pd.read_csv(weights_path)
    w["symbol"] = w["symbol"].astype(str).str.upper()
    # normalize just in case
    t = w["weight"].sum()
    if t > 0:
        w["weight"] = w["weight"] / t
    return w


def predict_per_symbol(
    model,
    feature_params: FeatureParams,
    label_params: LabelParams,
    feature_cols: List[str],
    symbols: List[str],
    interval: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sym in symbols:
        csv_path = COMPONENTS_DIR / f"{sym}_{interval}.csv"
        if not csv_path.exists():
            continue
        # build features for the symbol
        raw = pd.read_csv(csv_path, parse_dates=["timestamp"])  # timestamp, ohlcv
        # Append no new bar; use the last available to produce features and predict next
        ds, fc = build_dataset(raw, feature_params, label_params)
        if len(ds) == 0:
            continue
        # Split feature columns into base and symbol one-hots,
        # then build row accordingly
        sym_cols = [c for c in feature_cols if c.startswith("sym_")]
        base_cols = [c for c in feature_cols if c not in sym_cols]
        x_base = ds.iloc[[-1]][base_cols]
        # Build one-hot vector for current symbol
        sym_row = {c: 0.0 for c in sym_cols}
        sym_key = f"sym_{sym}"
        if sym_key in sym_row:
            sym_row[sym_key] = 1.0
        x_sym = pd.DataFrame([sym_row]) if sym_cols else pd.DataFrame()
        x = pd.concat([x_base.reset_index(drop=True), x_sym], axis=1)
        # Ensure column order matches training
        x = x[feature_cols]
        proba = model.predict_proba(x)[0]
        classes = list(model.classes_)
        pred_idx = int(proba.argmax())
        action = classes[pred_idx]
        out: Dict[str, Any] = {
            "symbol": sym,
            "action": action,
            "confidence": float(proba[pred_idx]),
        }
        for j, c in enumerate(classes):
            out[f"p_{c}"] = float(proba[j])
        rows.append(out)
    return rows


def action_to_score(action: str) -> int:
    mapping = {
        "strong_buy": 2,
        "buy": 1,
        "hold": 0,
        "sell": -1,
        "strong_sell": -2,
    }
    return mapping.get(action, 0)


def compute_wss(rows: List[Dict[str, Any]], weights_df: pd.DataFrame) -> Dict[str, Any]:
    if not rows:
        return {"wss": 0.0, "n": 0}
    preds = pd.DataFrame(rows)
    df = preds.merge(weights_df, on="symbol", how="left")
    # default small weight if missing
    df["weight"] = df["weight"].fillna(0.0)
    df["score"] = df["action"].apply(action_to_score)
    df["contrib"] = df["score"] * df["weight"]
    wss = float(df["contrib"].sum())
    return {
        "wss": wss,
        "n": int(len(df)),
        "table": df.sort_values("weight", ascending=False).to_dict(orient="records"),
    }


def batch_backtest(
    weights_df: pd.DataFrame,
    model,
    feature_params: FeatureParams,
    label_params: LabelParams,
    feature_cols: List[str],
    interval: str,
) -> List[Dict[str, Any]]:
    """Batch/backtest mode: process all historical data and generate WSS time series"""
    results = []

    # Get all available timestamps from the first symbol
    symbols = list(weights_df["symbol"].unique())
    if not symbols:
        return results

    first_csv = COMPONENTS_DIR / f"{symbols[0]}_{interval}.csv"
    if not first_csv.exists():
        return results

    # Read all timestamps
    first_df = pd.read_csv(first_csv, parse_dates=["timestamp"])
    timestamps = first_df["timestamp"].sort_values().unique()

    print(f"Processing {len(timestamps)} timestamps for backtesting...")

    for i, ts in enumerate(timestamps):
        if i % 100 == 0:  # Progress indicator
            print(f"Processing timestamp {i+1}/{len(timestamps)}: {ts}")

        # Get data up to this timestamp for each symbol
        rows = []
        for sym in symbols:
            csv_path = COMPONENTS_DIR / f"{sym}_{interval}.csv"
            if not csv_path.exists():
                continue

            raw = pd.read_csv(csv_path, parse_dates=["timestamp"])
            # Filter data up to current timestamp
            historical_data = raw[raw["timestamp"] <= ts].copy()

            if len(historical_data) < 20:  # Need minimum history for features
                continue

            # Build features using historical data
            ds, fc = build_dataset(historical_data, feature_params, label_params)
            if len(ds) == 0:
                continue

            # Prepare feature vector for prediction
            sym_cols = [c for c in feature_cols if c.startswith("sym_")]
            base_cols = [c for c in feature_cols if c not in sym_cols]
            x_base = ds.iloc[[-1]][base_cols]

            # Build one-hot vector for current symbol
            sym_row = {c: 0.0 for c in sym_cols}
            sym_key = f"sym_{sym}"
            if sym_key in sym_row:
                sym_row[sym_key] = 1.0
            x_sym = pd.DataFrame([sym_row]) if sym_cols else pd.DataFrame()
            x = pd.concat([x_base.reset_index(drop=True), x_sym], axis=1)
            x = x[feature_cols]

            # Predict
            proba = model.predict_proba(x)[0]
            classes = list(model.classes_)
            pred_idx = int(proba.argmax())
            action = classes[pred_idx]

            out = {
                "symbol": sym,
                "action": action,
                "confidence": float(proba[pred_idx]),
                "timestamp": ts.isoformat(),
            }
            for j, c in enumerate(classes):
                out[f"p_{c}"] = float(proba[j])
            rows.append(out)

        # Compute WSS for this timestamp
        if rows:
            wss_info = compute_wss(rows, weights_df)
            wss = wss_info["wss"]

            # Determine signal
            if wss >= 0.5:
                signal = "BUY /NQ"
            elif wss <= -0.5:
                signal = "SELL /NQ"
            else:
                signal = "HOLD"

            results.append(
                {
                    "timestamp": ts.isoformat(),
                    "wss": wss,
                    "signal": signal,
                    "num_symbols": len(rows),
                    "predictions": rows,
                }
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict per-symbol actions and compute index WSS"
    )
    parser.add_argument("--weights", type=str, default="data/weights/qqq_weights.csv")
    parser.add_argument("--interval", type=str, default="1h", choices=["1h", "5m"])
    parser.add_argument(
        "--signal_buy", type=float, default=0.5, help="WSS >= threshold -> BUY"
    )
    parser.add_argument(
        "--signal_sell", type=float, default=-0.5, help="WSS <= threshold -> SELL"
    )
    parser.add_argument("--batch", action="store_true", help="Run batch/backtest mode")
    args = parser.parse_args()

    meta = load_meta()
    feature_cols: List[str] = meta["feature_columns"]
    fparams = FeatureParams(**meta.get("feature_params", {}))
    lparams = LabelParams(**meta.get("label_params", {}))
    model = joblib.load(MODEL_PATH)

    weights_df = load_weights(Path(args.weights))
    symbols = list(weights_df["symbol"].unique())

    if args.batch:
        # Batch/backtest mode
        print("Running batch/backtest mode...")
        results = batch_backtest(
            weights_df, model, fparams, lparams, feature_cols, args.interval
        )

        # Save results
        out_dir = Path("artifacts/backtest")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save WSS time series
        wss_series = [
            {"timestamp": r["timestamp"], "wss": r["wss"], "signal": r["signal"]}
            for r in results
        ]
        wss_file = out_dir / "wss_time_series.json"
        wss_file.write_text(json.dumps(wss_series, indent=2), encoding="utf-8")

        # Save detailed results
        detailed_file = out_dir / "backtest_results.json"
        detailed_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

        # Log to MLflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("market-master-actions")
        with mlflow.start_run(run_name="batch_backtest"):
            mlflow.log_params(
                {
                    "interval": args.interval,
                    "num_symbols": len(symbols),
                    "num_timestamps": len(results),
                    "mode": "batch",
                }
            )
            mlflow.log_artifact(wss_file.as_posix(), artifact_path="backtest")
            mlflow.log_artifact(detailed_file.as_posix(), artifact_path="backtest")

        print(f"Batch backtest completed. Processed {len(results)} timestamps.")
        print(f"Results saved to: {out_dir}")
        print(f"WSS time series: {wss_file}")

    else:
        # Single prediction mode (existing logic)
        rows = predict_per_symbol(
            model, fparams, lparams, feature_cols, symbols, args.interval
        )
        wss_info = compute_wss(rows, weights_df)
        wss = wss_info["wss"]
        if wss >= args.signal_buy:
            signal = "BUY /NQ"
        elif wss <= args.signal_sell:
            signal = "SELL /NQ"
        else:
            signal = "HOLD"

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("market-master-actions")
        with mlflow.start_run(run_name="index_signal"):
            mlflow.log_params(
                {
                    "interval": args.interval,
                    "signal_buy": args.signal_buy,
                    "signal_sell": args.signal_sell,
                    "num_symbols": len(symbols),
                }
            )
            mlflow.log_metrics({"wss": wss})
            out_dir = Path("artifacts/index")
            out_dir.mkdir(parents=True, exist_ok=True)
            preds_json = out_dir / "per_symbol_predictions.json"
            preds_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            mlflow.log_artifact(preds_json.as_posix(), artifact_path="index_signal")
            wss_json = out_dir / "wss_summary.json"
            wss_json.write_text(
                json.dumps(
                    {"wss": wss, "signal": signal, "detail": wss_info}, indent=2
                ),
                encoding="utf-8",
            )
            mlflow.log_artifact(wss_json.as_posix(), artifact_path="index_signal")
        print(json.dumps({"wss": wss, "signal": signal}, indent=2))


if __name__ == "__main__":
    main()

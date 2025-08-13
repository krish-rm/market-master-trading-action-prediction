from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from .features import FeatureParams, LabelParams, build_dataset
from .data import save_metadata
from .model_candidates import get_candidates  # reuse model candidates


ARTIFACTS_DIR = Path("artifacts/model")
ALL_MODELS_DIR = Path("artifacts/models")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
ALL_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def compute_dynamic_thresholds(abs_r_next: pd.Series, q1: float = 0.40, q2: float = 0.80) -> Tuple[float, float]:
    # Avoid NaNs; compute quantiles on absolute returns
    s = abs_r_next.dropna().abs()
    if len(s) < 10:
        # Fallback to fixed small thresholds
        return 0.001, 0.003
    t1 = float(s.quantile(q1))
    t2 = float(s.quantile(q2))
    if t2 <= t1:
        t2 = t1 * 2.0
    return t1, t2


def load_symbol_df(path: Path, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = symbol
    return df


def build_pooled_feature_dataset(components_dir: Path, feature_params: FeatureParams, label_params: LabelParams) -> Tuple[pd.DataFrame, List[str]]:
    # Build per-symbol features then concatenate to avoid cross-symbol leakage in rolling windows
    pooled: List[pd.DataFrame] = []
    for csv_path in sorted(components_dir.glob("*_1h.csv")):
        symbol = csv_path.stem.replace("_1h", "")
        raw = load_symbol_df(csv_path, symbol)
        ds_sym, feat_cols = build_dataset(raw, feature_params, label_params)
        ds_sym["symbol"] = symbol
        pooled.append(ds_sym)
    if not pooled:
        raise RuntimeError(f"No component CSVs found in {components_dir}")
    ds = pd.concat(pooled, ignore_index=True)
    # One-hot encode symbol and extend feature columns
    sym_dum = pd.get_dummies(ds["symbol"], prefix="sym")
    ds = pd.concat([ds, sym_dum], axis=1)
    feature_columns = feat_cols + list(sym_dum.columns)
    return ds, feature_columns


def time_split(ds: pd.DataFrame, feature_columns: List[str], test_ratio: float = 0.2):
    ds = ds.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(ds) * (1 - test_ratio))
    train_df = ds.iloc[:split_idx].reset_index(drop=True)
    test_df = ds.iloc[split_idx:].reset_index(drop=True)
    X_train = train_df[feature_columns]
    y_train = train_df["action"].astype(str)
    X_test = test_df[feature_columns]
    y_test = test_df["action"].astype(str)
    return X_train, y_train, X_test, y_test


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("market-master-actions")

    components_dir = Path("data/components")
    # Compute dynamic thresholds from pooled absolute next returns
    abs_r_all: List[pd.Series] = []
    for csv_path in components_dir.glob("*_1h.csv"):
        df = pd.read_csv(csv_path)
        r_next = df["close"].shift(-1).sub(df["close"]).div(df["close"])  # simple forward return
        abs_r_all.append(r_next)
    if not abs_r_all:
        raise RuntimeError("No component CSVs available to compute thresholds")
    abs_r_pooled = pd.concat(abs_r_all, ignore_index=True)
    t1, t2 = compute_dynamic_thresholds(abs_r_pooled)

    feature_params = FeatureParams(
        window_small_ma=5,
        window_large_ma=20,
        window_volatility=20,
        window_rsi=14,
        window_atr=14,
        window_bb=20,
    )
    label_params = LabelParams(threshold_t1=t1, threshold_t2=t2)

    with mlflow.start_run(run_name="pooled_model_comparison"):
        # Build pooled dataset
        ds, feature_cols = build_pooled_feature_dataset(components_dir, feature_params, label_params)
        X_train, y_train, X_test, y_test = time_split(ds, feature_cols, test_ratio=0.2)

        mlflow.log_params({
            "label_t1": t1,
            "label_t2": t2,
            "interval": "1h",
            "features": ",".join(feature_cols),
        })

        candidates = get_candidates()
        summary: List[Dict[str, object]] = []

        # Simple training without inner CV for runtime; reuse best grids from earlier if needed
        for name, (base_model, grid) in candidates.items():
            with mlflow.start_run(run_name=f"pooled_{name}", nested=True):
                model = base_model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                f1m = f1_score(y_test, y_pred, average="macro")
                acc = accuracy_score(y_test, y_pred)
                mlflow.log_metrics({"test_f1_macro": f1m, "test_accuracy": acc})
                tmp_path = ALL_MODELS_DIR / f"pooled_{name}.pkl"
                joblib.dump(model, tmp_path)
                summary.append({"name": name, "f1_macro": f1m, "acc": acc, "path": str(tmp_path)})

        # Pick best
        summary.sort(key=lambda r: (r["f1_macro"], r["acc"]), reverse=True)
        best = summary[0]
        best_src = Path(best["path"])  # type: ignore
        final_path = ARTIFACTS_DIR / "model.pkl"
        best_model = joblib.load(best_src)
        joblib.dump(best_model, final_path)
        mlflow.log_param("best_model", best["name"])  # type: ignore
        mlflow.log_metrics({"best_f1_macro": best["f1_macro"], "best_accuracy": best["acc"]})  # type: ignore

        meta_path = save_metadata(
            ARTIFACTS_DIR,
            feature_columns=feature_cols,
            feature_params=feature_params,
            label_params=label_params,
            dataset_info={"rows": str(len(ds)), "interval": "1h", "pooled": "true"},
        )
        mlflow.log_artifact(final_path.as_posix(), artifact_path="pooled_selected")
        mlflow.log_artifact(meta_path.as_posix(), artifact_path="pooled_selected")

        cmp_json = ALL_MODELS_DIR / "pooled_summary.json"
        cmp_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_artifact(cmp_json.as_posix(), artifact_path="pooled_comparison")

        # Register champion model in MLflow Model Registry and set alias to Staging
        try:
            model_name = "market-master-component-classifier"
            # Log model to this run under a distinct artifact path and register
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="champion_model",
                registered_model_name=model_name,
            )
            client = MlflowClient()
            run_id = mlflow.active_run().info.run_id  # type: ignore
            versions = client.search_model_versions(f"name='{model_name}'")
            this_version = None
            for v in versions:
                if v.run_id == run_id:
                    this_version = int(v.version)
                    break
            if this_version is not None:
                client.set_registered_model_alias(model_name, "Staging", this_version)
        except Exception as e:
            # Continue even if registry operations fail (registry is optional in local MVP)
            print(json.dumps({"registry_warning": str(e)}))

        print(json.dumps({"best_model": best["name"], "f1_macro": best["f1_macro"], "accuracy": best["acc"]}, indent=2))  # type: ignore


if __name__ == "__main__":
    main()



from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import classification_report

from .features import FeatureParams, LabelParams, build_dataset


ARTIFACTS_DIR = Path("artifacts/model")
META_PATH = ARTIFACTS_DIR / "metadata.json"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
COMPONENTS_DIR = Path("data/components")
MON_DIR = Path("data/monitoring")


def load_meta() -> Tuple[List[str], FeatureParams, LabelParams]:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    feature_cols: List[str] = meta["feature_columns"]
    fparams = FeatureParams(**meta.get("feature_params", {}))
    lparams = LabelParams(**meta.get("label_params", {}))
    return feature_cols, fparams, lparams


def load_pooled_dataset(feature_params: FeatureParams, label_params: LabelParams) -> Tuple[pd.DataFrame, List[str]]:
    pooled: List[pd.DataFrame] = []
    last_feat_cols: List[str] = []
    for csv_path in sorted(COMPONENTS_DIR.glob("*_1h.csv")):
        raw = pd.read_csv(csv_path)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        ds_sym, feat_cols = build_dataset(raw, feature_params, label_params)
        ds_sym["symbol"] = csv_path.stem.replace("_1h", "")
        pooled.append(ds_sym)
        last_feat_cols = feat_cols
    if not pooled:
        raise RuntimeError("No component CSVs for monitoring")
    ds = pd.concat(pooled, ignore_index=True)
    # one-hot encode symbol to match training
    sym_dum = pd.get_dummies(ds["symbol"], prefix="sym")
    ds = pd.concat([ds, sym_dum], axis=1)
    feature_columns = last_feat_cols + list(sym_dum.columns)
    ds = ds.sort_values("timestamp").reset_index(drop=True)
    return ds, feature_columns


def split_reference_current(ds: pd.DataFrame, ref_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(ds) * ref_ratio)
    ref = ds.iloc[:split_idx].reset_index(drop=True)
    cur = ds.iloc[split_idx:].reset_index(drop=True)
    return ref, cur


def compute_feature_drift(ref: pd.DataFrame, cur: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        # Skip non-numeric columns if any
        if not np.issubdtype(ref[col].dtype, np.number) or not np.issubdtype(cur[col].dtype, np.number):
            continue
        r = ref[col].dropna().values
        c = cur[col].dropna().values
        if len(r) == 0 or len(c) == 0:
            pval = np.nan
            stat = np.nan
        else:
            stat, pval = ks_2samp(r, c)
        rows.append({"feature": col, "ks_stat": float(stat) if stat == stat else None, "p_value": float(pval) if pval == pval else None, "drift_flag": bool(pval is not None and pval < 0.05)})
    return pd.DataFrame(rows)


def plot_top_drift(drift_df: pd.DataFrame, out_path: Path, top_n: int = 10) -> None:
    df = drift_df.dropna().sort_values("ks_stat", ascending=False).head(top_n)
    plt.figure(figsize=(8, 4))
    plt.bar(df["feature"], df["ks_stat"], color="#3b82f6")
    plt.xticks(rotation=45, ha="right")
    plt.title("Top Feature Drift (KS Statistic)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def compute_classification_quality(model, cur: pd.DataFrame, feature_cols: List[str]) -> Dict[str, any]:
    X = cur[feature_cols]
    y = cur["action"].astype(str)
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    return report


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("market-master-actions")

    feature_cols, fparams, lparams = load_meta()
    ds, feat_cols = load_pooled_dataset(fparams, lparams)
    ref, cur = split_reference_current(ds)

    drift_df = compute_feature_drift(ref, cur, feat_cols)
    model = joblib.load(MODEL_PATH)
    quality = compute_classification_quality(model, cur, feat_cols)

    MON_DIR.mkdir(parents=True, exist_ok=True)
    drift_csv = MON_DIR / "drift_summary.csv"
    drift_df.to_csv(drift_csv, index=False)
    drift_plot = MON_DIR / "drift_top.png"
    plot_top_drift(drift_df, drift_plot)
    qual_json = MON_DIR / "classification_quality.json"
    qual_json.write_text(json.dumps(quality, indent=2), encoding="utf-8")

    with mlflow.start_run(run_name="monitoring_report"):
        mlflow.log_artifact(drift_csv.as_posix(), artifact_path="monitoring")
        mlflow.log_artifact(drift_plot.as_posix(), artifact_path="monitoring")
        mlflow.log_artifact(qual_json.as_posix(), artifact_path="monitoring")
        # Simple aggregated metrics
        mlflow.log_metrics({
            "drift_features": int(drift_df["drift_flag"].sum(skipna=True)),
        })
    print(f"Saved drift and classification quality reports to {MON_DIR}")


if __name__ == "__main__":
    main()



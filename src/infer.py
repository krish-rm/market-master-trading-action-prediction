from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from .features import FeatureParams, LabelParams, build_dataset


def run_batch(
    input_csv: str | Path,
    model_path: str | Path = "artifacts/model/model.pkl",
    meta_path: str | Path = "artifacts/model/metadata.json",
    output_csv: str | Path = "data/predictions.csv",
) -> Path:
    model = joblib.load(model_path)
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    fparams = FeatureParams(**meta.get("feature_params", {}))
    lparams = LabelParams(**meta.get("label_params", {}))
    feature_cols = meta["feature_columns"]

    raw = pd.read_csv(input_csv, parse_dates=["timestamp"])  # must match training schema
    ds, fc = build_dataset(raw, fparams, lparams)
    if feature_cols != fc:
        # keep strict alignment to avoid leakage/mismatch
        ds = ds[feature_cols + ["timestamp"]]

    proba = model.predict_proba(ds[feature_cols])
    classes = list(model.classes_)
    preds = proba.argmax(axis=1)
    actions = [classes[i] for i in preds]
    records: list[Dict[str, Any]] = []
    for i, row in ds.iterrows():
        rec = {
            "timestamp": row["timestamp"],
            "action": actions[i],
        }
        rec.update({f"p_{c}": float(proba[i, j]) for j, c in enumerate(classes)})
        records.append(rec)

    out = pd.DataFrame(records)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    p = run_batch("data/raw/sample_ohlcv.csv")
    print(f"Saved predictions to {p}")



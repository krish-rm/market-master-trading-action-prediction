from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .features import FeatureParams, LabelParams, build_dataset


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure timestamp is datetime (timezone-naive okay for modeling)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    return df


def prepare_dataset(
    csv_path: str | Path,
    feature_params: FeatureParams | None = None,
    label_params: LabelParams | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    feature_params = feature_params or FeatureParams()
    label_params = label_params or LabelParams()
    raw_df = load_csv(csv_path)
    ds, feature_columns = build_dataset(raw_df, feature_params, label_params)
    return ds, feature_columns


def train_test_split_time(
    ds: pd.DataFrame,
    feature_columns: List[str],
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    ds = ds.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(ds) * (1 - test_ratio))
    train_df = ds.iloc[:split_idx].reset_index(drop=True)
    test_df = ds.iloc[split_idx:].reset_index(drop=True)
    X_train = train_df[feature_columns]
    y_train = train_df["action"].astype(str)
    X_test = test_df[feature_columns]
    y_test = test_df["action"].astype(str)
    return X_train, y_train, X_test, y_test


def save_metadata(
    output_dir: str | Path,
    *,
    feature_columns: List[str],
    feature_params: FeatureParams,
    label_params: LabelParams,
    dataset_info: Dict[str, str] | None = None,
) -> Path:
    import json
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "feature_columns": feature_columns,
        "feature_params": asdict(feature_params),
        "label_params": asdict(label_params),
    }
    if dataset_info:
        meta["dataset_info"] = dataset_info
    meta_path = output_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta_path



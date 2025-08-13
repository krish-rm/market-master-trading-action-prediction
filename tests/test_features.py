import pandas as pd
import numpy as np

from src.features import FeatureParams, LabelParams, build_dataset


def make_dummy(n: int = 100) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=n, freq="5min")
    base = 100 + np.cumsum(np.random.randn(n) * 0.1)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": base + np.random.randn(n) * 0.01,
            "high": base + np.random.rand(n) * 0.2,
            "low": base - np.random.rand(n) * 0.2,
            "close": base,
            "volume": (np.random.rand(n) * 1e6).astype(int),
        }
    )
    return df


def test_build_dataset_no_nan_features():
    raw = make_dummy(200)
    ds, cols = build_dataset(raw, FeatureParams(), LabelParams())
    assert len(ds) > 0
    assert set(cols).issubset(ds.columns)
    assert not ds[cols].isna().any().any()
    # label exists and no leakage (no NaN)
    assert ds["action"].notna().all()



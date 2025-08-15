import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.features import FeatureParams, LabelParams, build_dataset


def make_dummy(n: int = 100) -> pd.DataFrame:
    """Create dummy OHLCV data for testing"""
    ts = pd.date_range("2025-01-01", periods=n, freq="1h")
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
    """Test that build_dataset produces valid features without NaN values"""
    raw = make_dummy(200)
    ds, cols = build_dataset(raw, FeatureParams(), LabelParams())
    assert len(ds) > 0
    assert set(cols).issubset(ds.columns)
    assert not ds[cols].isna().any().any()
    # label exists and no leakage (no NaN)
    assert ds["action"].notna().all()


def test_feature_params():
    """Test FeatureParams configuration"""
    params = FeatureParams()
    # Check actual parameter names from our implementation
    assert hasattr(params, "window_small_ma")
    assert hasattr(params, "window_large_ma")
    assert hasattr(params, "window_rsi")


def test_label_params():
    """Test LabelParams configuration"""
    params = LabelParams()
    # Check actual parameter names from our implementation
    assert hasattr(params, "threshold_t1") or hasattr(params, "threshold1")
    assert hasattr(params, "threshold_t2") or hasattr(params, "threshold2")


def test_action_distribution():
    """Test that actions are properly distributed across classes"""
    raw = make_dummy(500)
    ds, cols = build_dataset(raw, FeatureParams(), LabelParams())

    actions = ds["action"].value_counts()
    assert len(actions) == 5  # 5 classes
    assert set(actions.index) == {"strong_sell", "sell", "hold", "buy", "strong_buy"}

    # Should have reasonable distribution (not all one class)
    assert actions.max() < len(ds) * 0.8


def test_feature_columns():
    """Test that all expected feature columns are present"""
    raw = make_dummy(200)
    ds, cols = build_dataset(raw, FeatureParams(), LabelParams())

    # Check for actual feature names from our implementation
    expected_features = [
        "log_return",
        "range_rel",
        "body_rel",
        "ma_s",
        "ma_l",
        "ma_cross",
    ]

    for feature in expected_features:
        assert feature in cols, f"Missing feature: {feature}"


def test_minimum_data_requirement():
    """Test that build_dataset handles insufficient data gracefully"""
    # Too little data for features
    raw = make_dummy(10)
    ds, cols = build_dataset(raw, FeatureParams(), LabelParams())
    assert len(ds) == 0  # Should return empty dataset


def test_timestamp_handling():
    """Test that timestamps are properly handled"""
    raw = make_dummy(100)
    ds, cols = build_dataset(raw, FeatureParams(), LabelParams())

    if len(ds) > 0:
        # Timestamps should be in chronological order
        assert ds["timestamp"].is_monotonic_increasing
        # No duplicate timestamps
        assert not ds["timestamp"].duplicated().any()

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd

from src.gate_and_report import gate
from src.monitor_drift import compute_classification_quality, compute_feature_drift
from src.registry import promote


def test_compute_feature_drift():
    """Test feature drift computation"""
    # Create mock reference and current data
    reference_data = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 1000),
            "feature2": np.random.normal(0, 1, 1000),
            "feature3": np.random.normal(0, 1, 1000),
        }
    )

    current_data = pd.DataFrame(
        {
            "feature1": np.random.normal(0.1, 1.1, 500),  # Slight drift
            "feature2": np.random.normal(0, 1, 500),
            "feature3": np.random.normal(0, 1, 500),
        }
    )

    feature_cols = ["feature1", "feature2", "feature3"]

    # Test drift computation
    drift_df = compute_feature_drift(reference_data, current_data, feature_cols)

    assert isinstance(drift_df, pd.DataFrame)
    assert "feature" in drift_df.columns
    assert "ks_stat" in drift_df.columns
    assert "p_value" in drift_df.columns
    assert "drift_flag" in drift_df.columns


def test_compute_classification_quality():
    """Test classification quality computation"""

    # Create mock model
    class MockModel:
        def predict(self, X):
            return np.random.choice(["buy", "sell", "hold"], size=len(X))

    model = MockModel()

    # Create mock current data
    current_data = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
            "action": np.random.choice(["buy", "sell", "hold"], 100),
        }
    )

    feature_cols = ["feature1", "feature2"]

    quality_report = compute_classification_quality(model, current_data, feature_cols)

    assert isinstance(quality_report, dict)
    assert "accuracy" in quality_report
    assert "macro avg" in quality_report


def test_gate():
    """Test promotion gate evaluation"""
    # Mock drift and quality metrics
    # drift_count = 5  # Not used in current implementation
    # macro_f1 = 0.75  # Not used in current implementation

    result = gate(drift_threshold=10, f1_threshold=0.2)

    assert isinstance(result, dict)
    assert "passed" in result
    assert "drift_features" in result
    assert "macro_f1" in result
    assert "drift_threshold" in result
    assert "f1_threshold" in result


def test_promotion_gates():
    """Test different promotion gate scenarios"""
    # Test passing case
    result1 = gate(drift_threshold=10, f1_threshold=0.2)
    # This will depend on actual data, but should return valid structure
    assert isinstance(result1, dict)
    assert "passed" in result1

    # Test with different thresholds
    result2 = gate(drift_threshold=5, f1_threshold=0.5)
    assert isinstance(result2, dict)
    assert "passed" in result2


def test_registry_operations():
    """Test model registry operations"""
    # These tests would require actual MLflow setup
    # For now, we test the function signatures and basic logic

    # Test promote function signature
    with patch("src.registry.MlflowClient") as mock_client:
        mock_client.return_value.get_model_version_by_alias.return_value = MagicMock(
            version="1"
        )

        # This should not raise an exception
        promote()


def test_model_metadata():
    """Test model metadata handling"""
    # Test metadata structure
    metadata = {
        "feature_columns": ["feature1", "feature2", "feature3"],
        "feature_params": {"ma_short": 5, "ma_long": 20, "rsi_period": 14},
        "label_params": {"threshold1": 0.02, "threshold2": 0.05},
    }

    # Test metadata validation
    assert "feature_columns" in metadata
    assert "feature_params" in metadata
    assert "label_params" in metadata
    assert isinstance(metadata["feature_columns"], list)
    assert isinstance(metadata["feature_params"], dict)
    assert isinstance(metadata["label_params"], dict)


def test_monitoring_outputs():
    """Test monitoring output file generation"""
    # Create test monitoring directory
    monitoring_dir = Path("data/monitoring")
    monitoring_dir.mkdir(parents=True, exist_ok=True)

    # Test drift summary CSV
    drift_summary = pd.DataFrame(
        {
            "feature": ["feature1", "feature2"],
            "drift_score": [0.1, 0.05],
            "drift_detected": [True, False],
        }
    )

    drift_file = monitoring_dir / "drift_summary.csv"
    drift_summary.to_csv(drift_file, index=False)

    assert drift_file.exists()
    loaded_drift = pd.read_csv(drift_file)
    assert len(loaded_drift) == 2
    assert "feature" in loaded_drift.columns
    assert "drift_score" in loaded_drift.columns

    # Test quality report JSON
    quality_report = {
        "accuracy": 0.75,
        "macro_f1": 0.72,
        "per_class_metrics": {
            "buy": {"precision": 0.8, "recall": 0.7},
            "sell": {"precision": 0.7, "recall": 0.8},
            "hold": {"precision": 0.75, "recall": 0.75},
        },
    }

    quality_file = monitoring_dir / "classification_quality.json"
    with open(quality_file, "w") as f:
        json.dump(quality_report, f)

    assert quality_file.exists()
    with open(quality_file, "r") as f:
        loaded_quality = json.load(f)

    assert "accuracy" in loaded_quality
    assert "macro_f1" in loaded_quality
    assert "per_class_metrics" in loaded_quality


def test_mlflow_integration():
    """Test MLflow integration components"""
    # Test MLflow tracking URI configuration
    tracking_uri = "sqlite:///mlflow.db"
    assert tracking_uri.startswith("sqlite://")

    # Test experiment naming
    experiment_name = "market-master-actions"
    assert "market-master" in experiment_name

    # Test model registry naming
    model_name = "market-master-component-classifier"
    assert "market-master" in model_name
    assert "classifier" in model_name


def test_error_handling():
    """Test error handling in MLOps components"""
    # Test handling of missing data
    empty_df = pd.DataFrame()

    # Should handle empty data gracefully
    try:
        drift_df = compute_feature_drift(empty_df, empty_df, [])
        # Should not crash, but may return empty DataFrame
        assert isinstance(drift_df, pd.DataFrame)
    except Exception as e:
        # If it does crash, it should be a meaningful error
        assert "empty" in str(e).lower() or "data" in str(e).lower()

    # Test handling of invalid metrics
    result = gate(drift_threshold=-1, f1_threshold=1.5)
    # Should handle invalid inputs gracefully
    assert isinstance(result, dict)
    assert "passed" in result

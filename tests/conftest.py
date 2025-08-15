import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def sample_ohlcv_data():
    """Create sample OHLCV data for testing"""
    np.random.seed(42)  # For reproducible tests

    n_samples = 100
    timestamps = pd.date_range("2025-01-01", periods=n_samples, freq="1h")

    base_price = 100
    prices = base_price + np.cumsum(np.random.randn(n_samples) * 0.1)

    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices + np.random.randn(n_samples) * 0.01,
            "high": prices + np.random.rand(n_samples) * 0.2,
            "low": prices - np.random.rand(n_samples) * 0.2,
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, n_samples),
        }
    )

    return data


@pytest.fixture(scope="session")
def sample_weights():
    """Create sample weights data for testing"""
    weights = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
            "weight": [0.25, 0.20, 0.15, 0.20, 0.20],
        }
    )
    return weights


@pytest.fixture(scope="session")
def mock_model():
    """Create a mock model for testing"""

    class MockModel:
        def __init__(self):
            self.classes_ = np.array(
                ["strong_sell", "sell", "hold", "buy", "strong_buy"]
            )

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.random.choice(self.classes_, size=len(X))

        def predict_proba(self, X):
            # Return random probabilities that sum to 1
            probs = np.random.rand(len(X), len(self.classes_))
            return probs / probs.sum(axis=1, keepdims=True)

    return MockModel()


@pytest.fixture(scope="function")
def clean_artifacts():
    """Clean up artifacts before and after each test"""
    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    artifacts_dir.mkdir(exist_ok=True)

    yield

    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)


@pytest.fixture(scope="function")
def clean_data():
    """Clean up data directories before and after each test"""
    data_dirs = ["data/components", "data/monitoring"]

    # Clean up before test
    for dir_path in data_dirs:
        path = Path(dir_path)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    yield

    # Clean up after test
    for dir_path in data_dirs:
        path = Path(dir_path)
        if path.exists():
            shutil.rmtree(path)


@pytest.fixture(scope="session")
def test_config():
    """Test configuration parameters"""
    return {
        "interval": "1h",
        "days": 7,
        "max_symbols": 3,
        "drift_threshold": 10,
        "f1_threshold": 0.2,
    }

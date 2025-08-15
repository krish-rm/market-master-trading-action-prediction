import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data import prepare_dataset
from src.fetch_symbol import fetch_symbol, save_csv
from src.fetch_weights_qqq import fetch_qqq_holdings
from src.model_candidates import get_candidates
from src.predict_index import action_to_score, compute_wss


def test_fetch_qqq_weights():
    """Test QQQ weights fetching with fallback"""
    with patch("src.fetch_weights_qqq.requests.get") as mock_get:
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = """
        <table>
            <tr><td>AAPL</td><td>10.5%</td></tr>
            <tr><td>MSFT</td><td>9.2%</td></tr>
        </table>
        """
        mock_get.return_value = mock_response

        weights = fetch_qqq_holdings()
        assert isinstance(weights, pd.DataFrame)
        assert "symbol" in weights.columns
        assert "weight" in weights.columns
        assert len(weights) > 0


def test_fetch_symbol():
    """Test symbol data fetching"""
    with patch("src.fetch_symbol.yf.download") as mock_download:
        # Mock successful download
        mock_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [102, 103, 104],
                "Low": [99, 100, 101],
                "Close": [101, 102, 103],
                "Volume": [1000000, 1100000, 1200000],
            },
            index=pd.date_range("2025-01-01", periods=3, freq="1h"),
        )

        mock_download.return_value = mock_data

        df = fetch_symbol("AAPL", "1h", 7)
        assert isinstance(df, pd.DataFrame)
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns


def test_save_csv():
    """Test CSV saving functionality"""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="1h"),
            "open": [100, 101, 102],
            "close": [101, 102, 103],
        }
    )

    test_path = Path("test_output.csv")
    save_csv(df, test_path)

    assert test_path.exists()
    loaded_df = pd.read_csv(test_path)
    assert len(loaded_df) == 3

    # Cleanup
    test_path.unlink()


def test_prepare_dataset():
    """Test dataset preparation"""
    # Create test CSV file
    test_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=50, freq="1h"),
            "open": np.random.randn(50) + 100,
            "high": np.random.randn(50) + 102,
            "low": np.random.randn(50) + 98,
            "close": np.random.randn(50) + 100,
            "volume": np.random.randint(1000000, 5000000, 50),
        }
    )

    test_path = Path("test_data.csv")
    test_data.to_csv(test_path, index=False)

    try:
        ds, feature_cols = prepare_dataset(test_path)
        assert isinstance(ds, pd.DataFrame)
        assert isinstance(feature_cols, list)
        assert len(feature_cols) > 0
        assert "action" in ds.columns
    finally:
        test_path.unlink()


def test_get_candidates():
    """Test model candidates generation"""
    candidates = get_candidates()
    assert len(candidates) > 0

    # Check that each candidate has required attributes
    for name, (model, params) in candidates.items():
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")
        assert isinstance(params, dict)


def test_action_to_score():
    """Test action to score mapping"""
    assert action_to_score("strong_buy") == 2
    assert action_to_score("buy") == 1
    assert action_to_score("hold") == 0
    assert action_to_score("sell") == -1
    assert action_to_score("strong_sell") == -2
    assert action_to_score("invalid") == 0  # Default case


def test_compute_wss():
    """Test Weighted Sentiment Score computation"""
    # Mock predictions
    rows = [
        {"symbol": "AAPL", "action": "buy", "confidence": 0.8},
        {"symbol": "MSFT", "action": "hold", "confidence": 0.6},
        {"symbol": "NVDA", "action": "strong_buy", "confidence": 0.9},
    ]

    # Mock weights
    weights_df = pd.DataFrame(
        {"symbol": ["AAPL", "MSFT", "NVDA"], "weight": [0.4, 0.35, 0.25]}
    )

    result = compute_wss(rows, weights_df)

    assert "wss" in result
    assert "n" in result
    assert "table" in result
    assert isinstance(result["wss"], float)
    assert result["n"] == 3


def test_pipeline_integration():
    """Test basic pipeline integration"""
    # This test verifies that the main components can work together
    # without actually running the full pipeline

    # Test that we can create a minimal dataset
    # df = pd.DataFrame(  # Not used in current test
    #     {
    #         "timestamp": pd.date_range("2025-01-01", periods=50, freq="1h"),
    #         "open": np.random.randn(50) + 100,
    #         "high": np.random.randn(50) + 102,
    #         "low": np.random.randn(50) + 98,
    #         "close": np.random.randn(50) + 100,
    #         "volume": np.random.randint(1000000, 5000000, 50),
    #     }
    # )

    # Test that we can get model candidates
    candidates = get_candidates()
    assert len(candidates) > 0

    # Test that we can compute WSS
    rows = [{"symbol": "TEST", "action": "hold", "confidence": 0.5}]
    weights_df = pd.DataFrame({"symbol": ["TEST"], "weight": [1.0]})
    wss_result = compute_wss(rows, weights_df)
    assert "wss" in wss_result

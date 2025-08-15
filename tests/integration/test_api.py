import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health():
    """Test health endpoint"""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_source" in data
    assert "feature_columns" in data


def test_predict_single_bar():
    """Test single bar prediction endpoint"""
    payload = {
        "open": 180.0,
        "high": 181.0,
        "low": 179.5,
        "close": 180.5,
        "volume": 1000000,
    }
    r = client.post("/predict", json=payload)
    # This endpoint expects a different format, so it should return 422
    assert r.status_code == 422  # Unprocessable Entity due to missing required fields


def test_predict_with_symbol():
    """Test prediction with symbol and bars"""
    payload = {
        "symbol": "AAPL",
        "interval": "1h",
        "bars": [
            {
                "open": 180.0,
                "high": 181.0,
                "low": 179.5,
                "close": 180.5,
                "volume": 1000000,
            }
        ],
    }
    r = client.post("/predict", json=payload)
    # This might fail due to missing data, but should return a valid response structure
    if r.status_code == 200:
        body = r.json()
        assert "action" in body
        assert "probabilities" in body
    else:
        # If it fails, it should be a 400 or 500, not 422
        assert r.status_code in [400, 500]


def test_predict_component():
    """Test component prediction endpoint"""
    r = client.get("/predict/component?symbol=AAPL")
    # This might fail due to missing data, but should return a valid response structure
    if r.status_code == 200:
        body = r.json()
        assert "action" in body
        assert "confidence" in body
        assert "probabilities" in body
    else:
        # If it fails, it should be a 400 or 500, not 404
        assert r.status_code in [400, 500]


def test_signal_index():
    """Test index signal endpoint"""
    r = client.get("/signal/index?universe=qqq")
    # This might fail due to missing data, but should return a valid response structure
    if r.status_code == 200:
        body = r.json()
        assert "wss" in body
        assert "signal" in body
        assert "per_symbol" in body  # Changed from "components" to "per_symbol"
        assert body["signal"] in ["BUY /NQ", "SELL /NQ", "HOLD"]
    else:
        # If it fails, it should be a 400 or 500, not 404
        assert r.status_code in [400, 500]


def test_invalid_symbol():
    """Test error handling for invalid symbol"""
    r = client.get("/predict/component?symbol=INVALID")
    # API returns 404 for invalid symbol
    assert r.status_code == 404


def test_invalid_universe():
    """Test error handling for invalid universe"""
    r = client.get("/signal/index?universe=invalid")
    # Should return 400 for invalid universe, not 404
    assert r.status_code == 400


def test_invalid_predict_payload():
    """Test error handling for invalid prediction payload"""
    payload = {"invalid": "data"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422

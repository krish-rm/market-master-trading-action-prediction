from fastapi.testclient import TestClient
from src.api import app


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


def test_predict_smoke():
    payload = {
        "open": 180.0,
        "high": 181.0,
        "low": 179.5,
        "close": 180.5,
        "volume": 1000000,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "action" in body and "probabilities" in body



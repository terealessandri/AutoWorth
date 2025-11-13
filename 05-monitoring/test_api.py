"""External API tests for the running FastAPI NYC Taxi Duration service.

Requires the server already running (e.g. `python app.py` showing Uvicorn on port 9696).

These are deployment-level tests that issue real HTTP requests instead of using FastAPI's in-process TestClient.
"""
import requests

BASE_URL = "http://localhost:9696"


def test_health_endpoint():
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} body={resp.text}"
    data = resp.json()
    assert data.get("status") == "ok"
    assert isinstance(data.get("run_id"), str) and len(data["run_id"]) > 5


def test_predict_endpoint():
    payload = {
        "PULocationID": 138,
        "DOLocationID": 236,
        "trip_distance": 2.5,
    }
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} body={resp.text}"
    data = resp.json()

    # Validate structure
    assert "duration" in data and "model_version" in data
    # Sanity checks
    assert isinstance(data["duration"], float)
    assert data["duration"] > 0
    assert isinstance(data["model_version"], str) and len(data["model_version"]) > 5

"""
Unit tests for FastAPI endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app, state


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "version" in data


def test_api_dataset_audit(client):
    response = client.post("/dataset/audit")
    assert response.status_code == 200
    data = response.json()
    assert "total_rows" in data
    assert data["dataset_name"] == "CICIDS2017"


def test_api_train_and_predict(client):
    train_res = client.post("/train", json={"sample_size": 2000, "random_seed": 42})
    assert train_res.status_code == 200
    train_data = train_res.json()
    assert train_data["status"] == "TRAINED"

    # Single flow prediction
    dummy_features = [0.0] * train_data["feature_count"]
    pred_res = client.post("/predict", json={"features": dummy_features})
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert "prediction" in pred_data
    assert "risk_score" in pred_data
    assert "evidence" in pred_data


def test_api_hierarchical_risk_endpoints(client):
    # Check node risk endpoint
    nodes_res = client.get("/risk/nodes")
    assert nodes_res.status_code == 200
    assert isinstance(nodes_res.json(), list)

    # Check clusters endpoint
    clusters_res = client.get("/risk/clusters")
    assert clusters_res.status_code == 200
    assert isinstance(clusters_res.json(), dict)

    # Check global risk endpoint
    global_res = client.get("/risk/global")
    assert global_res.status_code == 200
    assert "global_risk" in global_res.json()


def test_api_alerts_and_stream_status(client):
    alerts_res = client.get("/alerts")
    assert alerts_res.status_code == 200
    assert "action_counts" in alerts_res.json()

    status_res = client.get("/stream/status")
    assert status_res.status_code == 200
    assert "is_streaming" in status_res.json()

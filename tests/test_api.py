"""Tests for the FastAPI REST API."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = np.array([-8.5])
    return model


@pytest.fixture
def mock_artifacts(mock_model):
    return {
        "model": mock_model,
        "scaler": None,
        "feature_names": ["miss_distance", "relative_speed", "time_to_tca"],
        "model_name": "XGBoost",
    }


@pytest.fixture
def app(mock_artifacts):
    from orbital_sentinel.api.app import create_app
    application = create_app()
    application.state.model_artifacts = mock_artifacts
    import time
    application.state.start_time = time.time()
    return application


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["version"] == "1.0.0"


def test_health_no_model():
    from orbital_sentinel.api.app import create_app
    from fastapi.testclient import TestClient
    import time

    app = create_app()
    app.state.model_artifacts = None
    app.state.start_time = time.time()
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is False


def test_status_endpoint(client):
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "model" in data


def test_models_list_endpoint(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 200


def test_model_info_endpoint(client):
    response = client.get("/api/v1/models/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "XGBoost"
    assert data["feature_count"] == 3


def test_predict_requires_model():
    from orbital_sentinel.api.app import create_app
    from fastapi.testclient import TestClient
    import time

    app = create_app()
    app.state.model_artifacts = None
    app.state.start_time = time.time()
    client = TestClient(app)

    response = client.post("/api/v1/predict", json={"miss_distance": 500})
    assert response.status_code == 503


def test_sample_endpoint(client):
    with patch("orbital_sentinel.ingestion.dataset_loader.load_esa_kelvins") as mock_load:
        import pandas as pd
        mock_df = pd.DataFrame({
            "event_id": [1, 2],
            "risk": [-8.0, -3.0],
            "miss_distance": [500, 100],
        })
        mock_load.return_value = mock_df

        response = client.get("/api/v1/cdm/sample")
        assert response.status_code == 200
        data = response.json()
        assert "samples" in data
        assert data["total_rows"] == 2

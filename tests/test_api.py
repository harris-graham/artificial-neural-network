"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.api.main import app
from src.api.schemas import PredictionRequest, TrainRequest


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_model_service():
    """Mock model service."""
    service = Mock()
    service.get_health.return_value = {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00",
        "version": "0.1.0",
        "models_loaded": 0,
    }
    return service


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    @patch("src.api.main.model_service")
    def test_health_check(self, mock_service, client):
        """Test health check endpoint."""
        mock_service.get_health.return_value = {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00",
            "version": "0.1.0",
            "models_loaded": 0,
        }
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @patch("src.api.main.model_service", None)
    def test_health_check_uninitialized(self, client):
        """Test health check when service is not initialized."""
        response = client.get("/health")
        assert response.status_code == 503


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    @patch("src.api.main.model_service")
    def test_predict_success(self, mock_service, client):
        """Test successful prediction."""
        mock_service.predict.return_value = {
            "predictions": [0.5, 0.75, 0.25],
            "true_values": [0.5, 0.75, 0.25],
            "model_id": "test-id",
            "model_type": "attention_lstm",
            "timestamp": "2024-01-01T00:00:00",
            "metrics": {"mse": 0.001, "mae": 0.01},
        }

        request = PredictionRequest(A=3.5, steps=3)
        response = client.post("/predict", json=request.dict())
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 3

    @patch("src.api.main.model_service")
    def test_predict_no_models(self, mock_service, client):
        """Test prediction when no models are available."""
        mock_service.predict.side_effect = ValueError("No models available")

        request = PredictionRequest(A=3.5, steps=3)
        response = client.post("/predict", json=request.dict())
        assert response.status_code == 400

    @patch("src.api.main.model_service", None)
    def test_predict_uninitialized(self, client):
        """Test prediction when service is not initialized."""
        request = PredictionRequest(A=3.5, steps=3)
        response = client.post("/predict", json=request.dict())
        assert response.status_code == 503


class TestTrainEndpoint:
    """Tests for training endpoint."""

    @patch("src.api.main.model_service")
    def test_train_success(self, mock_service, client):
        """Test successful training."""
        mock_service.train_model.return_value = {
            "model_id": "test-id",
            "model_type": "attention_lstm",
            "status": "completed",
            "timestamp": "2024-01-01T00:00:00",
            "training_time": 45.0,
            "epochs_trained": 50,
            "metrics": {"final_loss": 0.001, "final_mae": 0.01},
            "hyperparameters": {"units": 128},
        }

        request = TrainRequest(A=3.5, model_type="attention_lstm", epochs=10)
        response = client.post("/train", json=request.dict())
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "model_id" in data

    @patch("src.api.main.model_service")
    def test_train_invalid_hyperparameters(self, mock_service, client):
        """Test training with invalid hyperparameters."""
        request = TrainRequest(
            A=3.5,
            model_type="attention_lstm",
            hyperparameters={"invalid_param": 1},
        )
        response = client.post("/train", json=request.dict())
        assert response.status_code == 422  # Validation error


class TestListModelsEndpoint:
    """Tests for list models endpoint."""

    @patch("src.api.main.model_service")
    def test_list_models(self, mock_service, client):
        """Test listing models."""
        mock_service.list_models.return_value = [
            {
                "id": "model-1",
                "model_type": "attention_lstm",
                "created_at": "2024-01-01T00:00:00",
                "config": {"units": 128},
                "metrics": {"loss": 0.001},
            }
        ]

        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "model-1"


class TestGetModelEndpoint:
    """Tests for get model endpoint."""

    @patch("src.api.main.model_service")
    def test_get_model_success(self, mock_service, client):
        """Test getting a specific model."""
        mock_service.get_model_info.return_value = {
            "model_id": "test-id",
            "model_type": "attention_lstm",
            "created_at": "2024-01-01T00:00:00",
            "config": {"units": 128},
            "metrics": {"loss": 0.001},
        }

        response = client.get("/models/test-id")
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "test-id"

    @patch("src.api.main.model_service")
    def test_get_model_not_found(self, mock_service, client):
        """Test getting a non-existent model."""
        mock_service.get_model_info.return_value = None

        response = client.get("/models/nonexistent-id")
        assert response.status_code == 404


class TestDeleteModelEndpoint:
    """Tests for delete model endpoint."""

    @patch("src.api.main.model_service")
    def test_delete_model_success(self, mock_service, client):
        """Test deleting a model."""
        mock_service.delete_model.return_value = True

        response = client.delete("/models/test-id")
        assert response.status_code == 200
        data = response.json()
        assert "deleted successfully" in data["message"]

    @patch("src.api.main.model_service")
    def test_delete_model_not_found(self, mock_service, client):
        """Test deleting a non-existent model."""
        mock_service.delete_model.return_value = False

        response = client.delete("/models/nonexistent-id")
        assert response.status_code == 404

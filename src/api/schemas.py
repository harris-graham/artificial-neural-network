"""Pydantic schemas for API request/response validation."""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """Request schema for predictions."""

    A: float = Field(..., description="Logistic map parameter A", ge=0, le=4)
    steps: int = Field(default=10, description="Number of steps to predict", ge=1, le=100)
    model_id: Optional[str] = Field(None, description="Specific model ID to use")
    window_size: int = Field(default=1, description="Sliding window size", ge=1, le=10)

    class Config:
        json_schema_extra = {
            "example": {
                "A": 3.5,
                "steps": 20,
                "model_id": None,
                "window_size": 1,
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for predictions."""

    predictions: List[float] = Field(..., description="Predicted values")
    true_values: Optional[List[float]] = Field(None, description="True values for comparison")
    model_id: str = Field(..., description="ID of the model used")
    model_type: str = Field(..., description="Type of model used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    confidence: Optional[float] = Field(None, description="Prediction confidence score")
    metrics: Optional[Dict[str, float]] = Field(None, description="Performance metrics")

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [0.875, 0.391, 0.835, 0.487],
                "true_values": [0.875, 0.391, 0.835, 0.487],
                "model_id": "uuid-string",
                "model_type": "attention_lstm",
                "timestamp": "2024-01-01T00:00:00",
                "confidence": 0.95,
                "metrics": {"mse": 0.001, "mae": 0.02},
            }
        }


class TrainRequest(BaseModel):
    """Request schema for training a model."""

    model_type: str = Field(
        default="attention_lstm",
        description="Type of model to train",
        regex="^(lstm|attention_lstm|gru|attention_gru|transformer)$",
    )
    A: float = Field(..., description="Logistic map parameter A", ge=0, le=4)
    hyperparameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom hyperparameters for the model",
    )
    epochs: int = Field(default=100, description="Number of training epochs", ge=1, le=1000)
    batch_size: int = Field(default=32, description="Training batch size", ge=1, le=256)
    window_size: int = Field(default=1, description="Sliding window size", ge=1, le=10)
    train_ratio: float = Field(default=0.8, description="Training data ratio", ge=0.5, le=0.9)
    val_ratio: float = Field(default=0.1, description="Validation data ratio", ge=0.05, le=0.3)

    @validator("hyperparameters")
    def validate_hyperparameters(cls, v):
        if v is not None:
            allowed_keys = {
                "num_layers",
                "units",
                "dropout_rate",
                "use_bidirectional",
                "attention_type",
                "learning_rate",
            }
            invalid_keys = set(v.keys()) - allowed_keys
            if invalid_keys:
                raise ValueError(f"Invalid hyperparameters: {invalid_keys}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "attention_lstm",
                "A": 3.5,
                "hyperparameters": {
                    "num_layers": 2,
                    "units": 128,
                    "dropout_rate": 0.2,
                    "use_bidirectional": True,
                    "attention_type": "bahdanau",
                    "learning_rate": 0.001,
                },
                "epochs": 100,
                "batch_size": 32,
                "window_size": 1,
                "train_ratio": 0.8,
                "val_ratio": 0.1,
            }
        }


class TrainResponse(BaseModel):
    """Response schema for training completion."""

    model_id: str = Field(..., description="ID of the trained model")
    model_type: str = Field(..., description="Type of model trained")
    status: str = Field(..., description="Training status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Training completion timestamp")
    training_time: float = Field(..., description="Training time in seconds")
    epochs_trained: int = Field(..., description="Number of epochs actually trained")
    metrics: Dict[str, float] = Field(..., description="Training metrics")
    hyperparameters: Dict[str, Any] = Field(..., description="Hyperparameters used")

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "uuid-string",
                "model_type": "attention_lstm",
                "status": "completed",
                "timestamp": "2024-01-01T00:00:00",
                "training_time": 45.2,
                "epochs_trained": 50,
                "metrics": {
                    "final_loss": 0.001,
                    "final_mae": 0.02,
                    "final_val_loss": 0.0015,
                    "final_val_mae": 0.025,
                },
                "hyperparameters": {
                    "num_layers": 2,
                    "units": 128,
                    "dropout_rate": 0.2,
                },
            }
        }


class ModelInfo(BaseModel):
    """Schema for model information."""

    model_id: str = Field(..., description="Model ID")
    model_type: str = Field(..., description="Model type")
    created_at: str = Field(..., description="Model creation timestamp")
    config: Dict[str, Any] = Field(..., description="Model configuration")
    metrics: Dict[str, float] = Field(..., description="Model performance metrics")

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "uuid-string",
                "model_type": "attention_lstm",
                "created_at": "2024-01-01T00:00:00",
                "config": {
                    "num_layers": 2,
                    "units": 128,
                    "dropout_rate": 0.2,
                },
                "metrics": {
                    "final_loss": 0.001,
                    "final_mae": 0.02,
                },
            }
        }


class InterpretRequest(BaseModel):
    """Request schema for model interpretation."""

    A: float = Field(..., description="Logistic map parameter A", ge=0, le=4)
    model_id: str = Field(..., description="Model ID to interpret")
    method: str = Field(
        default="shap",
        description="Interpretation method",
        regex="^(shap|lime|attention)$",
    )
    num_samples: int = Field(default=10, description="Number of samples to analyze", ge=1, le=100)
    window_size: int = Field(default=1, description="Sliding window size", ge=1, le=10)

    class Config:
        json_schema_extra = {
            "example": {
                "A": 3.5,
                "model_id": "uuid-string",
                "method": "shap",
                "num_samples": 10,
                "window_size": 1,
            }
        }


class InterpretResponse(BaseModel):
    """Response schema for model interpretation."""

    model_id: str = Field(..., description="Model ID")
    method: str = Field(..., description="Interpretation method used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Interpretation timestamp")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance scores")
    shap_values: Optional[List[List[float]]] = Field(None, description="SHAP values (if method=shap)")
    attention_weights: Optional[List[List[float]]] = Field(
        None, description="Attention weights (if method=attention)"
    )
    explanation_dir: Optional[str] = Field(None, description="Directory with visualization plots")

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "uuid-string",
                "method": "shap",
                "timestamp": "2024-01-01T00:00:00",
                "feature_importance": {
                    "feature_0": 0.5,
                    "feature_1": 0.3,
                    "feature_2": 0.2,
                },
                "shap_values": [[0.1, -0.2, 0.3], [0.2, -0.1, 0.4]],
                "attention_weights": None,
                "explanation_dir": "explanations/uuid-string",
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    version: str = Field(..., description="API version")
    models_loaded: int = Field(..., description="Number of models loaded in cache")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T00:00:00",
                "version": "0.1.0",
                "models_loaded": 3,
            }
        }


class MetricsResponse(BaseModel):
    """Response schema for model metrics."""

    model_id: str = Field(..., description="Model ID")
    metrics: Dict[str, float] = Field(..., description="Performance metrics")
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "uuid-string",
                "metrics": {
                    "mse": 0.001,
                    "mae": 0.02,
                    "rmse": 0.03,
                    "r2": 0.95,
                },
                "timestamp": "2024-01-01T00:00:00",
            }
        }


class HyperoptRequest(BaseModel):
    """Request schema for hyperparameter optimization."""

    model_type: str = Field(
        default="attention_lstm",
        description="Model type to optimize",
        regex="^(lstm|attention_lstm|gru|attention_gru|transformer)$",
    )
    A: float = Field(..., description="Logistic map parameter A", ge=0, le=4)
    n_trials: int = Field(default=50, description="Number of optimization trials", ge=10, le=200)
    timeout: Optional[int] = Field(None, description="Optimization timeout in seconds", ge=60)
    window_size: int = Field(default=1, description="Sliding window size", ge=1, le=10)

    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "attention_lstm",
                "A": 3.5,
                "n_trials": 50,
                "timeout": 3600,
                "window_size": 1,
            }
        }


class HyperoptResponse(BaseModel):
    """Response schema for hyperparameter optimization."""

    study_name: str = Field(..., description="Optimization study name")
    best_params: Dict[str, Any] = Field(..., description="Best hyperparameters found")
    best_score: float = Field(..., description="Best validation score")
    n_trials: int = Field(..., description="Number of trials completed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Optimization timestamp")
    param_importance: Optional[Dict[str, float]] = Field(None, description="Parameter importance")

    class Config:
        json_schema_extra = {
            "example": {
                "study_name": "ann_hyperopt",
                "best_params": {
                    "num_layers": 2,
                    "units": 128,
                    "dropout_rate": 0.2,
                },
                "best_score": 0.001,
                "n_trials": 50,
                "timestamp": "2024-01-01T00:00:00",
                "param_importance": {
                    "units": 0.5,
                    "num_layers": 0.3,
                    "dropout_rate": 0.2,
                },
            }
        }

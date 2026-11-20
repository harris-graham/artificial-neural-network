"""FastAPI application for ANN prediction API."""

import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    PredictionRequest,
    PredictionResponse,
    TrainRequest,
    TrainResponse,
    ModelInfo,
    InterpretRequest,
    InterpretResponse,
    HealthResponse,
    MetricsResponse,
    HyperoptRequest,
    HyperoptResponse,
)
from .services import ModelService


# Global service instance
model_service: ModelService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global model_service

    # Initialize directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("explanations", exist_ok=True)

    # Initialize model service
    model_service = ModelService(models_dir="models")

    yield

    # Cleanup
    pass


# Create FastAPI app
app = FastAPI(
    title="Artificial Neural Network API",
    description="Advanced prediction API for chaotic time series with attention mechanisms and interpretability",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Artificial Neural Network API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    if model_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return model_service.get_health()


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """Make predictions for a logistic map sequence.

    This endpoint generates a logistic map sequence with the given parameter A,
    then uses a trained neural network to predict future values.

    - **A**: Logistic map parameter (0-4, values around 3.5-4 produce chaotic behavior)
    - **steps**: Number of steps to predict (1-100)
    - **model_id**: Optional specific model ID (uses latest if not provided)
    - **window_size**: Sliding window size for input (1-10)
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = model_service.predict(
            A=request.A,
            steps=request.steps,
            model_id=request.model_id,
            window_size=request.window_size,
        )
        return PredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/train", response_model=TrainResponse, tags=["Training"])
async def train_model(request: TrainRequest):
    """Train a new neural network model.

    This endpoint trains a new model on logistic map data with the given parameter A.
    Multiple model architectures are supported including LSTM, GRU, and Transformer
    with optional attention mechanisms.

    - **model_type**: Model architecture (lstm, attention_lstm, gru, attention_gru, transformer)
    - **A**: Logistic map parameter (0-4)
    - **hyperparameters**: Optional custom hyperparameters
    - **epochs**: Number of training epochs (1-1000)
    - **batch_size**: Training batch size (1-256)
    - **window_size**: Sliding window size (1-10)
    - **train_ratio**: Training data ratio (0.5-0.9)
    - **val_ratio**: Validation data ratio (0.05-0.3)
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = model_service.train_model(
            model_type=request.model_type,
            A=request.A,
            hyperparameters=request.hyperparameters,
            epochs=request.epochs,
            batch_size=request.batch_size,
            window_size=request.window_size,
            train_ratio=request.train_ratio,
            val_ratio=request.val_ratio,
        )
        return TrainResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models():
    """List all available trained models.

    Returns a list of all registered models with their metadata including
    model type, creation timestamp, configuration, and performance metrics.
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        models = model_service.list_models()
        return [ModelInfo(**model) for model in models]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/models/{model_id}", response_model=ModelInfo, tags=["Models"])
async def get_model(model_id: str):
    """Get detailed information about a specific model.

    - **model_id**: UUID of the model
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        model_info = model_service.get_model_info(model_id)
        if model_info is None:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        return ModelInfo(**model_info)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model: {str(e)}")


@app.delete("/models/{model_id}", tags=["Models"])
async def delete_model(model_id: str):
    """Delete a model from the registry.

    - **model_id**: UUID of the model to delete
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        success = model_service.delete_model(model_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        return {"message": f"Model {model_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@app.post("/interpret", response_model=InterpretResponse, tags=["Interpretability"])
async def interpret_model(request: InterpretRequest):
    """Generate model interpretation using SHAP or attention visualization.

    This endpoint provides interpretability insights for model predictions.
    SHAP (SHapley Additive exPlanations) values show feature importance,
    while attention visualization shows which parts of the input sequence
    the model focuses on.

    - **A**: Logistic map parameter (0-4)
    - **model_id**: UUID of the model to interpret
    - **method**: Interpretation method (shap, lime, attention)
    - **num_samples**: Number of samples to analyze (1-100)
    - **window_size**: Sliding window size (1-10)
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = model_service.interpret_model(
            A=request.A,
            model_id=request.model_id,
            method=request.method,
            num_samples=request.num_samples,
            window_size=request.window_size,
        )
        return InterpretResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interpretation failed: {str(e)}")


@app.get("/models/{model_id}/metrics", response_model=MetricsResponse, tags=["Models"])
async def get_model_metrics(model_id: str):
    """Get performance metrics for a specific model.

    - **model_id**: UUID of the model
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        metrics = model_service.get_model_metrics(model_id)
        if metrics is None:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        return MetricsResponse(**metrics)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.post("/hyperopt", response_model=HyperoptResponse, tags=["Optimization"])
async def optimize_hyperparameters(request: HyperoptRequest):
    """Run hyperparameter optimization using Optuna.

    This endpoint automatically searches for the best hyperparameters
    for a given model architecture using Bayesian optimization.

    - **model_type**: Model type to optimize (lstm, attention_lstm, gru, attention_gru, transformer)
    - **A**: Logistic map parameter (0-4)
    - **n_trials**: Number of optimization trials (10-200)
    - **timeout**: Optional timeout in seconds (minimum 60)
    - **window_size**: Sliding window size (1-10)
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = model_service.optimize_hyperparameters(
            model_type=request.model_type,
            A=request.A,
            n_trials=request.n_trials,
            timeout=request.timeout,
            window_size=request.window_size,
        )
        return HyperoptResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

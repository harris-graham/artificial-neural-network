"""Service layer for business logic."""

import os
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

import numpy as np
from tensorflow import keras

from ..models.model_factory import ModelFactory, ModelType, ModelRegistry
from ..models.attention_lstm import AttentionLSTMModel, ModelConfig
from ..models.hyperopt import HyperparameterOptimizer
from ..utils.data import generate_logistic_sequence, prepare_training_data
from ..utils.interpretability import SHAPExplainer, AttentionVisualizer, create_explanation_report


class ModelService:
    """Service for managing models and predictions."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        Path(models_dir).mkdir(exist_ok=True)
        self.factory = ModelFactory(models_dir)
        self.registry = ModelRegistry(models_dir)
        self._model_cache: Dict[str, keras.Model] = {}

    def _get_or_load_model(self, model_id: str) -> Optional[keras.Model]:
        """Get model from cache or load from disk."""
        if model_id in self._model_cache:
            return self._model_cache[model_id]

        model = self.factory.load_model(model_id)
        if model is not None:
            self._model_cache[model_id] = model
        return model

    def predict(
        self,
        A: float,
        steps: int,
        model_id: Optional[str] = None,
        window_size: int = 1,
    ) -> Dict[str, Any]:
        """Make predictions for a logistic map sequence.

        Args:
            A: Logistic map parameter
            steps: Number of steps to predict
            model_id: Specific model ID to use (if None, uses latest)
            window_size: Sliding window size

        Returns:
            Dictionary with predictions and metadata
        """
        # Generate sequence
        sequence = generate_logistic_sequence(A, n_steps=steps + 100)
        X, y, _, _, _, _ = prepare_training_data(sequence, window_size=window_size)

        # Get or create model
        if model_id is None:
            # Use the most recently trained model
            models = self.registry.list_models()
            if not models:
                raise ValueError("No models available. Please train a model first.")
            model_id = models[0]["id"]

        model = self._get_or_load_model(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found")

        model_info = self.registry.get(model_id)

        # Make predictions
        predictions = model.predict(X[:steps])
        predictions_flat = predictions.flatten()[:steps]
        true_values = y.flatten()[:steps]

        # Calculate metrics
        mse = np.mean((predictions_flat - true_values) ** 2)
        mae = np.mean(np.abs(predictions_flat - true_values))

        return {
            "predictions": predictions_flat.tolist(),
            "true_values": true_values.tolist(),
            "model_id": model_id,
            "model_type": model_info["model_type"] if model_info else "unknown",
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "mse": float(mse),
                "mae": float(mae),
            },
        }

    def train_model(
        self,
        model_type: str,
        A: float,
        hyperparameters: Optional[Dict[str, Any]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        window_size: int = 1,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> Dict[str, Any]:
        """Train a new model.

        Args:
            model_type: Type of model to train
            A: Logistic map parameter
            hyperparameters: Custom hyperparameters
            epochs: Number of training epochs
            batch_size: Batch size
            window_size: Sliding window size
            train_ratio: Training data ratio
            val_ratio: Validation data ratio

        Returns:
            Dictionary with training results
        """
        # Generate sequence
        sequence = generate_logistic_sequence(A, n_steps=500)
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_training_data(
            sequence, window_size=window_size, train_ratio=train_ratio, val_ratio=val_ratio
        )

        # Create config
        if hyperparameters:
            config = ModelConfig(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                **hyperparameters,
            )
        else:
            config = ModelConfig(input_shape=(X_train.shape[1], X_train.shape[2]))

        # Train model
        model_type_enum = ModelType(model_type)
        model_id = self.factory.train_and_register(
            model_type=model_type_enum,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            config=config,
            epochs=epochs,
            batch_size=batch_size,
        )

        # Clear cache
        self._model_cache.clear()

        # Get model info
        model_info = self.registry.get(model_id)

        return {
            "model_id": model_id,
            "model_type": model_type,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "training_time": model_info.get("training_time", 0) if model_info else 0,
            "epochs_trained": model_info.get("epochs_trained", 0) if model_info else 0,
            "metrics": model_info.get("metrics", {}) if model_info else {},
            "hyperparameters": config.dict(),
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        return self.registry.list_models()

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        model_info = self.registry.get(model_id)
        if model_info is None:
            return None

        return {
            "model_id": model_id,
            "model_type": model_info["model_type"],
            "created_at": model_info["created_at"],
            "config": model_info["config"],
            "metrics": model_info["metrics"],
        }

    def interpret_model(
        self,
        A: float,
        model_id: str,
        method: str = "shap",
        num_samples: int = 10,
        window_size: int = 1,
    ) -> Dict[str, Any]:
        """Generate model interpretation.

        Args:
            A: Logistic map parameter
            model_id: Model ID to interpret
            method: Interpretation method (shap, lime, attention)
            num_samples: Number of samples to analyze
            window_size: Sliding window size

        Returns:
            Dictionary with interpretation results
        """
        # Load model
        model = self._get_or_load_model(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found")

        # Generate sequence
        sequence = generate_logistic_sequence(A, n_steps=200)
        X, y, _, _, _, _ = prepare_training_data(sequence, window_size=window_size)

        # Take subset for interpretation
        X_subset = X[:num_samples]
        y_subset = y[:num_samples]

        # Get predictions
        predictions = model.predict(X_subset)

        # Create explanation directory
        explanation_dir = os.path.join("explanations", model_id)
        os.makedirs(explanation_dir, exist_ok=True)

        if method == "shap":
            # SHAP interpretation
            background_data = X[:min(100, len(X))]
            explainer = SHAPExplainer(model, background_data)
            feature_importance = explainer.get_feature_importance(X_subset)
            shap_values = explainer.get_shap_values(X_subset)

            # Generate plots
            explainer.plot_summary(X_subset, save_path=os.path.join(explanation_dir, "summary.png"))
            explainer.plot_force(X_subset[0], save_path=os.path.join(explanation_dir, "force.png"))

            return {
                "model_id": model_id,
                "method": method,
                "timestamp": datetime.now().isoformat(),
                "feature_importance": feature_importance,
                "shap_values": shap_values.tolist() if isinstance(shap_values, np.ndarray) else None,
                "explanation_dir": explanation_dir,
            }

        elif method == "attention":
            # Attention visualization
            visualizer = AttentionVisualizer(model)
            attention_weights = visualizer.extract_attention_weights(X_subset)

            if attention_weights is not None:
                visualizer.plot_attention_weights(
                    X_subset,
                    save_path=os.path.join(explanation_dir, "attention_heatmap.png"),
                )
                visualizer.plot_attention_over_time(
                    X_subset,
                    save_path=os.path.join(explanation_dir, "attention_over_time.png"),
                )

            return {
                "model_id": model_id,
                "method": method,
                "timestamp": datetime.now().isoformat(),
                "feature_importance": {},
                "attention_weights": attention_weights.tolist() if attention_weights is not None else None,
                "explanation_dir": explanation_dir,
            }

        else:
            raise ValueError(f"Unsupported interpretation method: {method}")

    def get_model_metrics(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a model."""
        model_info = self.registry.get(model_id)
        if model_info is None:
            return None

        # Use registered metrics
        metrics = model_info.get("metrics", {})

        # Calculate additional metrics if not present
        if "rmse" not in metrics and "mse" in metrics:
            metrics["rmse"] = np.sqrt(metrics["mse"])

        return {
            "model_id": model_id,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

    def optimize_hyperparameters(
        self,
        model_type: str,
        A: float,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        window_size: int = 1,
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization.

        Args:
            model_type: Model type to optimize
            A: Logistic map parameter
            n_trials: Number of optimization trials
            timeout: Optimization timeout in seconds
            window_size: Sliding window size

        Returns:
            Dictionary with optimization results
        """
        # Generate sequence
        sequence = generate_logistic_sequence(A, n_steps=500)
        X_train, y_train, X_val, y_val, _, _ = prepare_training_data(
            sequence, window_size=window_size
        )

        # Run optimization
        optimizer = HyperparameterOptimizer(
            study_name=f"hyperopt_{model_type}_{uuid.uuid4().hex[:8]}",
            models_dir=self.models_dir,
            n_trials=n_trials,
            timeout=timeout,
        )

        result = optimizer.optimize_attention_lstm(
            X_train, y_train, X_val, y_val, ModelType(model_type)
        )

        # Get parameter importance
        param_importance = optimizer.get_importance()

        return {
            "study_name": optimizer.study_name,
            "best_params": result["best_params"],
            "best_score": result["best_score"],
            "n_trials": result["n_trials"],
            "timestamp": datetime.now().isoformat(),
            "param_importance": param_importance,
        }

    def delete_model(self, model_id: str) -> bool:
        """Delete a model from registry and disk."""
        # Remove from cache
        if model_id in self._model_cache:
            del self._model_cache[model_id]

        return self.registry.delete(model_id)

    def get_health(self) -> Dict[str, Any]:
        """Get service health status."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "0.1.0",
            "models_loaded": len(self._model_cache),
        }

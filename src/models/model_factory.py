"""Model factory for creating and managing different neural network architectures."""

import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

from .attention_lstm import AttentionLSTMModel, ModelConfig


class ModelType(str, Enum):
    """Supported model architectures."""
    LSTM = "lstm"
    ATTENTION_LSTM = "attention_lstm"
    GRU = "gru"
    ATTENTION_GRU = "attention_gru"
    TRANSFORMER = "transformer"


class ModelRegistry:
    """Registry for managing trained models."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load model registry from disk."""
        registry_file = self.models_dir / "registry.json"
        if registry_file.exists():
            import json
            with open(registry_file, "r") as f:
                self._registry = json.load(f)

    def _save_registry(self) -> None:
        """Save model registry to disk."""
        import json
        registry_file = self.models_dir / "registry.json"
        with open(registry_file, "w") as f:
            json.dump(self._registry, f, indent=2)

    def register(
        self,
        model_id: str,
        model_type: ModelType,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        filepath: str,
    ) -> None:
        """Register a trained model."""
        self._registry[model_id] = {
            "model_type": model_type.value,
            "config": config,
            "metrics": metrics,
            "filepath": filepath,
            "created_at": datetime.now().isoformat(),
        }
        self._save_registry()

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model metadata by ID."""
        return self._registry.get(model_id)

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        return [
            {"id": k, **v} for k, v in self._registry.items()
        ]

    def delete(self, model_id: str) -> bool:
        """Delete a model from registry and disk."""
        if model_id not in self._registry:
            return False

        model_info = self._registry[model_id]
        filepath = model_info["filepath"]

        # Delete model files
        if os.path.exists(filepath):
            os.remove(filepath)
        metadata_path = filepath.replace(".keras", "_metadata.json")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

        # Remove from registry
        del self._registry[model_id]
        self._save_registry()
        return True


class ModelFactory:
    """Factory for creating different model architectures."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.registry = ModelRegistry(models_dir)

    def create_model(
        self,
        model_type: ModelType,
        config: Optional[ModelConfig] = None,
        input_shape: tuple = (1, 1),
    ) -> keras.Model:
        """Create a model of the specified type."""
        if config is None:
            config = ModelConfig(input_shape=input_shape)

        if model_type == ModelType.ATTENTION_LSTM:
            model = AttentionLSTMModel(config)
            model.build_model()
            return model
        elif model_type == ModelType.LSTM:
            return self._create_lstm_model(config)
        elif model_type == ModelType.GRU:
            return self._create_gru_model(config)
        elif model_type == ModelType.ATTENTION_GRU:
            return self._create_attention_gru_model(config)
        elif model_type == ModelType.TRANSFORMER:
            return self._create_transformer_model(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _create_lstm_model(self, config: ModelConfig) -> keras.Model:
        """Create a basic LSTM model."""
        inputs = keras.Input(shape=config.input_shape)
        x = inputs

        for i in range(config.num_layers):
            return_sequences = i < config.num_layers - 1
            lstm_layer = layers.LSTM(
                config.units,
                return_sequences=return_sequences,
                dropout=config.dropout_rate,
            )
            if config.use_bidirectional:
                x = layers.Bidirectional(lstm_layer)(x)
            else:
                x = lstm_layer(x)

        outputs = layers.Dense(1)(x)
        return keras.Model(inputs=inputs, outputs=outputs, name="LSTM")

    def _create_gru_model(self, config: ModelConfig) -> keras.Model:
        """Create a basic GRU model."""
        inputs = keras.Input(shape=config.input_shape)
        x = inputs

        for i in range(config.num_layers):
            return_sequences = i < config.num_layers - 1
            gru_layer = layers.GRU(
                config.units,
                return_sequences=return_sequences,
                dropout=config.dropout_rate,
            )
            if config.use_bidirectional:
                x = layers.Bidirectional(gru_layer)(x)
            else:
                x = gru_layer(x)

        outputs = layers.Dense(1)(x)
        return keras.Model(inputs=inputs, outputs=outputs, name="GRU")

    def _create_attention_gru_model(self, config: ModelConfig) -> keras.Model:
        """Create an attention-enhanced GRU model."""
        from .attention_lstm import AttentionLayer

        inputs = keras.Input(shape=config.input_shape)
        x = inputs

        for i in range(config.num_layers):
            return_sequences = i < config.num_layers - 1 or True
            gru_layer = layers.GRU(
                config.units,
                return_sequences=return_sequences,
                dropout=config.dropout_rate,
            )
            if config.use_bidirectional:
                x = layers.Bidirectional(gru_layer)(x)
            else:
                x = gru_layer(x)

        # Attention layer
        attention = AttentionLayer(config.units)
        x = attention([x, x[:, -1, :]])

        outputs = layers.Dense(1)(x)
        return keras.Model(inputs=inputs, outputs=outputs, name="AttentionGRU")

    def _create_transformer_model(self, config: ModelConfig) -> keras.Model:
        """Create a Transformer-based model for time series."""
        inputs = keras.Input(shape=config.input_shape)

        # Positional encoding
        x = inputs
        x = layers.Dense(config.units)(x)

        # Multi-head attention
        for _ in range(config.num_layers):
            attn_output = layers.MultiHeadAttention(
                num_heads=4, key_dim=config.units // 4
            )(x, x)
            attn_output = layers.Dropout(config.dropout_rate)(attn_output)
            x = layers.LayerNormalization()(x + attn_output)

            # Feed-forward
            ffn_output = layers.Dense(config.units, activation="relu")(x)
            ffn_output = layers.Dense(x.shape[-1])(ffn_output)
            ffn_output = layers.Dropout(config.dropout_rate)(ffn_output)
            x = layers.LayerNormalization()(x + ffn_output)

        # Global average pooling and output
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(1)(x)

        return keras.Model(inputs=inputs, outputs=outputs, name="Transformer")

    def train_and_register(
        self,
        model_type: ModelType,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        config: Optional[ModelConfig] = None,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> str:
        """Train a model and register it in the registry."""
        if config is None:
            config = ModelConfig(input_shape=(X_train.shape[1], X_train.shape[2]))

        model_id = str(uuid.uuid4())

        if model_type == ModelType.ATTENTION_LSTM:
            model_instance = AttentionLSTMModel(config)
            metadata = model_instance.train(
                X_train, y_train, X_val, y_val, epochs, batch_size
            )
            filepath = os.path.join(self.models_dir, f"{model_id}.keras")
            model_instance.save(filepath)
        else:
            model = self.create_model(model_type, config)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
                loss="mean_squared_error",
                metrics=["mae", "mse"],
            )

            callbacks = [
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            ]

            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0,
            )

            filepath = os.path.join(self.models_dir, f"{model_id}.keras")
            model.save(filepath)

            metadata = {
                "final_loss": history.history["loss"][-1],
                "final_mae": history.history["mae"][-1],
                "final_mse": history.history["mse"][-1],
            }
            if X_val is not None:
                metadata["final_val_loss"] = history.history["val_loss"][-1]
                metadata["final_val_mae"] = history.history["val_mae"][-1]

        # Register the model
        self.registry.register(
            model_id=model_id,
            model_type=model_type,
            config=config.dict(),
            metrics=metadata,
            filepath=filepath,
        )

        return model_id

    def load_model(self, model_id: str) -> Optional[keras.Model]:
        """Load a model by ID."""
        model_info = self.registry.get(model_id)
        if model_info is None:
            return None

        model_type = ModelType(model_info["model_type"])
        filepath = model_info["filepath"]

        if model_type == ModelType.ATTENTION_LSTM:
            return AttentionLSTMModel.load(filepath)
        else:
            return keras.models.load_model(filepath)

    def compare_models(
        self,
        model_ids: List[str],
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models on test data."""
        results = {}

        for model_id in model_ids:
            model = self.load_model(model_id)
            if model is None:
                continue

            predictions = model.predict(X_test)

            # Calculate metrics
            mse = np.mean((predictions.flatten() - y_test.flatten()) ** 2)
            mae = np.mean(np.abs(predictions.flatten() - y_test.flatten()))
            rmse = np.sqrt(mse)

            # R² score
            ss_res = np.sum((y_test.flatten() - predictions.flatten()) ** 2)
            ss_tot = np.sum((y_test.flatten() - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            results[model_id] = {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
            }

        return results

"""Attention-enhanced LSTM model for time series prediction."""

import os
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """Configuration for the attention LSTM model."""

    input_shape: Tuple[int, int] = (1, 1)
    num_layers: int = Field(default=2, ge=1, le=5)
    units: int = Field(default=128, ge=32, le=512)
    dropout_rate: float = Field(default=0.2, ge=0.0, le=0.5)
    use_bidirectional: bool = True
    attention_type: str = Field(default="bahdanau", regex="^(bahdanau|luong)$")
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.01)
    use_attention: bool = True

    @validator("input_shape")
    def validate_input_shape(cls, v):
        if len(v) != 2:
            raise ValueError("input_shape must be a tuple of (timesteps, features)")
        return v


class AttentionLayer(layers.Layer):
    """Bahdanau attention mechanism for sequence modeling."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Compute attention weights and context vector."""
        hidden_states, hidden_state = inputs

        # Calculate attention scores
        hidden_with_time_axis = tf.expand_dims(hidden_state, 1)
        score = tf.nn.tanh(self.W1(hidden_states) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # Calculate context vector
        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"units": self.units})
        return config


class LuongAttentionLayer(layers.Layer):
    """Luong (multiplicative) attention mechanism."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Compute attention weights and context vector."""
        hidden_states, hidden_state = inputs

        # Calculate attention scores (multiplicative)
        hidden_with_time_axis = tf.expand_dims(hidden_state, 2)
        score = tf.matmul(hidden_states, self.W(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(score, axis=1)

        # Calculate context vector
        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"units": self.units})
        return config


class AttentionLSTMModel:
    """Attention-enhanced LSTM model for time series prediction."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Optional[Model] = None
        self.history: Optional[Dict[str, Any]] = None
        self.metadata: Dict[str, Any] = {}

    def build_model(self) -> Model:
        """Build the attention LSTM model."""
        inputs = keras.Input(shape=self.config.input_shape)
        x = inputs

        # LSTM layers
        for i in range(self.config.num_layers):
            return_sequences = i < self.config.num_layers - 1 or self.config.use_attention
            lstm_layer = layers.LSTM(
                self.config.units,
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate,
            )

            if self.config.use_bidirectional:
                x = layers.Bidirectional(lstm_layer)(x)
            else:
                x = lstm_layer(x)

        # Attention layer
        if self.config.use_attention:
            if self.config.attention_type == "bahdanau":
                attention = AttentionLayer(self.config.units)
            else:
                attention = LuongAttentionLayer(self.config.units)

            # Get the last hidden state for attention
            if self.config.use_bidirectional:
                # For bidirectional, concatenate the last hidden states
                x = attention([x, x[:, -1, :]])
            else:
                x = attention([x, x[:, -1, :]])

        # Output layer
        outputs = layers.Dense(1)(x)

        self.model = Model(inputs=inputs, outputs=outputs, name="AttentionLSTM")
        return self.model

    def compile(self) -> None:
        """Compile the model with optimizer and loss."""
        if self.model is None:
            self.build_model()

        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss="mean_squared_error",
            metrics=["mae", "mse"],
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """Train the model with early stopping and learning rate scheduling."""
        if self.model is None:
            self.build_model()
            self.compile()

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=patience,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            ),
        ]

        # Training
        start_time = datetime.now()
        validation_data = (X_val, y_val) if X_val is not None else None

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        ).history

        training_time = (datetime.now() - start_time).total_seconds()

        # Store metadata
        self.metadata = {
            "config": self.config.dict(),
            "training_time": training_time,
            "epochs_trained": len(self.history["loss"]),
            "final_loss": self.history["loss"][-1],
            "final_mae": self.history["mae"][-1],
            "final_mse": self.history["mse"][-1],
        }

        if X_val is not None:
            self.metadata["final_val_loss"] = self.history["val_loss"][-1]
            self.metadata["final_val_mae"] = self.history["val_mae"][-1]
            self.metadata["final_val_mse"] = self.history["val_mse"][-1]

        return self.metadata

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model has not been built or trained")
        return self.model.predict(X)

    def save(self, filepath: str) -> None:
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("Model has not been built or trained")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save model
        self.model.save(filepath)

        # Save metadata
        metadata_path = filepath.replace(".keras", "_metadata.json")
        import json

        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "AttentionLSTMModel":
        """Load a model from disk."""
        # Load metadata
        metadata_path = filepath.replace(".keras", "_metadata.json")
        import json

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        config = ModelConfig(**metadata["config"])
        instance = cls(config)
        instance.model = keras.models.load_model(filepath, custom_objects={
            "AttentionLayer": AttentionLayer,
            "LuongAttentionLayer": LuongAttentionLayer,
        })
        instance.metadata = metadata

        return instance

    def get_attention_weights(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Extract attention weights for interpretability."""
        if self.model is None or not self.config.use_attention:
            return None

        # This would require creating a separate model that outputs attention weights
        # For now, return None as this is complex to implement without modifying the architecture
        return None

    def summary(self) -> str:
        """Return model summary."""
        if self.model is None:
            self.build_model()
        return self.model.summary()

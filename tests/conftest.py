"""Pytest configuration and fixtures."""

import os
import pytest
import numpy as np
from pathlib import Path

from src.models.attention_lstm import AttentionLSTMModel, ModelConfig
from src.models.model_factory import ModelFactory, ModelType
from src.utils.data import generate_logistic_sequence, prepare_training_data


@pytest.fixture
def sample_sequence():
    """Generate a sample logistic map sequence."""
    return generate_logistic_sequence(A=3.5, n_steps=200)


@pytest.fixture
def sample_data(sample_sequence):
    """Prepare sample training data."""
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_training_data(
        sample_sequence, window_size=1
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


@pytest.fixture
def model_config():
    """Create a model configuration."""
    return ModelConfig(
        input_shape=(1, 1),
        num_layers=1,
        units=32,
        dropout_rate=0.1,
        use_bidirectional=False,
        attention_type="bahdanau",
        learning_rate=0.001,
    )


@pytest.fixture
def attention_lstm_model(model_config):
    """Create an attention LSTM model."""
    model = AttentionLSTMModel(model_config)
    model.build_model()
    model.compile()
    return model


@pytest.fixture
def trained_model(attention_lstm_model, sample_data):
    """Create a trained model for testing."""
    model = attention_lstm_model
    metadata = model.train(
        sample_data["X_train"],
        sample_data["y_train"],
        sample_data["X_val"],
        sample_data["y_val"],
        epochs=2,
        batch_size=16,
        patience=1,
        verbose=0,
    )
    return model


@pytest.fixture
def model_factory(tmp_path):
    """Create a model factory with temporary directory."""
    factory = ModelFactory(models_dir=str(tmp_path / "models"))
    return factory


@pytest.fixture
def tmp_path(tmp_path):
    """Create temporary directories for testing."""
    os.makedirs(tmp_path / "models", exist_ok=True)
    os.makedirs(tmp_path / "data", exist_ok=True)
    os.makedirs(tmp_path / "explanations", exist_ok=True)
    return tmp_path

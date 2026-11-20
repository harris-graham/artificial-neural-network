"""Tests for model components."""

import pytest
import numpy as np
import os

from src.models.attention_lstm import AttentionLSTMModel, ModelConfig, AttentionLayer, LuongAttentionLayer
from src.models.model_factory import ModelFactory, ModelType, ModelRegistry


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ModelConfig()
        assert config.input_shape == (1, 1)
        assert config.num_layers == 2
        assert config.units == 128
        assert config.dropout_rate == 0.2
        assert config.use_bidirectional is True
        assert config.attention_type == "bahdanau"

    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelConfig(
            input_shape=(5, 2),
            num_layers=3,
            units=256,
            dropout_rate=0.3,
            use_bidirectional=False,
            attention_type="luong",
        )
        assert config.input_shape == (5, 2)
        assert config.num_layers == 3
        assert config.units == 256
        assert config.dropout_rate == 0.3
        assert config.use_bidirectional is False
        assert config.attention_type == "luong"

    def test_invalid_input_shape(self):
        """Test validation of invalid input shape."""
        with pytest.raises(ValueError):
            ModelConfig(input_shape=(1, 1, 1))

    def test_invalid_attention_type(self):
        """Test validation of invalid attention type."""
        with pytest.raises(ValueError):
            ModelConfig(attention_type="invalid")


class TestAttentionLayers:
    """Tests for attention layers."""

    def test_bahdanau_attention_layer(self):
        """Test Bahdanau attention layer."""
        layer = AttentionLayer(units=64)
        hidden_states = np.random.randn(2, 10, 64)
        hidden_state = np.random.randn(2, 64)

        context = layer([hidden_states, hidden_state])
        assert context.shape == (2, 64)

    def test_luong_attention_layer(self):
        """Test Luong attention layer."""
        layer = LuongAttentionLayer(units=64)
        hidden_states = np.random.randn(2, 10, 64)
        hidden_state = np.random.randn(2, 64)

        context = layer([hidden_states, hidden_state])
        assert context.shape == (2, 64)


class TestAttentionLSTMModel:
    """Tests for AttentionLSTMModel."""

    def test_build_model(self, model_config):
        """Test model building."""
        model = AttentionLSTMModel(model_config)
        keras_model = model.build_model()
        assert keras_model is not None
        assert keras_model.input_shape == (None, *model_config.input_shape)

    def test_compile_model(self, model_config):
        """Test model compilation."""
        model = AttentionLSTMModel(model_config)
        model.compile()
        assert model.model is not None
        assert model.model.optimizer is not None

    def test_train_model(self, attention_lstm_model, sample_data):
        """Test model training."""
        metadata = attention_lstm_model.train(
            sample_data["X_train"],
            sample_data["y_train"],
            sample_data["X_val"],
            sample_data["y_val"],
            epochs=2,
            batch_size=16,
            patience=1,
            verbose=0,
        )
        assert "training_time" in metadata
        assert "epochs_trained" in metadata
        assert "final_loss" in metadata

    def test_predict(self, trained_model, sample_data):
        """Test model prediction."""
        predictions = trained_model.predict(sample_data["X_test"])
        assert predictions is not None
        assert predictions.shape[0] == sample_data["X_test"].shape[0]

    def test_save_and_load(self, trained_model, tmp_path):
        """Test model saving and loading."""
        filepath = os.path.join(tmp_path, "test_model.keras")
        trained_model.save(filepath)

        loaded_model = AttentionLSTMModel.load(filepath)
        assert loaded_model is not None
        assert loaded_model.config == trained_model.config


class TestModelFactory:
    """Tests for ModelFactory."""

    def test_create_lstm_model(self, model_factory):
        """Test creating LSTM model."""
        model = model_factory.create_model(ModelType.LSTM)
        assert model is not None
        assert model.name == "LSTM"

    def test_create_attention_lstm_model(self, model_factory):
        """Test creating attention LSTM model."""
        model = model_factory.create_model(ModelType.ATTENTION_LSTM)
        assert model is not None
        assert model.name == "AttentionLSTM"

    def test_create_gru_model(self, model_factory):
        """Test creating GRU model."""
        model = model_factory.create_model(ModelType.GRU)
        assert model is not None
        assert model.name == "GRU"

    def test_create_transformer_model(self, model_factory):
        """Test creating Transformer model."""
        model = model_factory.create_model(ModelType.TRANSFORMER)
        assert model is not None
        assert model.name == "Transformer"

    def test_train_and_register(self, model_factory, sample_data):
        """Test training and registering a model."""
        model_id = model_factory.train_and_register(
            ModelType.ATTENTION_LSTM,
            sample_data["X_train"],
            sample_data["y_train"],
            sample_data["X_val"],
            sample_data["y_val"],
            epochs=2,
            batch_size=16,
        )
        assert model_id is not None
        assert len(model_id) > 0

    def test_load_model(self, model_factory, sample_data):
        """Test loading a registered model."""
        model_id = model_factory.train_and_register(
            ModelType.ATTENTION_LSTM,
            sample_data["X_train"],
            sample_data["y_train"],
            sample_data["X_val"],
            sample_data["y_val"],
            epochs=2,
            batch_size=16,
        )
        model = model_factory.load_model(model_id)
        assert model is not None

    def test_compare_models(self, model_factory, sample_data):
        """Test comparing multiple models."""
        model_id1 = model_factory.train_and_register(
            ModelType.LSTM,
            sample_data["X_train"],
            sample_data["y_train"],
            sample_data["X_val"],
            sample_data["y_val"],
            epochs=2,
            batch_size=16,
        )
        model_id2 = model_factory.train_and_register(
            ModelType.ATTENTION_LSTM,
            sample_data["X_train"],
            sample_data["y_train"],
            sample_data["X_val"],
            sample_data["y_val"],
            epochs=2,
            batch_size=16,
        )

        results = model_factory.compare_models(
            [model_id1, model_id2],
            sample_data["X_test"],
            sample_data["y_test"],
        )
        assert model_id1 in results
        assert model_id2 in results
        assert "mse" in results[model_id1]
        assert "mae" in results[model_id1]


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_model(self, tmp_path):
        """Test registering a model."""
        registry = ModelRegistry(models_dir=str(tmp_path / "models"))
        registry.register(
            model_id="test-id",
            model_type=ModelType.ATTENTION_LSTM,
            config={"units": 128},
            metrics={"loss": 0.1},
            filepath="test.keras",
        )
        model_info = registry.get("test-id")
        assert model_info is not None
        assert model_info["model_type"] == "attention_lstm"

    def test_list_models(self, tmp_path):
        """Test listing models."""
        registry = ModelRegistry(models_dir=str(tmp_path / "models"))
        registry.register(
            model_id="test-id-1",
            model_type=ModelType.LSTM,
            config={"units": 64},
            metrics={"loss": 0.2},
            filepath="test1.keras",
        )
        registry.register(
            model_id="test-id-2",
            model_type=ModelType.ATTENTION_LSTM,
            config={"units": 128},
            metrics={"loss": 0.1},
            filepath="test2.keras",
        )
        models = registry.list_models()
        assert len(models) == 2

    def test_delete_model(self, tmp_path):
        """Test deleting a model."""
        registry = ModelRegistry(models_dir=str(tmp_path / "models"))
        registry.register(
            model_id="test-id",
            model_type=ModelType.LSTM,
            config={"units": 64},
            metrics={"loss": 0.2},
            filepath="test.keras",
        )
        success = registry.delete("test-id")
        assert success is True
        assert registry.get("test-id") is None

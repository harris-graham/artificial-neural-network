"""Tests for utility functions."""

import pytest
import numpy as np

from src.utils.data import (
    generate_logistic_sequence,
    prepare_training_data,
    normalize_data,
    denormalize_data,
    create_multiple_sequences,
)


class TestGenerateLogisticSequence:
    """Tests for logistic sequence generation."""

    def test_basic_sequence(self):
        """Test basic sequence generation."""
        sequence = generate_logistic_sequence(A=2.0, n_steps=100)
        assert len(sequence) == 100
        assert all(0 <= x <= 1 for x in sequence)

    def test_custom_initial_value(self):
        """Test sequence with custom initial value."""
        sequence = generate_logistic_sequence(A=2.0, n_steps=100, x0=0.5)
        assert sequence[0] == 0.5

    def test_chaotic_behavior(self):
        """Test that high A values produce chaotic behavior."""
        sequence = generate_logistic_sequence(A=4.0, n_steps=100)
        # Check for variability (chaotic sequences should have high variance)
        variance = np.var(sequence)
        assert variance > 0.1


class TestPrepareTrainingData:
    """Tests for training data preparation."""

    def test_basic_preparation(self, sample_sequence):
        """Test basic data preparation."""
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_training_data(
            sample_sequence, window_size=1
        )
        assert X_train.shape[1] == 1  # window_size
        assert X_train.shape[2] == 1  # features
        assert X_train.shape[0] > 0
        assert y_train.shape == X_train.shape

    def test_window_size(self, sample_sequence):
        """Test different window sizes."""
        X_train, y_train, _, _, _, _ = prepare_training_data(
            sample_sequence, window_size=5
        )
        assert X_train.shape[1] == 5

    def test_data_split_ratios(self, sample_sequence):
        """Test data split ratios."""
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_training_data(
            sample_sequence, window_size=1, train_ratio=0.7, val_ratio=0.15
        )
        total = len(X_train) + len(X_val) + len(X_test)
        assert abs(len(X_train) / total - 0.7) < 0.05
        assert abs(len(X_val) / total - 0.15) < 0.05


class TestNormalizeData:
    """Tests for data normalization."""

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        data = np.array([1, 2, 3, 4, 5])
        normalized, params = normalize_data(data, method="minmax")
        assert np.all(normalized >= 0) and np.all(normalized <= 1)
        assert params["method"] == "minmax"

    def test_standard_normalization(self):
        """Test standard (z-score) normalization."""
        data = np.array([1, 2, 3, 4, 5])
        normalized, params = normalize_data(data, method="standard")
        assert abs(np.mean(normalized)) < 0.1  # Should be close to 0
        assert abs(np.std(normalized) - 1) < 0.1  # Should be close to 1
        assert params["method"] == "standard"

    def test_denormalize_minmax(self):
        """Test denormalization with min-max."""
        data = np.array([1, 2, 3, 4, 5])
        normalized, params = normalize_data(data, method="minmax")
        denormalized = denormalize_data(normalized, params)
        np.testing.assert_array_almost_equal(data, denormalized)

    def test_denormalize_standard(self):
        """Test denormalization with standard."""
        data = np.array([1, 2, 3, 4, 5])
        normalized, params = normalize_data(data, method="standard")
        denormalized = denormalize_data(normalized, params)
        np.testing.assert_array_almost_equal(data, denormalized, decimal=5)

    def test_invalid_method(self):
        """Test invalid normalization method."""
        data = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError):
            normalize_data(data, method="invalid")


class TestCreateMultipleSequences:
    """Tests for creating multiple sequences."""

    def test_multiple_sequences(self):
        """Test creating sequences for multiple A values."""
        A_values = [2.0, 3.0, 4.0]
        sequences = create_multiple_sequences(A_values, n_steps=100)
        assert len(sequences) == 3
        assert 2.0 in sequences
        assert 3.0 in sequences
        assert 4.0 in sequences
        assert all(len(seq) == 100 for seq in sequences.values())

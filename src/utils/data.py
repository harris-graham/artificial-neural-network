"""Data generation and preprocessing utilities."""

import numpy as np
from typing import Tuple, Optional


def generate_logistic_sequence(A: float, n_steps: int = 500, x0: float = 0.1) -> np.ndarray:
    """Generate a logistic map sequence.

    Args:
        A: The logistic map parameter (typically 2-4 for chaotic behavior)
        n_steps: Number of steps to generate
        x0: Initial value

    Returns:
        Array of sequence values
    """
    x = [x0]
    for n in range(1, n_steps):
        x.append(A * x[n - 1] * (1 - x[n - 1]))
    return np.array(x)


def prepare_training_data(
    sequence: np.ndarray,
    window_size: int = 1,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Prepare training, validation, and test data from a sequence.

    Args:
        sequence: Input sequence
        window_size: Size of the sliding window
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Create sliding window sequences
    X = []
    y = []
    for i in range(len(sequence) - window_size):
        X.append(sequence[i:i + window_size])
        y.append(sequence[i + window_size])

    X = np.array(X)
    y = np.array(y)

    # Reshape for LSTM (samples, timesteps, features)
    X = X.reshape(-1, window_size, 1)
    y = y.reshape(-1, 1, 1)

    # Split data
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_data(data: np.ndarray, method: str = "minmax") -> Tuple[np.ndarray, dict]:
    """Normalize data using specified method.

    Args:
        data: Input data
        method: Normalization method ('minmax' or 'standard')

    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    if method == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        params = {"method": "minmax", "min": min_val, "max": max_val}
    elif method == "standard":
        mean = np.mean(data)
        std = np.std(data)
        normalized = (data - mean) / (std + 1e-8)
        params = {"method": "standard", "mean": mean, "std": std}
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, params


def denormalize_data(data: np.ndarray, params: dict) -> np.ndarray:
    """Denormalize data using provided parameters.

    Args:
        data: Normalized data
        params: Normalization parameters

    Returns:
        Denormalized data
    """
    if params["method"] == "minmax":
        return data * (params["max"] - params["min"]) + params["min"]
    elif params["method"] == "standard":
        return data * params["std"] + params["mean"]
    else:
        raise ValueError(f"Unknown normalization method: {params['method']}")


def create_multiple_sequences(
    A_values: list[float],
    n_steps: int = 500,
    x0: float = 0.1,
) -> dict[float, np.ndarray]:
    """Generate multiple logistic map sequences for different A values.

    Args:
        A_values: List of A parameter values
        n_steps: Number of steps per sequence
        x0: Initial value

    Returns:
        Dictionary mapping A values to sequences
    """
    sequences = {}
    for A in A_values:
        sequences[A] = generate_logistic_sequence(A, n_steps, x0)
    return sequences

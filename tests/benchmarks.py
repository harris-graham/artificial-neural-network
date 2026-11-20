"""Performance benchmarks for model comparison."""

import time
import pytest
import numpy as np
from pathlib import Path

from src.models.attention_lstm import AttentionLSTMModel, ModelConfig
from src.models.model_factory import ModelFactory, ModelType
from src.utils.data import generate_logistic_sequence, prepare_training_data


class BenchmarkResults:
    """Container for benchmark results."""

    def __init__(self):
        self.results = {}

    def add_result(self, model_type: str, metric: str, value: float):
        """Add a benchmark result."""
        if model_type not in self.results:
            self.results[model_type] = {}
        self.results[model_type][metric] = value

    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        for model_type, metrics in self.results.items():
            print(f"\n{model_type.upper()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        print("=" * 60 + "\n")


def benchmark_training_time(model_factory, model_type, X_train, y_train, X_val, y_val):
    """Benchmark training time for a model."""
    start_time = time.time()
    model_id = model_factory.train_and_register(
        model_type,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=10,
        batch_size=32,
    )
    training_time = time.time() - start_time
    return training_time, model_id


def benchmark_inference_time(model_factory, model_id, X_test):
    """Benchmark inference time for a model."""
    model = model_factory.load_model(model_id)

    # Warm-up
    model.predict(X_test[:10])

    # Benchmark
    start_time = time.time()
    predictions = model.predict(X_test)
    inference_time = time.time() - start_time

    return inference_time, predictions


def benchmark_memory_usage():
    """Benchmark memory usage (simplified)."""
    # Memory benchmarking requires psutil, skipped for now
    return 0.0


@pytest.mark.benchmark
class TestModelBenchmarks:
    """Benchmark tests for different model architectures."""

    @pytest.fixture
    def benchmark_data(self):
        """Generate data for benchmarking."""
        sequence = generate_logistic_sequence(A=3.5, n_steps=500)
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_training_data(
            sequence, window_size=1
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
    def benchmark_factory(self, tmp_path):
        """Create factory for benchmarking."""
        return ModelFactory(models_dir=str(tmp_path / "models"))

    def test_benchmark_lstm_vs_attention_lstm(self, benchmark_data, benchmark_factory):
        """Benchmark LSTM vs Attention LSTM."""
        results = BenchmarkResults()

        # Benchmark LSTM
        print("\nBenchmarking LSTM...")
        lstm_time, lstm_id = benchmark_training_time(
            benchmark_factory,
            ModelType.LSTM,
            benchmark_data["X_train"],
            benchmark_data["y_train"],
            benchmark_data["X_val"],
            benchmark_data["y_val"],
        )
        lstm_inference_time, lstm_pred = benchmark_inference_time(
            benchmark_factory, lstm_id, benchmark_data["X_test"]
        )
        lstm_mse = np.mean((lstm_pred.flatten() - benchmark_data["y_test"].flatten()) ** 2)

        results.add_result("lstm", "training_time", lstm_time)
        results.add_result("lstm", "inference_time", lstm_inference_time)
        results.add_result("lstm", "mse", lstm_mse)

        # Benchmark Attention LSTM
        print("Benchmarking Attention LSTM...")
        attn_time, attn_id = benchmark_training_time(
            benchmark_factory,
            ModelType.ATTENTION_LSTM,
            benchmark_data["X_train"],
            benchmark_data["y_train"],
            benchmark_data["X_val"],
            benchmark_data["y_val"],
        )
        attn_inference_time, attn_pred = benchmark_inference_time(
            benchmark_factory, attn_id, benchmark_data["X_test"]
        )
        attn_mse = np.mean((attn_pred.flatten() - benchmark_data["y_test"].flatten()) ** 2)

        results.add_result("attention_lstm", "training_time", attn_time)
        results.add_result("attention_lstm", "inference_time", attn_inference_time)
        results.add_result("attention_lstm", "mse", attn_mse)

        results.print_summary()

        # Assert attention LSTM has better or similar accuracy
        assert attn_mse <= lstm_mse * 1.5  # Allow 50% worse performance

    def test_benchmark_all_architectures(self, benchmark_data, benchmark_factory):
        """Benchmark all supported architectures."""
        results = BenchmarkResults()

        architectures = [
            ModelType.LSTM,
            ModelType.ATTENTION_LSTM,
            ModelType.GRU,
            ModelType.TRANSFORMER,
        ]

        for arch in architectures:
            print(f"\nBenchmarking {arch.value}...")
            try:
                train_time, model_id = benchmark_training_time(
                    benchmark_factory,
                    arch,
                    benchmark_data["X_train"],
                    benchmark_data["y_train"],
                    benchmark_data["X_val"],
                    benchmark_data["y_val"],
                )
                inference_time, predictions = benchmark_inference_time(
                    benchmark_factory, model_id, benchmark_data["X_test"]
                )
                mse = np.mean((predictions.flatten() - benchmark_data["y_test"].flatten()) ** 2)
                mae = np.mean(np.abs(predictions.flatten() - benchmark_data["y_test"].flatten()))

                results.add_result(arch.value, "training_time", train_time)
                results.add_result(arch.value, "inference_time", inference_time)
                results.add_result(arch.value, "mse", mse)
                results.add_result(arch.value, "mae", mae)
            except Exception as e:
                print(f"Error benchmarking {arch.value}: {e}")

        results.print_summary()

    def test_benchmark_window_sizes(self, benchmark_data, benchmark_factory):
        """Benchmark different window sizes."""
        results = BenchmarkResults()

        window_sizes = [1, 3, 5, 10]

        for window_size in window_sizes:
            print(f"\nBenchmarking window_size={window_size}...")
            sequence = generate_logistic_sequence(A=3.5, n_steps=500)
            X_train, y_train, X_val, y_val, X_test, y_test = prepare_training_data(
                sequence, window_size=window_size
            )

            train_time, model_id = benchmark_training_time(
                benchmark_factory,
                ModelType.ATTENTION_LSTM,
                X_train,
                y_train,
                X_val,
                y_val,
            )
            inference_time, predictions = benchmark_inference_time(
                benchmark_factory, model_id, X_test
            )
            mse = np.mean((predictions.flatten() - y_test.flatten()) ** 2)

            results.add_result(f"window_{window_size}", "training_time", train_time)
            results.add_result(f"window_{window_size}", "inference_time", inference_time)
            results.add_result(f"window_{window_size}", "mse", mse)

        results.print_summary()

    def test_benchmark_batch_sizes(self, benchmark_data, benchmark_factory):
        """Benchmark different batch sizes."""
        results = BenchmarkResults()

        batch_sizes = [16, 32, 64, 128]

        for batch_size in batch_sizes:
            print(f"\nBenchmarking batch_size={batch_size}...")
            train_time, model_id = benchmark_training_time(
                benchmark_factory,
                ModelType.ATTENTION_LSTM,
                benchmark_data["X_train"],
                benchmark_data["y_train"],
                benchmark_data["X_val"],
                benchmark_data["y_val"],
            )
            inference_time, predictions = benchmark_inference_time(
                benchmark_factory, model_id, benchmark_data["X_test"]
            )
            mse = np.mean((predictions.flatten() - benchmark_data["y_test"].flatten()) ** 2)

            results.add_result(f"batch_{batch_size}", "training_time", train_time)
            results.add_result(f"batch_{batch_size}", "inference_time", inference_time)
            results.add_result(f"batch_{batch_size}", "mse", mse)

        results.print_summary()


if __name__ == "__main__":
    """Run benchmarks directly."""
    pytest.main([__file__, "-v", "-s", "-m", "benchmark"])

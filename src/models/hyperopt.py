"""Hyperparameter optimization using Optuna."""

import os
from typing import Optional, Dict, Any, Callable
from pathlib import Path

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from .attention_lstm import AttentionLSTMModel, ModelConfig
from .model_factory import ModelFactory, ModelType


class HyperparameterOptimizer:
    """Hyperparameter optimization for neural network models."""

    def __init__(
        self,
        study_name: str = "ann_hyperopt",
        storage: Optional[str] = None,
        models_dir: str = "models",
        n_trials: int = 100,
        timeout: Optional[int] = None,
    ):
        self.study_name = study_name
        self.storage = storage
        self.models_dir = models_dir
        self.n_trials = n_trials
        self.timeout = timeout
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None

    def create_study(
        self,
        direction: str = "minimize",
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> optuna.Study:
        """Create or load an Optuna study."""
        if sampler is None:
            sampler = TPESampler(seed=42)

        if pruner is None:
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
        )

        return self.study

    def optimize_attention_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_type: ModelType = ModelType.ATTENTION_LSTM,
    ) -> Dict[str, Any]:
        """Optimize hyperparameters for attention LSTM model."""
        if self.study is None:
            self.create_study()

        def objective(trial: optuna.Trial) -> float:
            # Define search space
            num_layers = trial.suggest_int("num_layers", 1, 4)
            units = trial.suggest_categorical("units", [32, 64, 128, 256])
            dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
            use_bidirectional = trial.suggest_categorical("use_bidirectional", [True, False])
            attention_type = trial.suggest_categorical("attention_type", ["bahdanau", "luong"])
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

            # Create config
            config = ModelConfig(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                num_layers=num_layers,
                units=units,
                dropout_rate=dropout_rate,
                use_bidirectional=use_bidirectional,
                attention_type=attention_type,
                learning_rate=learning_rate,
            )

            # Train model
            if model_type == ModelType.ATTENTION_LSTM:
                model = AttentionLSTMModel(config)
                metadata = model.train(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    epochs=50,
                    batch_size=batch_size,
                    patience=5,
                    verbose=0,
                )
                val_loss = metadata.get("final_val_loss", metadata["final_loss"])
            else:
                factory = ModelFactory(self.models_dir)
                model = factory.create_model(model_type, config)
                model.compile(
                    optimizer="adam",
                    loss="mean_squared_error",
                    metrics=["mae"],
                )

                history = model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=batch_size,
                    callbacks=[
                        optuna.integration.TFKerasPruningCallback(
                            trial, "val_loss"
                        )
                    ],
                    verbose=0,
                )
                val_loss = history.history["val_loss"][-1]

            # Report intermediate value for pruning
            trial.report(val_loss, step=0)

            return val_loss

        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )

        # Get best parameters
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": len(self.study.trials),
        }

    def optimize_model_comparison(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_types: list[ModelType],
    ) -> Dict[str, Dict[str, Any]]:
        """Compare optimization across different model types."""
        results = {}

        for model_type in model_types:
            study_name = f"{self.study_name}_{model_type.value}"
            optimizer = HyperparameterOptimizer(
                study_name=study_name,
                storage=self.storage,
                models_dir=self.models_dir,
                n_trials=self.n_trials // len(model_types),
            )

            result = optimizer.optimize_attention_lstm(
                X_train, y_train, X_val, y_val, model_type
            )
            results[model_type.value] = result

        return results

    def get_best_model(
        self,
        model_type: ModelType = ModelType.ATTENTION_LSTM,
    ) -> AttentionLSTMModel:
        """Create a model with the best hyperparameters."""
        if self.best_params is None:
            raise ValueError("No optimization has been performed yet")

        config = ModelConfig(**self.best_params)
        model = AttentionLSTMModel(config)
        return model

    def save_study(self, filepath: str) -> None:
        """Save the Optuna study to disk."""
        if self.storage is None:
            # If using in-memory storage, save to SQLite
            storage_path = f"sqlite:///{filepath}"
            self.study = optuna.load_study(
                study_name=self.study_name,
                storage=storage_path,
            )

    def load_study(self, filepath: str) -> optuna.Study:
        """Load an Optuna study from disk."""
        storage_path = f"sqlite:///{filepath}"
        self.storage = storage_path
        self.study = optuna.load_study(
            study_name=self.study_name,
            storage=storage_path,
        )
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        return self.study

    def get_importance(self) -> Dict[str, float]:
        """Get hyperparameter importance from the study."""
        if self.study is None:
            raise ValueError("No study has been created")

        return optuna.importance.get_param_importances(self.study)

    def get_trials_dataframe(self) -> Any:
        """Get trials as a pandas DataFrame."""
        if self.study is None:
            raise ValueError("No study has been created")

        return self.study.trials_dataframe()

    def plot_optimization_history(self, save_path: Optional[str] = None) -> None:
        """Plot optimization history."""
        if self.study is None:
            raise ValueError("No study has been created")

        import matplotlib.pyplot as plt

        fig = optuna.visualization.plot_optimization_history(self.study)
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_param_importances(self, save_path: Optional[str] = None) -> None:
        """Plot hyperparameter importances."""
        if self.study is None:
            raise ValueError("No study has been created")

        import matplotlib.pyplot as plt

        fig = optuna.visualization.plot_param_importances(self.study)
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_parallel_coordinate(self, save_path: Optional[str] = None) -> None:
        """Plot parallel coordinate plot."""
        if self.study is None:
            raise ValueError("No study has been created")

        import matplotlib.pyplot as plt

        fig = optuna.visualization.plot_parallel_coordinate(self.study)
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()


def quick_optimize(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Quick hyperparameter optimization with sensible defaults."""
    optimizer = HyperparameterOptimizer(
        study_name="quick_optimize",
        n_trials=n_trials,
        timeout=timeout,
    )

    result = optimizer.optimize_attention_lstm(
        X_train, y_train, X_val, y_val
    )

    return result

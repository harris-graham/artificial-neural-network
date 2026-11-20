"""Model interpretability using SHAP and attention visualization."""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import shap
from tensorflow import keras


class SHAPExplainer:
    """SHAP explainer for neural network models."""

    def __init__(self, model: keras.Model, background_data: np.ndarray):
        """Initialize SHAP explainer.

        Args:
            model: Trained Keras model
            background_data: Background data for SHAP values computation
        """
        self.model = model
        self.background_data = background_data
        self.explainer: Optional[shap.Explainer] = None
        self._initialize_explainer()

    def _initialize_explainer(self) -> None:
        """Initialize the appropriate SHAP explainer."""
        try:
            # Try DeepExplainer for deep learning models
            self.explainer = shap.DeepExplainer(self.model, self.background_data)
        except Exception as e:
            # Fallback to GradientExplainer
            print(f"DeepExplainer failed, using GradientExplainer: {e}")
            self.explainer = shap.GradientExplainer(self.model, self.background_data)

    def get_shap_values(
        self,
        X: np.ndarray,
        nsamples: int = 100,
    ) -> np.ndarray:
        """Compute SHAP values for input data.

        Args:
            X: Input data to explain
            nsamples: Number of samples for approximation

        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")

        # Reshape if needed for time series
        original_shape = X.shape
        if len(original_shape) == 3:
            # Flatten for SHAP
            X_flat = X.reshape(X.shape[0], -1)
            shap_values = self.explainer.shap_values(X_flat, nsamples=nsamples)
            # Reshape back
            if isinstance(shap_values, list):
                shap_values = [sv.reshape(original_shape) for sv in shap_values]
            else:
                shap_values = shap_values.reshape(original_shape)
        else:
            shap_values = self.explainer.shap_values(X, nsamples=nsamples)

        return shap_values

    def plot_summary(
        self,
        X: np.ndarray,
        save_path: Optional[str] = None,
        max_display: int = 10,
    ) -> None:
        """Plot SHAP summary plot.

        Args:
            X: Input data
            save_path: Path to save the plot
            max_display: Maximum number of features to display
        """
        shap_values = self.get_shap_values(X)

        # Flatten for summary plot
        X_flat = X.reshape(X.shape[0], -1)
        if isinstance(shap_values, list):
            shap_values_flat = [sv.reshape(X.shape[0], -1) for sv in shap_values]
        else:
            shap_values_flat = shap_values.reshape(X.shape[0], -1)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values_flat,
            X_flat,
            max_display=max_display,
            show=False,
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_force(
        self,
        X: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot SHAP force plot for a single prediction.

        Args:
            X: Single input sample
            save_path: Path to save the plot
        """
        shap_values = self.get_shap_values(X[np.newaxis, :])

        # Flatten for force plot
        X_flat = X.reshape(-1)
        if isinstance(shap_values, list):
            shap_values_flat = shap_values[0].reshape(-1)
        else:
            shap_values_flat = shap_values.reshape(-1)

        shap.force_plot(
            self.explainer.expected_value,
            shap_values_flat,
            X_flat,
            matplotlib=True,
            show=False,
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_waterfall(
        self,
        X: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot SHAP waterfall plot.

        Args:
            X: Single input sample
            save_path: Path to save the plot
        """
        shap_values = self.get_shap_values(X[np.newaxis, :])

        # Flatten for waterfall plot
        X_flat = X.reshape(-1)
        if isinstance(shap_values, list):
            shap_values_flat = shap_values[0].reshape(-1)
        else:
            shap_values_flat = shap_values.reshape(-1)

        explanation = shap.Explanation(
            values=shap_values_flat,
            base_values=self.explainer.expected_value,
            data=X_flat,
        )

        shap.waterfall(explanation, show=False)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def get_feature_importance(
        self,
        X: np.ndarray,
    ) -> Dict[str, float]:
        """Get feature importance from SHAP values.

        Args:
            X: Input data

        Returns:
            Dictionary of feature importances
        """
        shap_values = self.get_shap_values(X)

        # Calculate mean absolute SHAP values
        if isinstance(shap_values, list):
            shap_values = np.abs(shap_values[0])
        else:
            shap_values = np.abs(shap_values)

        mean_importance = np.mean(shap_values, axis=0)

        # Create feature names
        n_features = mean_importance.shape[0]
        feature_names = [f"feature_{i}" for i in range(n_features)]

        importance_dict = dict(zip(feature_names, mean_importance.flatten()))
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return importance_dict


class AttentionVisualizer:
    """Visualize attention weights from attention-based models."""

    def __init__(self, model: keras.Model):
        """Initialize attention visualizer.

        Args:
            model: Trained model with attention mechanism
        """
        self.model = model
        self.attention_layer = self._find_attention_layer()

    def _find_attention_layer(self) -> Optional[keras.layers.Layer]:
        """Find the attention layer in the model."""
        for layer in self.model.layers:
            if "attention" in layer.name.lower():
                return layer
        return None

    def extract_attention_weights(
        self,
        X: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Extract attention weights from the model.

        Args:
            X: Input data

        Returns:
            Attention weights array or None if not available
        """
        if self.attention_layer is None:
            print("No attention layer found in model")
            return None

        # Create a new model that outputs attention weights
        # This is a simplified version - actual implementation depends on model architecture
        try:
            attention_model = keras.Model(
                inputs=self.model.inputs,
                outputs=self.attention_layer.output,
            )
            attention_weights = attention_model.predict(X)
            return attention_weights
        except Exception as e:
            print(f"Failed to extract attention weights: {e}")
            return None

    def plot_attention_weights(
        self,
        X: np.ndarray,
        save_path: Optional[str] = None,
        sample_idx: int = 0,
    ) -> None:
        """Plot attention weights heatmap.

        Args:
            X: Input data
            save_path: Path to save the plot
            sample_idx: Index of sample to visualize
        """
        attention_weights = self.extract_attention_weights(X)

        if attention_weights is None:
            return

        # Get attention for single sample
        if len(attention_weights.shape) > 2:
            weights = attention_weights[sample_idx]
        else:
            weights = attention_weights

        plt.figure(figsize=(10, 6))
        plt.imshow(weights, cmap="viridis", aspect="auto")
        plt.colorbar(label="Attention Weight")
        plt.xlabel("Timestep")
        plt.ylabel("Hidden State")
        plt.title("Attention Weights")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_attention_over_time(
        self,
        X: np.ndarray,
        save_path: Optional[str] = None,
        sample_idx: int = 0,
    ) -> None:
        """Plot attention weights over time.

        Args:
            X: Input data
            save_path: Path to save the plot
            sample_idx: Index of sample to visualize
        """
        attention_weights = self.extract_attention_weights(X)

        if attention_weights is None:
            return

        # Get attention for single sample
        if len(attention_weights.shape) > 2:
            weights = attention_weights[sample_idx]
        else:
            weights = attention_weights

        # Average over hidden states
        avg_attention = np.mean(weights, axis=-1) if len(weights.shape) > 1 else weights

        plt.figure(figsize=(12, 4))
        plt.plot(avg_attention, marker="o")
        plt.xlabel("Timestep")
        plt.ylabel("Average Attention Weight")
        plt.title("Attention Weights Over Time")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def create_explanation_report(
    model: keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    predictions: np.ndarray,
    output_dir: str = "explanations",
) -> Dict[str, Any]:
    """Create a comprehensive explanation report.

    Args:
        model: Trained model
        X: Input data
        y: True values
        predictions: Model predictions
        output_dir: Directory to save explanation artifacts

    Returns:
        Dictionary with explanation results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize SHAP explainer
    background_data = X[:min(100, len(X))]
    explainer = SHAPExplainer(model, background_data)

    # Compute SHAP values
    shap_values = explainer.get_shap_values(X[:10])

    # Get feature importance
    feature_importance = explainer.get_feature_importance(X)

    # Generate plots
    explainer.plot_summary(X[:50], save_path=os.path.join(output_dir, "summary.png"))
    explainer.plot_force(X[0], save_path=os.path.join(output_dir, "force.png"))

    # Calculate performance metrics
    mse = np.mean((predictions.flatten() - y.flatten()) ** 2)
    mae = np.mean(np.abs(predictions.flatten() - y.flatten()))

    # Try attention visualization
    attention_viz = AttentionVisualizer(model)
    attention_weights = attention_viz.extract_attention_weights(X[:10])

    if attention_weights is not None:
        attention_viz.plot_attention_weights(
            X[:10],
            save_path=os.path.join(output_dir, "attention_heatmap.png"),
        )

    report = {
        "mse": float(mse),
        "mae": float(mae),
        "feature_importance": feature_importance,
        "has_attention": attention_weights is not None,
        "explanation_dir": output_dir,
    }

    return report

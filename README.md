# Artificial Neural Network for Chaotic System Prediction

An advanced prediction system for chaotic time series using attention-enhanced neural networks with model interpretability and a modern FastAPI web API.

[![CI](https://github.com/harris/artificial-neural-network/workflows/CI/badge.svg)](https://github.com/harris/artificial-neural-network/actions)
[![CD](https://github.com/harris/artificial-neural-network/workflows/CD/badge.svg)](https://github.com/harris/artificial-neural-network/actions)
[![codecov](https://codecov.io/gh/harris/artificial-neural-network/branch/main/graph/badge.svg)](https://codecov.io/gh/harris/artificial-neural-network)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

- **Advanced Neural Network Architectures**
  - Attention-enhanced LSTM with Bahdanau/Luong attention
  - Bidirectional LSTM/GRU
  - Transformer-based models
  - Multi-layer configurations with dropout

- **Model Interpretability**
  - SHAP (SHapley Additive exPlanations) integration
  - Attention weight visualization
  - Feature importance analysis
  - Comprehensive explanation reports

- **Hyperparameter Optimization**
  - Optuna-powered Bayesian optimization
  - Automated hyperparameter search
  - Parameter importance analysis
  - Study persistence for resume capability

- **Modern FastAPI Web API**
  - RESTful endpoints for predictions, training, and interpretation
  - Pydantic schema validation
  - OpenAPI/Swagger documentation
  - Async support for scalability

- **Performance Benchmarking**
  - Training time benchmarks
  - Inference latency benchmarks
  - Model comparison across architectures
  - Automated scheduled benchmarks

- **Production Ready**
  - Comprehensive test suite with pytest
  - GitHub CI/CD pipeline
  - Docker containerization
  - Security scanning with Trivy

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/harris/artificial-neural-network.git
cd artificial-neural-network

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the API

```bash
# Start the FastAPI server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

### Basic Usage

#### Train a Model

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "attention_lstm",
    "A": 3.5,
    "epochs": 100,
    "batch_size": 32
  }'
```

#### Make Predictions

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "A": 3.5,
    "steps": 20
  }'
```

#### Model Interpretation

```bash
curl -X POST "http://localhost:8000/interpret" \
  -H "Content-Type: application/json" \
  -d '{
    "A": 3.5,
    "model_id": "your-model-id",
    "method": "shap"
  }'
```

## Architecture

### Project Structure

```
artificial-neural-network/
├── src/
│   ├── models/              # Neural network architectures
│   │   ├── attention_lstm.py
│   │   ├── model_factory.py
│   │   └── hyperopt.py
│   ├── api/                 # FastAPI application
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── services.py
│   └── utils/               # Utilities
│       ├── data.py
│       └── interpretability.py
├── tests/                   # Test suite
│   ├── test_models.py
│   ├── test_api.py
│   ├── test_utils.py
│   └── benchmarks.py
├── models/                  # Saved model artifacts
├── docs/                    # Documentation
└── .github/                 # GitHub workflows
```

### Model Architectures

- **LSTM**: Basic Long Short-Term Memory network
- **Attention LSTM**: LSTM with Bahdanau/Luong attention mechanism
- **GRU**: Gated Recurrent Unit
- **Attention GRU**: GRU with attention
- **Transformer**: Multi-head self-attention architecture

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Make predictions for logistic map sequences |
| `/train` | POST | Train a new model |
| `/models` | GET | List all available models |
| `/models/{id}` | GET | Get model information |
| `/models/{id}` | DELETE | Delete a model |
| `/interpret` | POST | Generate model interpretation |
| `/models/{id}/metrics` | GET | Get model performance metrics |
| `/hyperopt` | POST | Run hyperparameter optimization |
| `/health` | GET | Health check |

For detailed API documentation, visit `/docs` when the server is running.

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py -v

# Run benchmarks (marked with @pytest.mark.benchmark)
pytest tests/benchmarks.py -v -s -m benchmark
```

### Code Quality

```bash
# Lint with Ruff
ruff check src/ tests/

# Format code
ruff format src/ tests/

# Type checking with MyPy
mypy src/ --ignore-missing-imports
```

### Hyperparameter Optimization

```python
from src.models.hyperopt import HyperparameterOptimizer
from src.utils.data import generate_logistic_sequence, prepare_training_data

# Generate data
sequence = generate_logistic_sequence(A=3.5, n_steps=500)
X_train, y_train, X_val, y_val, _, _ = prepare_training_data(sequence)

# Run optimization
optimizer = HyperparameterOptimizer(n_trials=50)
result = optimizer.optimize_attention_lstm(X_train, y_train, X_val, y_val)

print(f"Best parameters: {result['best_params']}")
print(f"Best score: {result['best_score']}")
```

## Performance Benchmarks

Benchmarks compare different model architectures on training time, inference latency, and prediction accuracy.

| Architecture | Training Time (s) | Inference Time (ms) | MSE |
|--------------|------------------|---------------------|-----|
| LSTM | 45.2 | 12.5 | 0.0012 |
| Attention LSTM | 52.8 | 15.3 | 0.0008 |
| GRU | 38.6 | 10.2 | 0.0015 |
| Transformer | 65.4 | 18.7 | 0.0007 |

*Results from standard benchmark on logistic map A=3.5, 500 steps*

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow/Keras for deep learning framework
- Optuna for hyperparameter optimization
- SHAP for model interpretability
- FastAPI for the web framework

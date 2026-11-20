# Contributing to Artificial Neural Network

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/artificial-neural-network.git
   cd artificial-neural-network
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If it exists
   ```

## Code Style

We use Ruff for linting and formatting:

```bash
# Check code style
ruff check src/ tests/

# Format code
ruff format src/ tests/
```

## Type Checking

We use MyPy for type checking:

```bash
mypy src/ --ignore-missing-imports
```

## Testing

Run tests before submitting a PR:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and commit them with clear messages
3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
4. Create a pull request from your fork to the main repository

## Commit Message Convention

We follow conventional commits:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test changes
- `refactor:` for code refactoring
- `chore:` for maintenance tasks

Example:
```
feat: add transformer model architecture
fix: correct attention layer bug
docs: update API documentation
```

## Adding New Features

When adding new features:

1. Add tests for the new functionality
2. Update the documentation (README, API docs)
3. Ensure all existing tests pass
4. Add type hints to new functions
5. Update the CHANGELOG if applicable

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages/stack traces

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

# Contributing to Quantum Transformer

Thank you for your interest in contributing to Quantum Transformer!

## Getting Started

### Development Setup

```bash
# Clone repository
git clone https://github.com/quantum-ai/q-genesis.git
cd q-genesis

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=qgenesis --cov-report=html

# Run specific test
pytest tests/test_quantum_transformer.py -v
```

## Code Style

We use:
- **Black** for formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

```bash
black qgenesis/ tests/
isort qgenesis/ tests/
flake8 qgenesis/
mypy qgenesis/
```

## Pull Request Process

1. Fork the repository
2. Create a branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Ensure tests pass
5. Submit PR

### PR Checklist

- [ ] Tests pass
- [ ] Code is formatted
- [ ] Documentation updated
- [ ] CHANGELOG entry added

## Code Organization

```
qgenesis/
├── core/           # Quantum Transformer core
├── attention/      # Quantum attention (SWAP test)
├── encoding/       # Positional and data encoding
├── layers/         # Quantum FFN, normalization
├── circuits/       # Quantum circuit implementations
├── molecular/      # Molecular applications
├── optimization/   # Quantum optimizers
└── utils/          # Metrics, visualization
```

## Adding New Quantum Attention Type

1. Add implementation to `qgenesis/attention/quantum_attention.py`
2. Register in `qgenesis/attention/__init__.py`
3. Add tests
4. Update documentation

## Questions?

Open an issue for bugs or feature requests.

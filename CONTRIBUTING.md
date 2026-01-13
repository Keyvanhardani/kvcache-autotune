# Contributing to KVCache Auto-Tuner

Thank you for your interest in contributing to KVCache Auto-Tuner! This document provides guidelines and information for contributors.

## Code of Conduct

Be respectful, inclusive, and constructive. We welcome contributors of all backgrounds and experience levels.

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- (Optional) CUDA-capable GPU for full testing

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/kvcache-autotune.git
cd kvcache-autotune

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[full,dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=kvat --cov-report=html

# Run specific test file
pytest tests/test_schema.py -v

# Run tests excluding GPU tests
pytest tests/ -v -m "not gpu"
```

### Code Style

We use `ruff` for linting and formatting:

```bash
# Check code
ruff check kvat/

# Fix auto-fixable issues
ruff check kvat/ --fix

# Format code
ruff format kvat/
```

### Type Checking

```bash
mypy kvat/
```

## How to Contribute

### Reporting Issues

1. Check if the issue already exists
2. Use the issue template
3. Include:
   - Python version
   - Package versions (`pip list`)
   - Full error traceback
   - Minimal reproduction code

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Add/update tests
5. Run tests and linting
6. Commit with clear messages
7. Push to your fork
8. Open a Pull Request

### Commit Messages

Follow conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `chore`: Maintenance

Examples:
```
feat(profiles): add streaming workload profile
fix(transformers): handle missing pad token
docs(readme): add installation instructions
test(metrics): add scoring edge cases
```

## Project Structure

```
kvat/
├── core/           # Core logic (schema, metrics, profiles, search)
├── engines/        # Engine adapters (transformers, vllm, etc.)
├── probes/         # Resource monitoring (gpu, cpu)
└── cli.py          # Command-line interface

tests/              # Unit tests
examples/           # Usage examples
```

## Adding New Features

### New Engine Adapter

1. Create `kvat/engines/your_engine.py`
2. Implement `EngineAdapter` interface
3. Add tests in `tests/test_your_engine.py`
4. Update documentation

### New Workload Profile

1. Add to `kvat/core/profiles.py`
2. Include in `BUILTIN_PROFILES`
3. Add tests
4. Document in README

### New Metric

1. Add to `kvat/core/schema.py`
2. Update `MetricsCollector` in `kvat/core/metrics.py`
3. Update scoring logic
4. Add tests

## Testing Guidelines

- Aim for >80% coverage
- Test edge cases
- Use fixtures for common setup
- Mark slow/GPU tests with pytest markers

```python
@pytest.mark.slow
def test_large_model():
    ...

@pytest.mark.gpu
def test_cuda_memory():
    ...
```

## Documentation

- Update README for user-facing changes
- Add docstrings to public functions
- Include examples in docstrings
- Update CONTEXT.md for architectural changes

## Questions?

- Open a GitHub issue
- Tag with `question` label

Thank you for contributing!

# Testing Guide - Pytest and CI/CD

## Overview

All tests in the MatFormPPO project have been converted to use **pytest** framework with comprehensive CI/CD integration via GitHub Actions.

## Quick Start

### Install Testing Dependencies

```bash
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/quick_test.py

# Run tests with specific markers
pytest -m "quick"
pytest -m "unit"
pytest -m "integration"
```

## Test Structure

### Test Categories (Markers)

Tests are organized with pytest markers:

- **`@pytest.mark.quick`**: Fast smoke tests (< 5s)
- **`@pytest.mark.unit`**: Unit tests for individual components
- **`@pytest.mark.integration`**: Integration tests for full pipelines
- **`@pytest.mark.ppo`**: PPO reinforcement learning tests
- **`@pytest.mark.slow`**: Long-running tests (> 30s)

### Test Files

| File | Purpose | Test Count | Markers |
|------|---------|------------|---------|
| `quick_test.py` | System validation tests | 5 | quick, unit, integration |
| `test_ppo_enhancements.py` | PPO trainer enhancements | 2 | ppo, unit, integration |
| `test_method_masking.py` | Method-level masking | 2 | ppo, unit, integration |
| `test_gnn_kg_placeholders.py` | GNN & KG processing | 5 | unit |
| `test_ppo_simple.py` | PPO basic functionality | 7 | unit, ppo, integration, slow |
| `test_pipeline.py` | Pipeline execution | 7 | integration, slow |

## Running Tests by Category

### Quick Validation (< 2 minutes)

```bash
pytest -m "quick or unit" -v
```

### Integration Tests

```bash
pytest -m "integration" -v
```

### PPO-Specific Tests

```bash
pytest -m "ppo" -v
```

### Skip Slow Tests

```bash
pytest -m "not slow" -v
```

## Test Configuration

### pytest.ini

Configuration file defines:
- Test discovery patterns
- Markers for categorization
- Logging settings
- Output formatting

### Key Settings

```ini
[pytest]
python_files = test_*.py
python_classes = Test*
python_functions = test_*
testpaths = tests
markers =
    unit: Unit tests
    integration: Integration tests
    ppo: PPO RL tests
    slow: Slow tests
    quick: Quick smoke tests
```

## CI/CD Pipeline

### GitHub Actions Workflow

**File**: `.github/workflows/ci.yml`

### Workflow Triggers

- **Push** to `main` or `2025-11-18-flexible-pipeline` branches
- **Pull requests** to `main`
- **Manual trigger** via `workflow_dispatch`

### CI Jobs

#### 1. Test Matrix

- **Operating Systems**: Ubuntu, Windows
- **Python Versions**: 3.8, 3.9, 3.10
- **Test Categories**: Quick, Unit, Integration

#### 2. Code Quality (Lint)

- Black code formatting check
- isort import sorting check
- flake8 linting

#### 3. Coverage Report

- Test coverage analysis
- HTML coverage report generation
- Coverage artifact upload

### Workflow Status

Check CI/CD status: `Actions` tab in GitHub repository

## Advanced Usage

### Run with Coverage

```bash
# Generate coverage report
pytest --cov=. --cov-report=html -v

# View coverage report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

### Run Tests in Parallel

```bash
pip install pytest-xdist
pytest -n auto  # Use all CPU cores
```

### Run with Timeout

```bash
pytest --timeout=300  # 5-minute timeout per test
```

### Debugging Failed Tests

```bash
# Show local variables on failure
pytest --showlocals -v

# Drop into debugger on failure
pytest --pdb

# Stop at first failure
pytest -x
```

### Custom Test Selection

```bash
# Run tests matching pattern
pytest -k "test_environment"

# Run specific test class
pytest tests/test_pipeline.py::test_basic_pipeline

# Run tests in specific file
pytest tests/quick_test.py -v
```

## Test Fixtures

### Available Fixtures

#### `sample_pipeline_data`

Provides sample data for pipeline testing:

```python
@pytest.fixture
def sample_pipeline_data():
    return {
        'X_train': np.random.rand(100, 10),
        'y_train': np.random.rand(100),
        'X_val': np.random.rand(20, 10),
        'y_val': np.random.rand(20)
    }
```

#### `fake_data`

Generates fake test data for GNN/KG tests:

```python
@pytest.fixture
def fake_data():
    n, d = 10, 4
    X = np.random.randn(n, d)
    return {
        'X_train': X,
        'X_val': X.copy(),
        'y_train': np.random.randn(n),
        'y_val': np.random.randn(n),
        'feature_names': [f"f{i}" for i in range(d)]
    }
```

#### `mock_training_data`

Mock PPO training data for visualization tests:

```python
@pytest.fixture
def mock_training_data():
    episodes = range(1, 21)
    rewards = np.linspace(-1.0, 0.5, 20) + np.random.normal(0, 0.1, 20)
    losses = np.exp(-np.linspace(0, 2, 20)) + np.random.normal(0, 0.05, 20)
    return episodes, rewards, losses
```

## Writing New Tests

### Test Function Template

```python
import pytest
import numpy as np

@pytest.mark.unit  # Add appropriate marker
def test_my_feature():
    """Test description"""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = my_function(input_data)
    
    # Assert
    assert result is not None
    assert 'expected_key' in result
```

### Parametrized Tests

```python
@pytest.mark.parametrize("strategy", ['entity', 'relation', 'none'])
def test_kg_strategies(strategy):
    """Test all KG strategies"""
    result = kg_process(data, strategy=strategy)
    assert 'X_train' in result
```

### Using Fixtures

```python
def test_with_fixture(sample_pipeline_data):
    """Test using fixture"""
    X_train = sample_pipeline_data['X_train']
    assert X_train.shape == (100, 10)
```

## Continuous Integration

### On Push

1. Tests run automatically on push to main branches
2. Multiple OS and Python version combinations tested
3. Results visible in GitHub Actions tab

### Pull Request Checks

1. All tests must pass before merging
2. Code quality checks enforced
3. Coverage report generated

### Manual Workflow

Trigger manually from GitHub Actions:
1. Go to `Actions` tab
2. Select `CI/CD Pipeline` workflow
3. Click `Run workflow`

## Troubleshooting

### Common Issues

#### Import Errors

```bash
# Ensure project root is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

#### Test Collection Errors

```bash
# Check syntax with pytest --collect-only
pytest --collect-only tests/

# Fix indentation errors in test files
```

#### Slow Test Performance

```bash
# Skip slow tests during development
pytest -m "not slow" -v

# Run only quick tests
pytest -m "quick" -v
```

## Test Results

### Current Status

✅ **16/16 tests passing** (quick + unit tests)
- Configuration tests: PASS
- Module imports: PASS
- N5 Knowledge Graph: PASS
- Environment flexibility: PASS
- Visualization files: PASS
- PPO enhancements: PASS
- Method masking: PASS
- GNN/KG placeholders: PASS

### Performance

- Quick tests: ~95 seconds
- Unit tests: ~95 seconds
- Integration tests: ~2-5 minutes
- Full test suite: ~5-10 minutes

## Best Practices

1. **Write tests first** (TDD approach)
2. **Use appropriate markers** for categorization
3. **Keep tests independent** - no shared state
4. **Use fixtures** for common test data
5. **Mock external dependencies** (API calls, file I/O)
6. **Test edge cases** and error conditions
7. **Maintain fast unit tests** (< 1s each)
8. **Document test purpose** in docstrings

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Markers](https://docs.pytest.org/en/stable/how-to/mark.html)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html)
- [GitHub Actions Docs](https://docs.github.com/en/actions)

## Next Steps

1. ✅ All test files converted to pytest
2. ✅ CI/CD pipeline configured
3. ✅ Quick tests passing
4. 🔄 Run full integration tests
5. 🔄 Add more test coverage
6. 🔄 Set up coverage thresholds

---

**Last Updated**: November 18, 2025
**Test Framework**: pytest 8.3.5
**Python Version**: 3.8.5

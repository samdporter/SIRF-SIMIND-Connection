# Contributing to SIRF-SIMIND-Connection

We welcome contributions to SIRF-SIMIND-Connection! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/SIRF-SIMIND-Connection.git
   cd SIRF-SIMIND-Connection
   ```

3. Create a virtual environment and install development dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Process

### 1. Create a Branch

Create a feature branch for your changes:
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clean, readable code following PEP 8
- Add type hints to all function signatures
- Include docstrings for all public functions/classes
- Update tests for new functionality
- Update documentation as needed

### 3. Code Style

We use several tools to maintain code quality:

- **autoflake** for cleaning unused imports and variables
- **Black** for code formatting
- **isort** for import sorting
- **Ruff** for line checking

Run all checks:
```bash
# Use the handy script
chmod +x ./scripts/fix.sh
./fix.sh

# Or use pre-commit to run all checks
pre-commit run --all-files
```

### 4. Testing

At the moment, I don't have any tests (whoops!) But the minimal example files do a decent job of testing some of the main functionality. This will come soon.

Write tests for new functionality:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sirf_simind_connection --cov-report=html

# Run specific test file
pytest tests/test_example.py

# Run with verbose output
pytest -v
```

Test categories:
- Unit tests: Test individual functions/methods
- Integration tests: Test component interactions
- End-to-end tests: Test complete workflows

### 5. Documentation
Again, this is a bit poor at the moment but improvements will follow

- Update docstrings using NumPy style
- Update relevant .md files
- Add examples if introducing new features
- Build docs locally to check:
  ```bash
  cd docs
  make html
  ```

## Submitting Changes

### 1. Commit Your Changes

Write clear, descriptive commit messages:
```bash
git add .
git commit -m "Add feature: description of what you added"
```

Commit message format:
- Start with a verb (Add, Fix, Update, etc.)
- Keep the first line under 50 characters
- Add detailed description after a blank line if needed

### 2. Push to GitHub

```bash
git push origin feature/your-feature-name
```

### 3. Create a Pull Request

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your branch
4. Fill out the PR template:
   - Describe what changes you made
   - Link any related issues
   - List any breaking changes
   - Include test results

### 4. Code Review

- Address reviewer feedback
- Push additional commits to your branch
- Mark conversations as resolved
- Request re-review when ready

## Pull Request Guidelines

### What we look for:

- **Tests**: New features must include tests
- **Documentation**: Code must be documented
- **Style**: Code must follow project style guidelines
- **Compatibility**: Changes must not break existing functionality
- **Performance**: Consider performance implications

### PR Checklist:

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] PR description explains the change
- [ ] No merge conflicts

## Reporting Issues

### Bug Reports

Include:
- Python version
- SIRF version
- SIMIND version
- Operating system
- Minimal code example reproducing the issue
- Full error traceback

### Feature Requests

Include:
- Use case description
- Expected behavior
- Example code (if applicable)
- Why this would benefit other users

## Development Tips

### Setting up SIMIND for Testing

1. Create a test configuration:
   ```python
   from sirf_simind_connection import SimulationConfig
   config = SimulationConfig('tests/fixtures/test_config.smc')
   ```

2. Use small test images to speed up tests
3. Mock SIMIND calls when testing non-simulation logic

### Common Issues

1. **Import errors**: Ensure SIRF is properly installed
2. **SIMIND not found**: Check PATH configuration
3. **File format issues**: Use provided conversion utilities

### Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging for specific module
logger = logging.getLogger('sirf_simind_connection.simulator')
logger.setLevel(logging.DEBUG)
```

## Code Organization

- `sirf_simind_connection/`
   - `builders/`: STIR object builders
   - `configs/`: Example configurations files
   - `core/`: Core functionality (simulator, projector, config)
   - `converters/`: File format converters
   - `data/`: Data used for SIMIND simulations (just attenuation correction at the moment)
   - `utils/`: Utility functions

- `examples/`: Usage examples
- `docs/`: Documentation source
- `tests/`: Currently empty. Will contain everything soon (I hope)

## Release Process

1. Update version in `pyproject.toml` and `__init__.py`
2. Update CHANGELOG.md
3. Create release PR
4. After merge, tag release: `git tag v1.2.3`
5. Push tag: `git push origin v1.2.3`
6. GitHub Actions will handle PyPI deployment

## Questions?

- Open an issue for questions
- Join discussions in existing issues
- Contact maintainers if needed

Thank you for contributing to SIRF-SIMIND-Connection!
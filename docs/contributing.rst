.. _contributing:

Contributing
============

.. include:: ../CONTRIBUTING.md
   :parser: myst_parser.sphinx_

Development Setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a development branch
4. Install in development mode:

   ```bash
   pip install -e ".[dev]"
   ```

5. Make your changes
6. Run tests to ensure everything works
7. Submit a pull request

Testing Your Changes
--------------------

Before submitting:

```bash
# Run all tests
python -m pytest tests/ -v

# Check code formatting
black --check .
ruff check .
isort --check-only .
```

Documentation Updates
---------------------

If you make changes that affect the documentation:

```bash
cd docs/
make html  # Build documentation locally
```

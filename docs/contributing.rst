.. _contributing:

Contributing
============

Contributions are welcome. For code changes, please follow the development and
testing checklist below before opening a pull request.

Development Setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a development branch
4. Install in development mode:

   .. code-block:: bash

      pip install -e ".[dev]"

5. Make your changes
6. Run tests to ensure everything works
7. Submit a pull request

Testing Your Changes
--------------------

Before submitting:

.. code-block:: bash

    # Run all tests
    python -m pytest tests/ -v

    # Check code formatting
    black --check .
    ruff check .
    isort --check-only .

Documentation Updates
---------------------

If you make changes that affect the documentation:

.. code-block:: bash

    cd docs/
    make html  # Build documentation locally

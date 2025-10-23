.. _testing:

Testing
=======

This document explains the testing strategy for SIRF-SIMIND-Connection, which handles the challenge of testing code that depends on external dependencies (SIRF and SIMIND) that may not be available in CI environments.

Test Organization
-----------------

Test Markers
~~~~~~~~~~~~

Tests are organized using pytest markers to indicate their dependencies:

* ``@pytest.mark.unit`` - Pure unit tests with no external dependencies (CI-friendly)
* ``@pytest.mark.requires_sirf`` - Tests that require SIRF to be installed
* ``@pytest.mark.requires_simind`` - Tests that require SIMIND command-line tool
* ``@pytest.mark.integration`` - Integration tests (may be slow)
* ``@pytest.mark.ci_skip`` - Tests to skip in CI environments

Automatic Dependency Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The test suite automatically detects available dependencies:

* **SIRF detection**: Attempts to ``import sirf``
* **SIMIND detection**: Checks if ``simind`` command is available
* **CI detection**: Checks ``CI`` or ``GITHUB_ACTIONS`` environment variables

Tests requiring unavailable dependencies are automatically skipped with informative messages.

Running Tests
-------------

Local Development (All Tests)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Run all available tests
    pytest

    # Run only unit tests (fast)
    pytest -m unit

    # Run only SIRF-dependent tests
    pytest -m requires_sirf

    # Run with coverage
    pytest --cov=sirf_simind_connection --cov-report=html

CI Environment (GitHub Actions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GitHub Actions workflow runs only CI-friendly tests:

.. code-block:: bash

    # CI command (no SIRF/SIMIND dependencies)
    pytest -m "not requires_sirf and not requires_simind and not ci_skip"

Alternative CI Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also use the dedicated CI pytest configuration:

.. code-block:: bash

    # Using CI-specific config
    pytest -c pytest-ci.ini

Test Categories
---------------

CI-Friendly Test Suites
~~~~~~~~~~~~~~~~~~~~~~~

The default CI job runs the tests that avoid heavyweight dependencies. These focus on:

* **Conversion maths** – ``tests/test_schneider_density.py`` exercises the HU-to-density pipelines.
* **Configuration & builders** – ``tests/test_simulation_config.py`` and ``tests/test_acquisition_builder_unit.py`` validate YAML/SMC handling plus Interfile header generation without a SIRF runtime.
* **Utility helpers** – ``tests/test_components.py``, ``tests/test_utils.py``, and ``tests/test_utils_small.py`` cover enums, file parsing, and temporary-directory/energy-window helpers.
* **Backend guards** – ``tests/test_backends.py`` ensures the dynamic backend selection code behaves without importing SIRF.

SIRF-Dependent Suites
~~~~~~~~~~~~~~~~~~~~~

Tests that construct real SIRF/STIR objects carry the ``requires_sirf`` marker and are skipped in CI. These include:

* ``tests/test_simind_simulator.py`` and ``tests/test_simind_projector.py`` for the legacy projector/simulator path.
* ``tests/test_arithmetic_operations.py`` for wrapped acquisition arithmetic.
* ``tests/test_coordinator.py`` for the SIMIND coordinator.
* SIRF-backed helpers in ``tests/test_utils.py`` and ``tests/test_components.py``.

CIL and Hybrid Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two suites rely on additional packages:

* ``tests/test_cil_partitioner.py`` (``requires_sirf`` + ``requires_cil``) exercises the CIL partitioner adapter.
* ``tests/test_step_size_rules.py`` (``requires_cil``) checks Armijo step-size helpers with lightweight stand-ins.

Integration Test
~~~~~~~~~~~~~~~~

``tests/test_integration.py`` orchestrates a full example run and needs both SIMIND and SIRF. It is marked with ``integration``, ``requires_simind``, and ``requires_sirf``.

Configuration Files
-------------------

pytest.ini (Local Development)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Runs all available tests based on detected dependencies
* Includes verbose output and duration reporting

pytest-ci.ini (CI Environment)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Specifically filters out dependency-requiring tests
* Optimized for GitHub Actions environment

tests/conftest.py
~~~~~~~~~~~~~~~~

* Configures pytest markers
* Implements automatic dependency detection and test skipping
* Handles CI environment detection

Adding New Tests
----------------

When adding new tests, use appropriate markers:

.. code-block:: python

    import pytest

    @pytest.mark.unit
    def test_pure_python_logic():
        """Test that doesn't need external dependencies."""
        assert True

    @pytest.mark.requires_sirf  
    def test_sirf_functionality():
        """Test that uses SIRF objects."""
        from sirf.STIR import ImageData
        # ... test code

    @pytest.mark.requires_simind
    def test_simind_execution():
        """Test that calls SIMIND command."""
        # ... test code that runs simind

This ensures your tests will be properly categorized and run in the appropriate environments.

Continuous Integration
----------------------

GitHub Actions is used to run tests automatically. The CI workflow:

1. **Installs only basic Python dependencies** (no SIRF/SIMIND)
2. **Runs code quality checks** (black, isort, ruff)
3. **Executes CI-friendly tests** using dependency markers
4. **Generates coverage reports** for the tested code
5. **Builds and validates the package**

This approach ensures reliable CI while maintaining comprehensive test coverage for local development.

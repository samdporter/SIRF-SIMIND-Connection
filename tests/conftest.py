import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may be slow)"
    )
    config.addinivalue_line(
        "markers", "requires_sirf: marks tests that require SIRF to be installed"
    )
    config.addinivalue_line(
        "markers", "requires_simind: marks tests that require SIMIND to be installed"
    )
    config.addinivalue_line(
        "markers", "ci_skip: marks tests to skip in CI environments"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as pure unit tests (CI-friendly)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available dependencies."""

    # Check if SIRF is available
    try:
        sirf_available = True
    except ImportError:
        sirf_available = False

    # Check if SIMIND is available (command line tool)
    import shutil

    simind_available = shutil.which("simind") is not None

    # Check if we're in CI environment
    import os

    in_ci = (
        os.getenv("CI", "false").lower() == "true"
        or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
    )

    skip_sirf = pytest.mark.skip(reason="SIRF not available")
    skip_simind = pytest.mark.skip(reason="SIMIND not available")
    skip_ci = pytest.mark.skip(reason="Skipped in CI environment")

    for item in items:
        # Skip SIRF-dependent tests if SIRF not available
        if "requires_sirf" in item.keywords and not sirf_available:
            item.add_marker(skip_sirf)

        # Skip SIMIND-dependent tests if SIMIND not available
        if "requires_simind" in item.keywords and not simind_available:
            item.add_marker(skip_simind)

        # Skip CI-incompatible tests in CI environments
        if "ci_skip" in item.keywords and in_ci:
            item.add_marker(skip_ci)

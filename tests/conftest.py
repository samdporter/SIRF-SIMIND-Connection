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
        "markers", "requires_stir: marks tests that require STIR Python to be installed"
    )
    config.addinivalue_line(
        "markers", "requires_simind: marks tests that require SIMIND to be installed"
    )
    config.addinivalue_line(
        "markers", "requires_cil: marks tests that require CIL to be installed"
    )
    config.addinivalue_line(
        "markers", "requires_setr: marks tests that require SETR to be installed"
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
        import sirf.STIR  # noqa: F401

        sirf_available = True
    except ImportError:
        sirf_available = False

    # Check if STIR Python is available
    try:
        import stir  # noqa: F401
        import stirextra  # noqa: F401

        stir_available = True
    except ImportError:
        stir_available = False

    # Check if CIL is available
    try:
        import cil  # noqa: F401

        cil_available = True
    except ImportError:
        cil_available = False

    # Check if SETR is available
    try:
        import setr  # noqa: F401

        setr_available = True
    except ImportError:
        setr_available = False

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
    skip_stir = pytest.mark.skip(reason="STIR Python not available")
    skip_cil = pytest.mark.skip(reason="CIL not available")
    skip_setr = pytest.mark.skip(reason="SETR not available")
    skip_simind = pytest.mark.skip(reason="SIMIND not available")
    skip_ci = pytest.mark.skip(reason="Skipped in CI environment")

    for item in items:
        # Skip SIRF-dependent tests if SIRF not available
        if "requires_sirf" in item.keywords and not sirf_available:
            item.add_marker(skip_sirf)

        # Skip STIR-dependent tests if STIR not available
        if "requires_stir" in item.keywords and not stir_available:
            item.add_marker(skip_stir)

        # Skip CIL-dependent tests if CIL not available
        if "requires_cil" in item.keywords and not cil_available:
            item.add_marker(skip_cil)

        # Skip SETR-dependent tests if SETR not available
        if "requires_setr" in item.keywords and not setr_available:
            item.add_marker(skip_setr)

        # Skip SIMIND-dependent tests if SIMIND not available
        if "requires_simind" in item.keywords and not simind_available:
            item.add_marker(skip_simind)

        # Skip CI-incompatible tests in CI environments
        if "ci_skip" in item.keywords and in_ci:
            item.add_marker(skip_ci)

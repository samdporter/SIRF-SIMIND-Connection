from importlib import metadata

import pytest

import simind_python_connector


pytestmark = pytest.mark.unit


def test_public_import_package_is_simind_python_connector():
    assert simind_python_connector.__name__ == "simind_python_connector"


def test_distribution_version_matches_import_package_version():
    assert simind_python_connector.__version__ == metadata.version(
        "simind-python-connector"
    )

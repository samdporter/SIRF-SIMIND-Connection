from importlib import metadata

import pytest

import py_smc


pytestmark = pytest.mark.unit


def test_public_import_package_is_py_smc():
    assert py_smc.__name__ == "py_smc"


def test_distribution_version_matches_import_package_version():
    assert py_smc.__version__ == metadata.version("py-smc")

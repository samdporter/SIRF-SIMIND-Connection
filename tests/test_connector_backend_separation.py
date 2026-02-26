import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONNECTORS_DIR = ROOT / "sirf_simind_connection" / "connectors"


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.unit
def test_stir_adaptor_module_is_backend_specific():
    imports = _import_roots(CONNECTORS_DIR / "stir_adaptor.py")
    assert "sirf" not in imports
    assert "pytomography" not in imports


@pytest.mark.unit
def test_sirf_adaptor_module_is_backend_specific():
    imports = _import_roots(CONNECTORS_DIR / "sirf_adaptor.py")
    assert "stir" not in imports
    assert "pytomography" not in imports


@pytest.mark.unit
def test_pytomography_adaptor_module_is_backend_specific():
    imports = _import_roots(CONNECTORS_DIR / "pytomography_adaptor.py")
    assert "sirf" not in imports
    assert "stir" not in imports

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


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
@pytest.mark.parametrize(
    "name",
    (
        "01_basic_simulation.py",
        "02_runtime_switch_comparison.py",
        "03_multi_window.py",
        "04_custom_config.py",
        "05_scattwin_vs_penetrate_comparison.py",
        "06_schneider_density_conversion.py",
    ),
)
def test_01_to_06_examples_are_python_connector_only(name: str):
    imports = _import_roots(ROOT / "examples" / name)
    assert "sirf" not in imports
    assert "stir" not in imports
    assert "pytomography" not in imports


@pytest.mark.unit
def test_07a_stir_example_is_stir_only():
    imports = _import_roots(ROOT / "examples" / "07A_stir_adaptor_osem.py")
    assert "stir" in imports
    assert "sirf" not in imports
    assert "pytomography" not in imports


@pytest.mark.unit
def test_07b_sirf_example_is_sirf_only():
    imports = _import_roots(ROOT / "examples" / "07B_sirf_adaptor_osem.py")
    assert "sirf" in imports
    assert "stir" not in imports
    assert "pytomography" not in imports


@pytest.mark.unit
def test_07c_pytomography_example_is_pytomography_only():
    imports = _import_roots(ROOT / "examples" / "07C_pytomography_adaptor_osem.py")
    assert "pytomography" in imports
    assert "sirf" not in imports
    assert "stir" not in imports


@pytest.mark.unit
def test_07_examples_use_minimal_example_yaml():
    for name in (
        "07A_stir_adaptor_osem.py",
        "07B_sirf_adaptor_osem.py",
        "07C_pytomography_adaptor_osem.py",
    ):
        text = (ROOT / "examples" / name).read_text()
        assert 'configs.get("Example.yaml")' in text

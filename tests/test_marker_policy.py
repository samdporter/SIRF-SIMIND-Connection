"""Policy checks for dependency markers across the test suite."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent
ALLOWED_MARKERS = {
    "unit",
    "integration",
    "ci_skip",
    "requires_sirf",
    "requires_stir",
    "requires_simind",
    "requires_pytomography",
    "requires_cil",
    "requires_setr",
}


def _has_supported_module_marker(source: str, tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            has_pytestmark_target = any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in node.targets
            )
            if not has_pytestmark_target:
                continue
            segment = ast.get_source_segment(source, node) or ""
            if any(marker in segment for marker in ALLOWED_MARKERS):
                return True
    return False


def _has_supported_function_marker(source: str, node: ast.FunctionDef) -> bool:
    for decorator in node.decorator_list:
        segment = ast.get_source_segment(source, decorator) or ""
        if "pytest.mark" in segment and any(
            marker in segment for marker in ALLOWED_MARKERS
        ):
            return True
    return False


def test_all_tests_have_dependency_or_category_marker() -> None:
    missing: list[str] = []

    for path in sorted(ROOT.glob("test_*.py")):
        source = path.read_text()
        tree = ast.parse(source)
        module_marked = _has_supported_module_marker(source, tree)

        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue
            if module_marked or _has_supported_function_marker(source, node):
                continue
            missing.append(f"{path.name}:{node.name}")

    assert not missing, (
        "Every test must have a dependency/category marker via function decorator "
        "or module-level pytestmark. Missing:\n- "
        + "\n- ".join(missing)
    )

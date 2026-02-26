import importlib
import os

import pytest


EXPECTATIONS = {
    "python": {
        "allowed": (),
        "blocked": ("sirf", "stir", "stirextra", "pytomography"),
    },
    "stir": {
        "allowed": ("stir", "stirextra"),
        "blocked": ("sirf", "pytomography"),
    },
    "sirf": {
        "allowed": ("sirf.STIR",),
        "blocked": ("stir", "stirextra", "pytomography"),
    },
    "pytomography": {
        "allowed": ("pytomography",),
        "blocked": ("sirf", "stir", "stirextra"),
    },
}


@pytest.mark.unit
def test_container_backend_library_isolation():
    backend = os.environ.get("SIMIND_CONNECTOR_BACKEND")
    if backend not in EXPECTATIONS:
        pytest.skip("Container backend isolation check only runs in docker services")

    expected = EXPECTATIONS[backend]

    for module in expected["allowed"]:
        importlib.import_module(module)

    for module in expected["blocked"]:
        with pytest.raises(ImportError):
            importlib.import_module(module)

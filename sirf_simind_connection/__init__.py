"""
SIRF ⇄ SIMIND connector – public API.

>>> from sirf_simind_connection import SimindSimulator
"""

import importlib
from importlib import metadata as _meta
from typing import Any


try:  # installed (pip/poetry)
    __version__ = _meta.version(__name__)
except _meta.PackageNotFoundError:  # editable / source checkout
    __version__ = "0.3.0"


def __getattr__(name: str) -> Any:
    if name in {"builders", "configs", "converters", "core", "data", "utils"}:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    elif name in {"SimindSimulator", "SimulationConfig"}:
        core = importlib.import_module(".core", __name__)
        obj = getattr(core, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SimulationConfig",
    "SimindSimulator",
    "builders",
    "configs",
    "converters",
    "core",
    "data",
    "utils",
]

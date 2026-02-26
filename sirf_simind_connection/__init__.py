"""
SIRF ⇄ SIMIND connector/adaptor API.

>>> from sirf_simind_connection import SimindSimulator
"""

import importlib
from importlib import metadata as _meta
from typing import Any


try:  # installed (pip/poetry)
    __version__ = _meta.version(__name__)
except _meta.PackageNotFoundError:  # editable / source checkout
    __version__ = "0.4.0"


def __getattr__(name: str) -> Any:
    if name in {
        "builders",
        "configs",
        "connectors",
        "converters",
        "core",
        "data",
        "utils",
    }:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    elif name in {"SimindSimulator", "SimulationConfig"}:
        core = importlib.import_module(".core", __name__)
        obj = getattr(core, name)
        globals()[name] = obj
        return obj
    elif name in {
        "BaseConnector",
        "NativeBackendConnector",
        "NumpyConnector",
        "PyTomographySimindAdaptor",
        "ProjectionResult",
        "RuntimeOperator",
        "SirfSimindAdaptor",
        "SimindPythonConnector",
        "StirSimindAdaptor",
    }:
        connectors = importlib.import_module(".connectors", __name__)
        obj = getattr(connectors, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseConnector",
    "NativeBackendConnector",
    "NumpyConnector",
    "ProjectionResult",
    "PyTomographySimindAdaptor",
    "RuntimeOperator",
    "SirfSimindAdaptor",
    "SimindPythonConnector",
    "SimulationConfig",
    "SimindSimulator",
    "StirSimindAdaptor",
    "builders",
    "configs",
    "connectors",
    "converters",
    "core",
    "data",
    "utils",
]

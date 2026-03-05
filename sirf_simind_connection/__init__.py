"""
SIRF ⇄ SIMIND connector/adaptor API.
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
    elif name in {"SimulationConfig"}:
        core = importlib.import_module(".core", __name__)
        obj = getattr(core, name)
        globals()[name] = obj
        return obj
    elif name in {
        "BaseConnector",
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
    "NumpyConnector",
    "ProjectionResult",
    "PyTomographySimindAdaptor",
    "RuntimeOperator",
    "SirfSimindAdaptor",
    "SimindPythonConnector",
    "SimulationConfig",
    "StirSimindAdaptor",
    "builders",
    "configs",
    "connectors",
    "converters",
    "core",
    "data",
    "utils",
]

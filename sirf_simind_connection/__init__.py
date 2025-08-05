"""
SIRF ⇄ SIMIND connector – public API.

>>> from sirf_simind_connection import SimindSimulator
"""

from importlib import metadata as _meta

try:  # installed (pip/poetry)
    __version__ = _meta.version(__name__)
except _meta.PackageNotFoundError:  # editable / source checkout
    __version__ = "1.0.0"


# Lazy imports to avoid SIRF dependencies in CI
def __getattr__(name):
    if name == "SimindSimulator":
        from .core import SimindSimulator

        return SimindSimulator
    elif name == "SimindProjector":
        from .core import SimindProjector

        return SimindProjector
    elif name == "SimulationConfig":
        from .core import SimulationConfig

        return SimulationConfig
    elif name == "builders":
        from . import builders

        return builders
    elif name == "configs":
        from . import configs

        return configs
    elif name == "core":
        from . import core

        return core
    elif name == "data":
        from . import data

        return data
    elif name == "utils":
        from . import utils

        return utils
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "SimulationConfig",
    "SimindProjector",
    "SimindSimulator",
    "builders",
    "configs",
    "core",
    "data",
    "utils",
]

"""
SIRF ⇄ SIMIND connector – public API.

>>> from sirf_simind_connection import SimindSimulator
"""

from importlib import metadata as _meta

try:  # installed (pip/poetry)
    __version__ = _meta.version(__name__)
except _meta.PackageNotFoundError:  # editable / source checkout
    __version__ = "0.1.0"

from . import builders, configs, data, utils, core
# Re-export key objects so users can do:
#   from sirf_simind_connection import SimindSimulator
from .core import SimindSimulator  # noqa: F401
from .core import SimindProjector, SimulationConfig

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

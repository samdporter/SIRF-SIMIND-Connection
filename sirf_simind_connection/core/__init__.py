"""
Numerical core: configuration parsing, projector, simulator.
"""

from .components import ScoringRoutine
from .config import SimulationConfig  # noqa: F401
from .projector import SimindProjector  # noqa: F401
from .simulator import (
    SimindSimulator,
    create_penetrate_simulator,
    create_scattwin_simulator,
    create_simulator_from_template,
)

__all__ = [
    "SimulationConfig",
    "SimindProjector",
    "SimindSimulator",
    "create_simulator_from_template",
    "create_penetrate_simulator",
    "create_scattwin_simulator",
    "ScoringRoutine",
]

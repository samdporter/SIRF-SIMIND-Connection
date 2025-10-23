"""
Numerical core: configuration parsing, projector, simulator.
"""


# Lazy imports to avoid SIRF dependencies in CI
def __getattr__(name):
    if name == "ScoringRoutine":
        from .types import ScoringRoutine

        return ScoringRoutine
    elif name == "SimulationConfig":
        from .config import SimulationConfig

        return SimulationConfig
    elif name == "CoordinatedProjector":
        from .projector import CoordinatedProjector

        return CoordinatedProjector
    elif name == "CoordinatedSubsetProjector":
        from .projector import CoordinatedSubsetProjector

        return CoordinatedSubsetProjector
    elif name == "SimindProjector":
        from .projector import SimindProjector

        return SimindProjector
    elif name == "SimindSubsetProjector":
        from .projector import SimindSubsetProjector

        return SimindSubsetProjector
    elif name == "Coordinator":
        from .coordinator import Coordinator

        return Coordinator
    elif name == "SimindCoordinator":
        from .coordinator import SimindCoordinator

        return SimindCoordinator
    elif name == "StirPsfCoordinator":
        from .coordinator import StirPsfCoordinator

        return StirPsfCoordinator
    elif name == "SimindSimulator":
        from .simulator import SimindSimulator

        return SimindSimulator
    elif name == "create_penetrate_simulator":
        from .simulator import create_penetrate_simulator

        return create_penetrate_simulator
    elif name == "create_scattwin_simulator":
        from .simulator import create_scattwin_simulator

        return create_scattwin_simulator
    elif name == "create_simulator_from_template":
        from .simulator import create_simulator_from_template

        return create_simulator_from_template
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "SimulationConfig",
    "CoordinatedProjector",
    "CoordinatedSubsetProjector",
    "SimindProjector",  # Backward compatibility alias
    "SimindSubsetProjector",  # Backward compatibility alias
    "Coordinator",
    "SimindCoordinator",
    "StirPsfCoordinator",
    "SimindSimulator",
    "create_simulator_from_template",
    "create_penetrate_simulator",
    "create_scattwin_simulator",
    "ScoringRoutine",
]

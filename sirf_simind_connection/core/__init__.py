"""
Numerical core: configuration parsing, projector, simulator.
"""


# Lazy imports to avoid SIRF dependencies in CI
def __getattr__(name):
    if name == "ScoringRoutine":
        from .components import ScoringRoutine

        return ScoringRoutine
    elif name == "SimulationConfig":
        from .config import SimulationConfig

        return SimulationConfig
    elif name == "SimindProjector":
        from .projector import SimindProjector

        return SimindProjector
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
    "SimindProjector",
    "SimindSimulator",
    "create_simulator_from_template",
    "create_penetrate_simulator",
    "create_scattwin_simulator",
    "ScoringRoutine",
]

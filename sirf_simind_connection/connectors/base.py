"""
Abstract connector contracts and shared native-backend connector logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Union

from sirf_simind_connection.backends import set_backend
from sirf_simind_connection.core.config import SimulationConfig
from sirf_simind_connection.core.types import ScoringRoutine


class BaseConnector(ABC):
    """Minimal contract shared by all connector types."""

    @abstractmethod
    def add_config_value(self, index: int, value: Any) -> None:
        """Set a SIMIND config value by index."""

    @abstractmethod
    def add_runtime_switch(self, switch: str, value: Any) -> None:
        """Set one runtime switch."""

    def set_runtime_switches(self, switches: Mapping[str, Any]) -> None:
        """Set many runtime switches."""
        for switch, value in switches.items():
            self.add_runtime_switch(switch, value)

    @abstractmethod
    def run(self) -> Any:
        """Execute the simulation."""

    @abstractmethod
    def get_outputs(self) -> Dict[str, Any]:
        """Return simulation outputs in connector-native types."""

    @abstractmethod
    def get_config(self) -> SimulationConfig:
        """Return the active simulation config object."""


class NativeBackendConnector(BaseConnector):
    """Shared implementation for SIRF/STIR-backed connectors."""

    def __init__(
        self,
        config_source: Union[str, SimulationConfig],
        output_dir: str,
        backend: str,
        output_prefix: str = "output",
        photon_multiplier: int = 1,
        quantization_scale: float = 1.0,
        scoring_routine: Union[ScoringRoutine, int] = ScoringRoutine.SCATTWIN,
    ) -> None:
        set_backend(backend)
        self.backend = backend
        from sirf_simind_connection.core.simulator import SimindSimulator

        self.simulator = SimindSimulator(
            config_source=config_source,
            output_dir=output_dir,
            output_prefix=output_prefix,
            photon_multiplier=photon_multiplier,
            quantization_scale=quantization_scale,
            scoring_routine=scoring_routine,
        )

    def set_source(self, source: Any) -> None:
        self.simulator.set_source(source)

    def set_mu_map(self, mu_map: Any) -> None:
        self.simulator.set_mu_map(mu_map)

    def set_template_sinogram(self, template_sinogram: Any) -> None:
        self.simulator.set_template_sinogram(template_sinogram)

    def set_energy_windows(
        self, lower_bounds: Any, upper_bounds: Any, scatter_orders: Any
    ) -> None:
        self.simulator.set_energy_windows(lower_bounds, upper_bounds, scatter_orders)

    def add_config_value(self, index: int, value: Any) -> None:
        self.simulator.add_config_value(index, value)

    def add_runtime_switch(self, switch: str, value: Any) -> None:
        self.simulator.add_runtime_switch(switch, value)

    def run(self) -> None:
        self.simulator.run_simulation()

    def get_outputs(self) -> Dict[str, Any]:
        return self.simulator.get_outputs(native=True, preferred_backend=self.backend)

    def get_total_output(self, window: int = 1) -> Any:
        return self.simulator.get_total_output(
            window=window, native=True, preferred_backend=self.backend
        )

    def get_scatter_output(self, window: int = 1) -> Any:
        return self.simulator.get_scatter_output(
            window=window, native=True, preferred_backend=self.backend
        )

    def get_primary_output(self, window: int = 1) -> Any:
        return self.simulator.get_primary_output(
            window=window, native=True, preferred_backend=self.backend
        )

    def get_penetrate_output(self, component: Any) -> Any:
        return self.simulator.get_penetrate_output(
            component=component, native=True, preferred_backend=self.backend
        )

    def list_available_outputs(self) -> list[str]:
        return self.simulator.list_available_outputs()

    def get_scoring_routine(self) -> ScoringRoutine:
        return self.simulator.get_scoring_routine()

    def get_config(self) -> SimulationConfig:
        return self.simulator.get_config()


__all__ = ["BaseConnector", "NativeBackendConnector"]

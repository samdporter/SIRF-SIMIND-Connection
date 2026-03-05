"""Abstract connector contract shared by all connector/adaptor implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping

from sirf_simind_connection.core.config import SimulationConfig


class BaseConnector(ABC):
    """Minimal interface for all SIMIND connector styles."""

    @abstractmethod
    def add_config_value(self, index: int, value: Any) -> None:
        """Set a SIMIND config value by index."""

    @abstractmethod
    def add_runtime_switch(self, switch: str, value: Any) -> None:
        """Set one runtime switch."""

    def set_runtime_switches(self, switches: Mapping[str, Any]) -> None:
        """Set multiple runtime switches."""
        for switch, value in switches.items():
            self.add_runtime_switch(switch, value)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute simulation and return connector-native outputs."""

    @abstractmethod
    def get_outputs(self) -> Dict[str, Any]:
        """Return outputs from the latest successful run."""

    @abstractmethod
    def get_config(self) -> SimulationConfig:
        """Return active simulation configuration."""


__all__ = ["BaseConnector"]

"""
SIRF/SIMIND adaptor facade.
"""

from __future__ import annotations

from typing import Union

from sirf_simind_connection.connectors.base import NativeBackendConnector
from sirf_simind_connection.core.config import SimulationConfig
from sirf_simind_connection.core.types import ScoringRoutine


class SirfSimindAdaptor(NativeBackendConnector):
    """Adaptor that consumes and returns SIRF-native data objects."""

    def __init__(
        self,
        config_source: Union[str, SimulationConfig],
        output_dir: str,
        output_prefix: str = "output",
        photon_multiplier: int = 1,
        quantization_scale: float = 1.0,
        scoring_routine: Union[ScoringRoutine, int] = ScoringRoutine.SCATTWIN,
    ) -> None:
        super().__init__(
            config_source=config_source,
            output_dir=output_dir,
            backend="sirf",
            output_prefix=output_prefix,
            photon_multiplier=photon_multiplier,
            quantization_scale=quantization_scale,
            scoring_routine=scoring_routine,
        )


__all__ = ["SirfSimindAdaptor"]

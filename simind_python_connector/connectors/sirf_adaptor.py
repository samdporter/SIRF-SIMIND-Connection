"""SIRF adaptor implemented on top of the connector-first NumPy pipeline."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from simind_python_connector.connectors._spacing import extract_voxel_size_mm
from simind_python_connector.connectors.base import BaseConnector
from simind_python_connector.connectors.python_connector import (
    ConfigSource,
    RuntimeOperator,
    SimindPythonConnector,
)
from simind_python_connector.core.config import SimulationConfig
from simind_python_connector.core.types import PenetrateOutputType, ScoringRoutine
from simind_python_connector.utils import get_array


try:
    import sirf.STIR as sirf
except ImportError:  # pragma: no cover - optional dependency
    sirf = None  # type: ignore[assignment]


class SirfSimindAdaptor(BaseConnector):
    """Adaptor consuming/returning SIRF-native objects."""

    def __init__(
        self,
        config_source: ConfigSource,
        output_dir: str,
        output_prefix: str = "output",
        photon_multiplier: int = 1,
        quantization_scale: float = 1.0,
        scoring_routine: ScoringRoutine | int = ScoringRoutine.SCATTWIN,
    ) -> None:
        if sirf is None:
            raise ImportError("SirfSimindAdaptor requires the SIRF Python package.")

        self.python_connector = SimindPythonConnector(
            config_source=config_source,
            output_dir=output_dir,
            output_prefix=output_prefix,
            quantization_scale=quantization_scale,
        )
        self._scoring_routine = (
            ScoringRoutine(scoring_routine)
            if isinstance(scoring_routine, int)
            else scoring_routine
        )
        self._source: Any = None
        self._mu_map: Any = None
        self._outputs: Optional[dict[str, Any]] = None

        self.add_runtime_switch("NN", photon_multiplier)

    def set_source(self, source: Any) -> None:
        self._source = source

    def set_mu_map(self, mu_map: Any) -> None:
        self._mu_map = mu_map

    def set_energy_windows(
        self,
        lower_bounds: float | list[float],
        upper_bounds: float | list[float],
        scatter_orders: int | list[int],
    ) -> None:
        self.python_connector.set_energy_windows(
            lower_bounds, upper_bounds, scatter_orders
        )

    def add_config_value(self, index: int, value: Any) -> None:
        self.python_connector.add_config_value(index, value)

    def add_runtime_switch(self, switch: str, value: Any) -> None:
        self.python_connector.add_runtime_switch(switch, value)

    def run(self, runtime_operator: Optional[RuntimeOperator] = None) -> dict[str, Any]:
        self._validate_inputs()
        assert self._source is not None
        assert self._mu_map is not None

        source_arr = np.asarray(get_array(self._source), dtype=np.float32)
        mu_arr = np.asarray(get_array(self._mu_map), dtype=np.float32)
        voxel_size_mm = self._extract_voxel_size_mm(self._source)

        self.python_connector.configure_voxel_phantom(
            source=source_arr,
            mu_map=mu_arr,
            voxel_size_mm=voxel_size_mm,
            scoring_routine=self._scoring_routine,
        )
        raw_outputs = self.python_connector.run(runtime_operator=runtime_operator)
        self._outputs = {
            key: sirf.AcquisitionData(str(result.header_path))
            for key, result in raw_outputs.items()
        }
        return self._outputs

    def get_outputs(self) -> Dict[str, Any]:
        if self._outputs is None:
            raise RuntimeError("Run the adaptor first to produce outputs.")
        return self._outputs

    def get_total_output(self, window: int = 1) -> Any:
        return self._get_component("tot", window)

    def get_scatter_output(self, window: int = 1) -> Any:
        return self._get_component("sca", window)

    def get_primary_output(self, window: int = 1) -> Any:
        return self._get_component("pri", window)

    def get_air_output(self, window: int = 1) -> Any:
        return self._get_component("air", window)

    def get_penetrate_output(self, component: PenetrateOutputType | str) -> Any:
        outputs = self.get_outputs()
        key = (
            component.slug if isinstance(component, PenetrateOutputType) else component
        )
        if key not in outputs:
            available = ", ".join(sorted(outputs))
            raise KeyError(f"Output {key!r} not available. Available: {available}")
        return outputs[key]

    def list_available_outputs(self) -> list[str]:
        return sorted(self.get_outputs().keys())

    def get_scoring_routine(self) -> ScoringRoutine:
        return self._scoring_routine

    def get_config(self) -> SimulationConfig:
        return self.python_connector.get_config()

    def _get_component(self, prefix: str, window: int) -> Any:
        outputs = self.get_outputs()
        key = f"{prefix}_w{window}"
        if key not in outputs:
            available = ", ".join(sorted(outputs))
            raise KeyError(f"Output {key!r} not available. Available: {available}")
        return outputs[key]

    @staticmethod
    def _extract_voxel_size_mm(image: Any) -> float:
        return extract_voxel_size_mm(image=image, backend_name="SIRF")

    def _validate_inputs(self) -> None:
        if self._source is None or self._mu_map is None:
            raise ValueError("Both source and mu_map must be set before run().")

        source_shape = np.asarray(get_array(self._source)).shape
        mu_shape = np.asarray(get_array(self._mu_map)).shape
        if source_shape != mu_shape:
            raise ValueError(
                f"source and mu_map must have matching shapes, got "
                f"{source_shape} and {mu_shape}"
            )


__all__ = ["SirfSimindAdaptor"]

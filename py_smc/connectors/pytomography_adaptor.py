"""
PyTomography/SIMIND adaptor.

This adaptor does not depend on SIRF/STIR objects for inputs or outputs.
It accepts torch tensors, configures SIMIND directly, executes the simulation,
and returns torch tensors for projection outputs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np

from py_smc.connectors.base import BaseConnector
from py_smc.connectors.python_connector import (
    ConfigSource,
    RuntimeOperator,
    SimindPythonConnector,
)
from py_smc.core.types import (
    ScoringRoutine,
    ValidationError,
)


try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from pytomography.io.SPECT import simind as pytomo_simind
except ImportError:  # pragma: no cover - optional dependency
    pytomo_simind = None  # type: ignore[assignment]


PathLike = Union[str, os.PathLike[str]]


class PyTomographySimindAdaptor(BaseConnector):
    """SIMIND adaptor that consumes and returns PyTomography-native tensors.

    Public tensor convention for this connector is object space ``(x, y, z)``.
    Internal SIMIND input files are written in SIMIND image order ``(z, y, x)``.
    """

    def __init__(
        self,
        config_source: ConfigSource,
        output_dir: PathLike,
        output_prefix: str = "output",
        photon_multiplier: int = 1,
        voxel_size_mm: float = 4.0,
        quantization_scale: float = 1.0,
        scoring_routine: Union[ScoringRoutine, int] = ScoringRoutine.SCATTWIN,
    ) -> None:
        if torch is None:
            raise ImportError(
                "PyTomographySimindAdaptor requires torch to be installed "
                "(and typically pytomography in your environment)."
            )

        self.python_connector = SimindPythonConnector(
            config_source=config_source,
            output_dir=output_dir,
            output_prefix=output_prefix,
            quantization_scale=quantization_scale,
        )
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_prefix = output_prefix
        self.voxel_size_mm = float(voxel_size_mm)
        if self.voxel_size_mm <= 0:
            raise ValueError("voxel_size_mm must be > 0")

        self._scoring_routine = (
            ScoringRoutine(scoring_routine)
            if isinstance(scoring_routine, int)
            else scoring_routine
        )

        self._source: Optional[torch.Tensor] = None
        self._mu_map: Optional[torch.Tensor] = None
        self._energy_windows: Optional[tuple[list[float], list[float], list[int]]] = (
            None
        )
        self._outputs: Optional[dict[str, torch.Tensor]] = None
        self._output_metadata: Optional[dict[str, Mapping[str, str]]] = None
        self._output_header_paths: Optional[dict[str, Path]] = None

        self.add_runtime_switch("NN", photon_multiplier)

    def set_source(self, source: torch.Tensor) -> None:
        self._source = self._validate_tensor(source, name="source")

    def set_mu_map(self, mu_map: torch.Tensor) -> None:
        self._mu_map = self._validate_tensor(mu_map, name="mu_map")

    def set_energy_windows(
        self,
        lower_bounds: Union[float, list[float]],
        upper_bounds: Union[float, list[float]],
        scatter_orders: Union[int, list[int]],
    ) -> None:
        lowers = (
            [float(lower_bounds)]
            if isinstance(lower_bounds, (int, float))
            else [float(v) for v in lower_bounds]
        )
        uppers = (
            [float(upper_bounds)]
            if isinstance(upper_bounds, (int, float))
            else [float(v) for v in upper_bounds]
        )
        orders = (
            [int(scatter_orders)]
            if isinstance(scatter_orders, (int, float))
            else [int(v) for v in scatter_orders]
        )
        if not (len(lowers) == len(uppers) == len(orders)):
            raise ValueError(
                "lower_bounds, upper_bounds, and scatter_orders must have equal lengths"
            )
        self._energy_windows = (lowers, uppers, orders)

    def add_config_value(self, index: int, value: Any) -> None:
        self.python_connector.add_config_value(index, value)

    def add_runtime_switch(self, switch: str, value: Any) -> None:
        self.python_connector.add_runtime_switch(switch, value)

    def run(
        self, runtime_operator: Optional[RuntimeOperator] = None
    ) -> Dict[str, torch.Tensor]:
        self._validate_inputs()
        assert self._source is not None  # for type checkers
        assert self._mu_map is not None
        assert self._energy_windows is not None

        source_zyx = self.to_simind_image_axes(self._source).numpy()
        mu_map_zyx = self.to_simind_image_axes(self._mu_map).numpy()
        self.python_connector.configure_voxel_phantom(
            source=source_zyx,
            mu_map=mu_map_zyx,
            voxel_size_mm=self.voxel_size_mm,
            scoring_routine=self._scoring_routine,
        )
        self.python_connector.set_energy_windows(*self._energy_windows)

        raw_outputs = self.python_connector.run(runtime_operator=runtime_operator)
        outputs: dict[str, torch.Tensor] = {}
        metadata: dict[str, Mapping[str, str]] = {}
        header_paths: dict[str, Path] = {}

        for key, value in raw_outputs.items():
            header_path = self.output_dir / f"{self.output_prefix}_{key}.h00"
            if pytomo_simind is not None and header_path.exists():
                projection = pytomo_simind.get_projections(str(header_path)).to(
                    dtype=torch.float32
                )
            else:
                projection = torch.from_numpy(
                    value.projection.astype(np.float32, copy=False)
                )
                # Fall back to converter-generated .hs header for metadata access.
                header_path = value.header_path

            outputs[key] = projection
            metadata[key] = value.metadata
            header_paths[key] = Path(header_path).resolve()

        self._outputs = outputs
        self._output_metadata = metadata
        self._output_header_paths = header_paths
        return outputs

    def get_outputs(self) -> Dict[str, torch.Tensor]:
        if self._outputs is None:
            raise RuntimeError("Run the connector first to produce outputs")
        return self._outputs

    def get_output_metadata(self, key: str) -> Mapping[str, str]:
        if self._output_metadata is None:
            raise RuntimeError("Run the connector first to produce outputs")
        if key not in self._output_metadata:
            available = ", ".join(sorted(self._output_metadata))
            raise KeyError(f"Unknown output key {key!r}. Available: {available}")
        return self._output_metadata[key]

    def get_output_header_path(self, key: str) -> Path:
        if self._output_header_paths is None:
            raise RuntimeError("Run the connector first to produce outputs")
        if key not in self._output_header_paths:
            available = ", ".join(sorted(self._output_header_paths))
            raise KeyError(f"Unknown output key {key!r}. Available: {available}")
        return self._output_header_paths[key]

    def get_total_output(self, window: int = 1) -> torch.Tensor:
        return self._get_component("tot", window)

    def get_scatter_output(self, window: int = 1) -> torch.Tensor:
        return self._get_component("sca", window)

    def get_primary_output(self, window: int = 1) -> torch.Tensor:
        return self._get_component("pri", window)

    def get_air_output(self, window: int = 1) -> torch.Tensor:
        return self._get_component("air", window)

    def get_config(self):
        return self.python_connector.get_config()

    def _get_component(self, prefix: str, window: int) -> torch.Tensor:
        outputs = self.get_outputs()
        key = f"{prefix}_w{window}"
        if key not in outputs:
            available = ", ".join(sorted(outputs))
            raise KeyError(f"Output {key!r} not available. Available: {available}")
        return outputs[key]

    def _validate_inputs(self) -> None:
        if self._source is None or self._mu_map is None:
            raise ValidationError("Both source and mu_map tensors must be set")
        if self._source.shape != self._mu_map.shape:
            raise ValidationError("source and mu_map must have identical tensor shapes")
        if self._energy_windows is None:
            raise ValidationError("At least one energy window must be configured")

    @staticmethod
    def _validate_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 3:
            raise ValueError(
                f"{name} must be a 3D tensor with shape (x, y, z); "
                f"got {tuple(value.shape)}"
            )
        return value.detach().cpu().to(dtype=torch.float32).contiguous()

    @staticmethod
    def from_simind_image_axes(value: torch.Tensor) -> torch.Tensor:
        """Convert SIMIND image order ``(z, y, x)`` to PyTomography ``(x, y, z)``."""
        if value.ndim != 3:
            raise ValueError(
                "Expected 3D tensor for axis conversion, "
                f"got shape {tuple(value.shape)}"
            )
        return value.permute(2, 1, 0).contiguous().to(dtype=torch.float32)

    @staticmethod
    def to_simind_image_axes(value: torch.Tensor) -> torch.Tensor:
        """Convert PyTomography object order ``(x, y, z)`` to SIMIND ``(z, y, x)``."""
        if value.ndim != 3:
            raise ValueError(
                "Expected 3D tensor for axis conversion, "
                f"got shape {tuple(value.shape)}"
            )
        return value.permute(2, 1, 0).contiguous().to(dtype=torch.float32)


__all__ = ["PyTomographySimindAdaptor", "RuntimeOperator"]

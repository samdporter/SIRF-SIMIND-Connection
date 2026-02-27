"""
Backend-agnostic SIMIND connector returning NumPy projection outputs.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from sirf_simind_connection.connectors.base import BaseConnector
from sirf_simind_connection.converters.simind_to_stir import SimindToStirConverter
from sirf_simind_connection.core.components import SimindExecutor
from sirf_simind_connection.core.config import RuntimeSwitches, SimulationConfig
from sirf_simind_connection.core.types import PenetrateOutputType, ScoringRoutine
from sirf_simind_connection.utils.interfile_numpy import load_interfile_array


ConfigSource = Union[str, os.PathLike[str], SimulationConfig]
PathLike = Union[str, os.PathLike[str]]


@dataclass(frozen=True)
class ProjectionResult:
    """Projection array together with the header and binary file references."""

    projection: np.ndarray
    header_path: Path
    data_path: Path
    metadata: dict[str, str]


@dataclass
class RuntimeOperator:
    """Runtime modifiers applied when invoking SIMIND."""

    switches: Dict[str, Any] = field(default_factory=dict)
    orbit_file: Optional[PathLike] = None


class SimindPythonConnector(BaseConnector):
    """Pure Python connector for SIMIND with NumPy-first outputs."""

    def __init__(
        self,
        config_source: ConfigSource,
        output_dir: PathLike,
        output_prefix: str = "output",
        quantization_scale: float = 1.0,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_prefix = output_prefix
        self.quantization_scale = float(quantization_scale)
        if self.quantization_scale <= 0:
            raise ValueError("quantization_scale must be > 0")

        self.config = self._initialize_config(config_source)
        self.runtime_switches = RuntimeSwitches()
        self.executor = SimindExecutor()
        self.converter = SimindToStirConverter()

        self._outputs: Optional[dict[str, ProjectionResult]] = None

    @staticmethod
    def _initialize_config(config_source: ConfigSource) -> SimulationConfig:
        if isinstance(config_source, SimulationConfig):
            return config_source

        config_path = Path(config_source).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_source}")

        suffix = config_path.suffix.lower()
        if suffix not in {".smc", ".yaml", ".yml"}:
            raise ValueError(
                f"Unsupported configuration file extension {suffix!r}. "
                "Expected one of .smc, .yaml, .yml"
            )

        return SimulationConfig(str(config_path))

    def add_runtime_switch(self, switch: str, value: Any) -> None:
        """Set a single runtime switch."""
        self.runtime_switches.set_switch(switch, value)

    def add_config_value(self, index: int, value: Any) -> None:
        """Set a SIMIND config value."""
        self.config.set_value(index, value)

    def run(
        self, runtime_operator: Optional[RuntimeOperator] = None
    ) -> dict[str, ProjectionResult]:
        """Run SIMIND and return projection outputs as NumPy arrays."""
        self._outputs = None

        orbit_file = None
        if runtime_operator is not None:
            self.set_runtime_switches(runtime_operator.switches)
            orbit_file = self._prepare_orbit_file(runtime_operator.orbit_file)

        config_path = self.output_dir / self.output_prefix
        self.config.save_file(config_path)

        original_cwd = Path.cwd()
        try:
            os.chdir(self.output_dir)
            self.executor.run_simulation(
                self.output_prefix, orbit_file, self.runtime_switches.switches
            )
        finally:
            os.chdir(original_cwd)

        header_files = self._ensure_interfile_headers()
        self._outputs = self._load_projection_outputs(header_files)
        return self._outputs

    def get_outputs(self) -> dict[str, ProjectionResult]:
        """Return cached outputs from the last completed run."""
        if self._outputs is None:
            raise RuntimeError("No outputs are available. Run the connector first.")
        return self._outputs

    def get_config(self) -> SimulationConfig:
        return self.config

    def _prepare_orbit_file(self, orbit_file: Optional[PathLike]) -> Optional[Path]:
        if orbit_file is None:
            return None

        orbit_path = Path(orbit_file).expanduser().resolve()
        if not orbit_path.exists():
            raise FileNotFoundError(f"Orbit file not found: {orbit_path}")

        if orbit_path.parent == self.output_dir:
            return orbit_path

        copied_path = self.output_dir / orbit_path.name
        shutil.copy2(orbit_path, copied_path)
        return copied_path

    def _ensure_interfile_headers(self) -> list[Path]:
        if self._is_penetrate_routine():
            h00_file = self.converter.find_penetrate_h00_file(
                self.output_prefix, str(self.output_dir)
            )
            if h00_file is None:
                raise FileNotFoundError(
                    f"No PENETRATE .h00 file found for prefix {self.output_prefix!r} "
                    f"in {self.output_dir}"
                )
            self.converter.create_penetrate_headers_from_template(
                h00_file, self.output_prefix, str(self.output_dir)
            )
            hs_files = sorted(
                self.output_dir.glob(f"{self.output_prefix}_component_*.hs")
            )
        else:
            h00_files = sorted(self.output_dir.glob(f"*{self.output_prefix}*.h00"))
            for h00_file in h00_files:
                hs_file = h00_file.with_suffix(".hs")
                self.converter.convert_file(str(h00_file), str(hs_file))

            hs_files = sorted(self.output_dir.glob(f"*{self.output_prefix}*.hs"))
        if not hs_files:
            raise FileNotFoundError(
                f"No projection headers (.hs) found for prefix {self.output_prefix!r} "
                f"in {self.output_dir}"
            )

        return hs_files

    def _load_projection_outputs(
        self, header_files: list[Path]
    ) -> dict[str, ProjectionResult]:
        outputs: dict[str, ProjectionResult] = {}

        for header_path in header_files:
            try:
                interfile = load_interfile_array(header_path)
            except Exception as exc:
                self.logger.warning(
                    "Skipping output %s due to parse/load error: %s",
                    header_path,
                    exc,
                )
                continue

            key = self._extract_output_key(header_path)
            outputs[key] = ProjectionResult(
                projection=interfile.array,
                header_path=interfile.header_path,
                data_path=interfile.data_path,
                metadata=interfile.metadata,
            )

        if not outputs:
            raise RuntimeError(
                f"No valid outputs were parsed from headers in {self.output_dir}"
            )

        return outputs

    def _extract_output_key(self, header_path: Path) -> str:
        stem = header_path.stem

        component_prefix = f"{self.output_prefix}_component_"
        if stem.startswith(component_prefix):
            suffix = stem[len(component_prefix) :]
            if suffix.isdigit():
                component_id = int(suffix)
                with contextlib.suppress(ValueError):
                    return PenetrateOutputType(component_id).slug
                return f"b{component_id:02d}"

        if stem.startswith(self.output_prefix):
            stem = stem[len(self.output_prefix) :]
        return stem.lstrip("_") or header_path.stem

    def _is_penetrate_routine(self) -> bool:
        try:
            scoring_routine = int(
                round(float(self.config.get_value("scoring_routine")))
            )
        except Exception:
            return False
        return scoring_routine == ScoringRoutine.PENETRATE.value


NumpyConnector = SimindPythonConnector


__all__ = [
    "NumpyConnector",
    "ProjectionResult",
    "RuntimeOperator",
    "SimindPythonConnector",
]

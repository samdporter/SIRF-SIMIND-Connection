"""Minimal SIMIND process runner used by connector-first APIs."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional

from .types import SimulationError


class SimindExecutor:
    """Run SIMIND as a subprocess with optional MPI switch support."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def run_simulation(
        self,
        output_prefix: str,
        orbit_file: Optional[Path] = None,
        runtime_switches: Optional[Dict] = None,
    ) -> None:
        mp_value = None
        if runtime_switches:
            for key, value in runtime_switches.items():
                if str(key).lower() == "mp":
                    mp_value = value
                    break

        if mp_value is not None:
            command = [
                "mpirun",
                "-np",
                str(mp_value),
                "simind",
                output_prefix,
                output_prefix,
            ]
        else:
            command = ["simind", output_prefix, output_prefix]

        if orbit_file:
            command.append(orbit_file.name)

        if mp_value is not None:
            command.append("-p")

        if runtime_switches:
            switch_parts = []
            for key, value in runtime_switches.items():
                if str(key).upper() == "MP":
                    switch_parts.append(f"/{key}")
                else:
                    switch_parts.append(f"/{key}:{value}")
            if switch_parts:
                command.append("".join(switch_parts))

        self.logger.info("Running SIMIND: %s", " ".join(command))
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise SimulationError(f"SIMIND execution failed: {exc}") from exc


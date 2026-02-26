"""
File management utilities for SIMIND simulation.

This module handles preparation of input files (source, attenuation) and
management of orbit files for non-circular acquisitions.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from sirf_simind_connection.utils import get_array

from .types import MAX_SOURCE, ORBIT_FILE_EXTENSION, SIMIND_VOXEL_UNIT_CONVERSION


class DataFileManager:
    """Manages input data file preparation."""

    def __init__(self, output_dir: Path, quantization_scale: float = 1.0):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.temp_files: List[Path] = []
        self.quantization_scale = float(quantization_scale)
        if self.quantization_scale <= 0:
            raise ValueError("quantization_scale must be > 0")

    def prepare_source_file(self, source, output_prefix: str) -> str:
        """Prepare source data file for SIMIND.

        Args:
            source: Source image object (backend-agnostic)
            output_prefix: Prefix for output filename

        Returns:
            str: Output file prefix with suffix
        """
        source_arr = get_array(source)

        # Normalize to uint16 range
        source_max = float(np.max(source_arr))
        if source_max > 0:
            source_arr = (
                source_arr / source_max * (MAX_SOURCE * self.quantization_scale)
            )

        source_arr = np.clip(np.round(source_arr), 0, MAX_SOURCE).astype(np.uint16)

        filename = f"{output_prefix}_src.smi"
        filepath = self.output_dir / filename
        source_arr.tofile(filepath)

        self.temp_files.append(filepath)
        return output_prefix + "_src"

    def prepare_attenuation_file(
        self,
        mu_map,
        output_prefix: str,
        use_attenuation: bool,
        photon_energy: float,
        input_dir: Path,
    ) -> str:
        """Prepare attenuation data file for SIMIND.

        Args:
            mu_map: Attenuation map image object (backend-agnostic)
            output_prefix: Prefix for output filename
            use_attenuation: Whether to use attenuation correction
            photon_energy: Photon energy in keV
            input_dir: Directory for input files

        Returns:
            str: Output file prefix with suffix
        """
        if use_attenuation:
            from sirf_simind_connection.converters.attenuation import (
                attenuation_to_density,
            )

            mu_map_arr = get_array(mu_map)
            mu_map_arr = (
                attenuation_to_density(mu_map_arr, photon_energy, input_dir) * 1000
            )
        else:
            mu_map_arr = np.zeros(get_array(mu_map).shape)

        mu_map_arr = mu_map_arr.astype(np.uint16)

        filename = f"{output_prefix}_dns.dmi"
        filepath = self.output_dir / filename
        mu_map_arr.tofile(filepath)

        self.temp_files.append(filepath)
        return output_prefix + "_dns"

    def cleanup_temp_files(self) -> None:
        """Remove temporary files."""
        for filepath in self.temp_files:
            if filepath.exists():
                filepath.unlink()
                self.logger.debug(f"Removed temp file: {filepath}")
        self.temp_files.clear()


class OrbitFileManager:
    """Manages orbit files for non-circular orbits."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

    def write_orbit_file(
        self,
        radii: List[float],
        output_prefix: str,
        center_of_rotation: Optional[float] = None,
    ) -> Path:
        """
        Write orbit file for non-circular orbits.

        Args:
            radii: List of radii in mm (STIR units)
            output_prefix: Prefix for output filename
            center_of_rotation: Center of rotation in pixels

        Returns:
            Path to the created orbit file

        Note:
            - Input radii are in mm (STIR), output file uses cm (SIMIND)
            - Uses suffix "_input.cor" to avoid conflict with SIMIND's output .cor file
        """
        if center_of_rotation is None:
            center_of_rotation = 64  # Default center

        # Use "_input.cor" suffix to avoid SIMIND overwriting it with output .cor
        orbit_file = self.output_dir / f"{output_prefix}_input{ORBIT_FILE_EXTENSION}"

        with open(orbit_file, "w") as f:
            for radius_mm in radii:
                # Convert from mm (STIR) to cm (SIMIND)
                radius_cm = radius_mm / SIMIND_VOXEL_UNIT_CONVERSION
                f.write(f"{radius_cm:.6f}\t{center_of_rotation}\t\n")

        self.logger.info(
            f"Orbit file written: {orbit_file} "
            f"({len(radii)} radii, mm->cm conversion applied)"
        )
        return orbit_file

    def read_orbit_file(self, orbit_file: Path) -> List[float]:
        """Read orbit file and return radii in mm."""
        radii = []
        with open(orbit_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    radius_cm = float(parts[0])
                    radii.append(
                        radius_cm * SIMIND_VOXEL_UNIT_CONVERSION
                    )  # Convert to mm
        return radii


__all__ = ["DataFileManager", "OrbitFileManager"]

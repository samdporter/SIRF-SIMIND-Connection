"""
Refactored SimindSimulator components with better separation of concerns and
penetrate support.
Each component has a single responsibility, making the code easier to maintain and test.
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from sirf_simind_connection.utils import get_array

# Import types that don't depend on SIRF
from .types import (
    MAX_SOURCE,
    ORBIT_FILE_EXTENSION,
    SIMIND_VOXEL_UNIT_CONVERSION,
    OutputError,
    PenetrateOutputType,
    RotationDirection,
    ScoringRoutine,
    SimulationError,
    ValidationError,
)


# Conditional import for SIRF to avoid CI dependencies
from sirf_simind_connection.utils.import_helpers import get_sirf_types

ImageData, AcquisitionData, SIRF_AVAILABLE = get_sirf_types()

# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ImageGeometry:
    """Represents 3D image geometry."""

    dim_x: int
    dim_y: int
    dim_z: int
    voxel_x: float  # mm
    voxel_y: float  # mm
    voxel_z: float  # mm

    @classmethod
    def from_image(cls, image) -> "ImageGeometry":
        """Extract geometry from a backend-agnostic image object.

        Args:
            image: Image object with dimensions() and voxel_sizes() methods
                   (supports both SIRF and STIR backends)

        Returns:
            ImageGeometry: Extracted geometry information
        """
        # Use duck typing - works with both SIRF and STIR backend wrappers
        if not (hasattr(image, 'dimensions') and hasattr(image, 'voxel_sizes')):
            raise TypeError(
                f"Image object must have dimensions() and voxel_sizes() methods. "
                f"Got {type(image)}"
            )

        dims = image.dimensions()
        voxels = image.voxel_sizes()
        return cls(
            dim_x=dims[2],
            dim_y=dims[1],
            dim_z=dims[0],
            voxel_x=voxels[2],
            voxel_y=voxels[1],
            voxel_z=voxels[0],
        )

    def validate_square_pixels(self) -> None:
        if abs(self.voxel_x - self.voxel_y) > 1e-6:
            raise ValidationError("Image must have square pixels")
        if self.dim_x != self.dim_y:
            raise ValidationError("Image must have same x,y dimensions")


@dataclass
class EnergyWindow:
    """Represents an energy window configuration."""

    lower_bound: float
    upper_bound: float
    scatter_order: int
    window_id: int = 1


@dataclass
class RotationParameters:
    """Represents rotation parameters for acquisition."""

    direction: RotationDirection
    rotation_angle: float  # degrees
    start_angle: float  # degrees
    num_projections: int

    def round_rotation_angle(self) -> None:
        """Round rotation angle to nearest 180 or 360 degrees."""
        if self.rotation_angle % 360 < 1e-2:
            self.rotation_angle = 360
        elif self.rotation_angle % 180 < 1e-2:
            self.rotation_angle = 180
        else:
            raise ValidationError(
                "Rotation angle must be a multiple of 180 or 360 degrees"
            )

    def to_simind_params(self) -> Tuple[int, float]:
        """Convert to SIMIND rotation switch and start angle."""
        # Map direction and angle to SIMIND rotation switches
        switch_map = {
            (RotationDirection.CCW, 360): 0,
            (RotationDirection.CCW, 180): 1,
            (RotationDirection.CW, 360): 2,
            (RotationDirection.CW, 180): 3,
        }

        self.round_rotation_angle()

        key = (self.direction, self.rotation_angle)
        if key not in switch_map:
            raise ValidationError(
                f"Unsupported rotation: {self.direction.value} {self.rotation_angle}°"
            )

        # Convert start angle (STIR to SIMIND coordinate system)
        simind_start = (self.start_angle + 180) % 360

        return switch_map[key], simind_start


# =============================================================================
# VALIDATORS
# =============================================================================


class ImageValidator:
    """Validates image inputs for SIMIND simulation."""

    @staticmethod
    def validate_compatibility(image1, image2) -> None:
        """Check that two images have compatible geometry.

        Args:
            image1: First image object (backend-agnostic)
            image2: Second image object (backend-agnostic)
        """
        geom1 = ImageGeometry.from_image(image1)
        geom2 = ImageGeometry.from_image(image2)

        if (geom1.voxel_x, geom1.voxel_y, geom1.voxel_z) != (
            geom2.voxel_x,
            geom2.voxel_y,
            geom2.voxel_z,
        ):
            raise ValidationError("Images must have same voxel sizes")

        if (geom1.dim_x, geom1.dim_y, geom1.dim_z) != (
            geom2.dim_x,
            geom2.dim_y,
            geom2.dim_z,
        ):
            raise ValidationError("Images must have same dimensions")

    @staticmethod
    def validate_square_pixels(image) -> None:
        """Check that image has square pixels.

        Args:
            image: Image object (backend-agnostic)
        """
        geom = ImageGeometry.from_image(image)
        geom.validate_square_pixels()


# =============================================================================
# CONFIGURATION MANAGERS
# =============================================================================


class GeometryManager:
    """Manages geometric configuration for SIMIND."""

    def __init__(self, config_writer):
        self.config = config_writer
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _geometry_scalars(
        geometry: ImageGeometry,
    ) -> tuple[float, float, float, float, float]:
        vox_xy_cm = geometry.voxel_x / SIMIND_VOXEL_UNIT_CONVERSION
        vox_z_cm = geometry.voxel_z / SIMIND_VOXEL_UNIT_CONVERSION
        half_z = geometry.dim_z * vox_z_cm / 2
        half_x = geometry.dim_x * vox_xy_cm / 2
        half_y = geometry.dim_y * vox_xy_cm / 2
        return vox_xy_cm, vox_z_cm, half_z, half_x, half_y

    def configure_source_geometry(self, geometry: ImageGeometry) -> None:
        """Configure source image geometry parameters."""
        vox_xy_cm, _, half_z, half_x, half_y = self._geometry_scalars(geometry)

        self.config.set_value(2, half_z)
        self.config.set_value(3, half_x)
        self.config.set_value(4, half_y)
        self.config.set_value(28, vox_xy_cm)
        self.config.set_value(76, geometry.dim_x)
        self.config.set_value(77, geometry.dim_y)

        self.logger.info(
            f"Source geometry: {geometry.dim_x}×{geometry.dim_y}×{geometry.dim_z}"
        )

    def configure_attenuation_geometry(self, geometry: ImageGeometry) -> None:
        """Configure attenuation map geometry parameters."""
        vox_xy_cm, vox_z_cm, half_z, half_x, half_y = self._geometry_scalars(geometry)

        self.config.set_value(5, half_z)
        self.config.set_value(6, half_x)
        self.config.set_value(7, half_y)
        self.config.set_value(31, vox_xy_cm)
        self.config.set_value(33, 1)
        self.config.set_value(34, geometry.dim_z)
        self.config.set_value(78, geometry.dim_x)
        self.config.set_value(79, geometry.dim_y)


class AcquisitionManager:
    """Manages acquisition parameters for SIMIND."""

    def __init__(self, config_writer, runtime_switches):
        self.config = config_writer
        self.runtime_switches = runtime_switches
        self.logger = logging.getLogger(__name__)

    def configure_rotation(
        self, rotation: RotationParameters, detector_distance: float
    ) -> None:
        """Configure rotation parameters."""
        rotation_switch, start_angle = rotation.to_simind_params()

        self.config.set_value(29, rotation.num_projections)
        self.config.set_value(
            12, detector_distance / SIMIND_VOXEL_UNIT_CONVERSION
        )  # convert to cm
        self.config.set_value(30, rotation_switch)
        self.config.set_value(41, start_angle)

        self.logger.info(
            f"Rotation: {rotation.direction.value} {rotation.rotation_angle}° "
            f"from {rotation.start_angle}°"
        )

    def configure_energy_windows(
        self, windows: List[EnergyWindow], output_prefix: str
    ) -> None:
        """Configure energy windows."""
        from sirf_simind_connection.utils.simind_utils import create_window_file

        lower_bounds = [w.lower_bound for w in windows]
        upper_bounds = [w.upper_bound for w in windows]
        scatter_orders = [w.scatter_order for w in windows]

        create_window_file(lower_bounds, upper_bounds, scatter_orders, output_prefix)
        self.logger.info(f"Configured {len(windows)} energy windows")


# =============================================================================
# FILE MANAGERS
# =============================================================================


# File managers have been moved to file_managers.py
# Import them here for backward compatibility
from .file_managers import DataFileManager, OrbitFileManager


# =============================================================================
# EXECUTION ENGINE
# =============================================================================


class SimindExecutor:
    """Handles SIMIND subprocess execution."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def run_simulation(
        self,
        output_prefix: str,
        orbit_file: Optional[Path] = None,
        runtime_switches: Optional[Dict] = None,
    ) -> None:
        """Execute SIMIND simulation."""
        # Check for MPI parallel run
        mp_value = None
        if runtime_switches:
            for k, v in runtime_switches.items():
                if k.lower() == "mp":
                    mp_value = v
                    break

        if mp_value is not None:
            # MPI parallel run
            # MP value is the number of cores to use
            command = [
                "mpirun",
                "-np",
                str(mp_value),
                "simind",
                output_prefix,
                output_prefix,
            ]
        else:
            # Standard serial run
            command = ["simind", output_prefix, output_prefix]

        # Add orbit file BEFORE -p flag (must be 3rd/4th/5th argument)
        # Use only filename (not full path) since we chdir to output_dir before running
        if orbit_file:
            command.append(orbit_file.name)

        # Add -p flag for MPI AFTER orbit file
        if mp_value is not None:
            command.append("-p")

        if runtime_switches:
            switch_parts = []
            for k, v in runtime_switches.items():
                if k.upper() == "MP":
                    # MP is just a flag when using mpirun
                    switch_parts.append(f"/{k}")
                else:
                    switch_parts.append(f"/{k}:{v}")
            switches = "".join(switch_parts)
            if switches:
                command.append(switches)

        self.logger.info(f"Running SIMIND: {' '.join(command)}")

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"SIMIND failed: {e}")
            if e.stderr:
                self.logger.error(f"SIMIND stderr: {e.stderr}")
            raise SimulationError(f"SIMIND execution failed: {e}")


# =============================================================================
# ENHANCED OUTPUT PROCESSOR
# =============================================================================


# OutputProcessor has been moved to output_processor.py
# Import it here for backward compatibility
from .output_processor import OutputProcessor

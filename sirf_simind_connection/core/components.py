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

# Import backend interfaces and factories
try:
    from sirf_simind_connection.backends import (
        AcquisitionDataInterface,
        ImageDataInterface,
        create_acquisition_data,
    )
    from sirf_simind_connection.utils.sirf_stir_utils import (
        ensure_acquisition_interface,
    )

    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    create_acquisition_data = None
    ensure_acquisition_interface = None
    ImageDataInterface = type(None)  # type: ignore
    AcquisitionDataInterface = type(None)  # type: ignore


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

    def configure_source_geometry(self, geometry: ImageGeometry) -> None:
        """Configure source image geometry parameters."""
        # Convert to SIMIND units (cm)
        vox_xy_cm = geometry.voxel_x / SIMIND_VOXEL_UNIT_CONVERSION
        vox_z_cm = geometry.voxel_z / SIMIND_VOXEL_UNIT_CONVERSION

        # Set geometric parameters
        self.config.set_value(2, vox_z_cm * geometry.dim_z / 2)
        self.config.set_value(3, vox_xy_cm * geometry.dim_x / 2)
        self.config.set_value(4, vox_xy_cm * geometry.dim_y / 2)
        self.config.set_value(28, vox_xy_cm)
        self.config.set_value(76, geometry.dim_x)
        self.config.set_value(77, geometry.dim_y)

        self.logger.info(
            f"Source geometry: {geometry.dim_x}×{geometry.dim_y}×{geometry.dim_z}"
        )

    def configure_attenuation_geometry(self, geometry: ImageGeometry) -> None:
        """Configure attenuation map geometry parameters."""
        vox_xy_cm = geometry.voxel_x / SIMIND_VOXEL_UNIT_CONVERSION
        vox_z_cm = geometry.voxel_z / SIMIND_VOXEL_UNIT_CONVERSION

        self.config.set_value(5, vox_z_cm * geometry.dim_z / 2)
        self.config.set_value(6, vox_xy_cm * geometry.dim_x / 2)
        self.config.set_value(7, vox_xy_cm * geometry.dim_y / 2)
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


class DataFileManager:
    """Manages input data file preparation."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.temp_files: List[Path] = []

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
        source_max = source.max()
        if source_max > 0:
            source_arr = source_arr / source_max * MAX_SOURCE

        source_arr = np.round(source_arr).astype(np.uint16)

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

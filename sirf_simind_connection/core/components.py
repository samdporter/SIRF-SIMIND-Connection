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
try:
    from sirf.STIR import AcquisitionData, ImageData

    SIRF_AVAILABLE = True
except ImportError:
    AcquisitionData = type(None)
    ImageData = type(None)
    SIRF_AVAILABLE = False

# Import backend factory for creating acquisition data objects
try:
    from sirf_simind_connection.backends import create_acquisition_data

    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    create_acquisition_data = None


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
    def from_image(cls, image: ImageData) -> "ImageGeometry":
        if not SIRF_AVAILABLE:
            raise ImportError("SIRF is required for ImageGeometry.from_image()")
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
    def validate_compatibility(image1: ImageData, image2: ImageData) -> None:
        """Check that two images have compatible geometry."""
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
    def validate_square_pixels(image: ImageData) -> None:
        """Check that image has square pixels."""
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

    def prepare_source_file(self, source: ImageData, output_prefix: str) -> str:
        """Prepare source data file for SIMIND."""
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
        mu_map: ImageData,
        output_prefix: str,
        use_attenuation: bool,
        photon_energy: float,
        input_dir: Path,
    ) -> str:
        """Prepare attenuation data file for SIMIND."""
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


class OutputProcessor:
    """Enhanced output processor that handles both scattwin and penetrate outputs."""

    def __init__(self, converter, output_dir: Path):
        """
        Initialize the output processor.

        Args:
            converter: SIMIND to STIR converter instance
            output_dir: Directory containing simulation outputs
        """
        self.converter = converter
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)

    def process_outputs(
        self,
        output_prefix: str,
        template_sinogram_path: Optional[str] = None,
        source: Optional[ImageData] = None,
        scoring_routine: ScoringRoutine = ScoringRoutine.SCATTWIN,
    ) -> Dict[str, AcquisitionData]:
        """
        Process outputs based on the scoring routine used.

        Args:
            output_prefix: Prefix used for output files
            template_sinogram_path: Path to template sinogram header file (.hs)
            source: Source image for geometry reference
            scoring_routine: Scoring routine that was used

        Returns:
            Dictionary of output name -> AcquisitionData
        """

        if scoring_routine == ScoringRoutine.SCATTWIN:
            return self._process_scattwin_outputs(
                output_prefix, template_sinogram_path, source
            )
        elif scoring_routine == ScoringRoutine.PENETRATE:
            return self._process_penetrate_outputs(
                output_prefix, template_sinogram_path, source
            )
        else:
            raise ValueError(
                f"Unsupported scoring routine for output processing: {scoring_routine}"
            )

    def _process_scattwin_outputs(
        self,
        output_prefix: str,
        template_sinogram_path: Optional[str],
        source: Optional[ImageData],
    ) -> Dict[str, AcquisitionData]:
        """Process scattwin routine outputs (existing functionality)."""

        h00_files = self._find_scattwin_output_files(output_prefix)

        if not h00_files:
            raise OutputError("No SIMIND scattwin output files found")

        # Process each file
        for h00_file in h00_files:
            self._process_single_scattwin_file(h00_file, template_sinogram_path, source)

        # Load and organize converted files
        return self._load_converted_scattwin_files(output_prefix)

    def _process_penetrate_outputs(self, output_prefix, template_sinogram, source):
        # Find the single .h00 file from penetrate routine
        h00_file = self.converter.find_penetrate_h00_file(
            output_prefix, str(self.output_dir)
        )

        if not h00_file:
            raise OutputError("No penetrate .h00 file found")

        # Create multiple .hs files, one for each .bXX file
        outputs = self.converter.create_penetrate_headers_from_template(
            h00_file, output_prefix, str(self.output_dir)
        )

        if not outputs:
            raise OutputError("No penetrate output files found")

        return outputs

    def _find_scattwin_output_files(self, output_prefix: str) -> List[Path]:
        """Find SIMIND scattwin output files."""
        scatter_types = ["_air_w", "_sca_w", "_tot_w", "_pri_w"]
        return [
            f
            for f in self.output_dir.glob("*.h00")
            if any(s in f.name for s in scatter_types) and output_prefix in f.name
        ]

    def _process_single_scattwin_file(
        self,
        h00_file: Path,
        template_sinogram_path: Optional[str],
        source: Optional[ImageData],
    ) -> None:
        """Process a single scattwin output file with corrections."""
        try:
            # Apply template-based corrections
            if template_sinogram_path:
                self._apply_template_corrections(h00_file, template_sinogram_path)

            if source:
                self._validate_scaling_factors(h00_file, source)

            # Convert to STIR format
            self.converter.convert_file(str(h00_file))
            self.logger.info(f"Processed {h00_file.name}")

        except Exception as e:
            self.logger.error(f"Failed to process {h00_file}: {e}")
            raise OutputError(f"Failed to process {h00_file}: {e}")

    def _apply_template_corrections(
        self, h00_file: Path, template_sinogram_path: str
    ) -> None:
        """Apply corrections based on template sinogram header file.

        Args:
            h00_file: SIMIND output file to correct
            template_sinogram_path: Path to template sinogram header file (.hs)
        """
        try:
            # Extract attributes from template sinogram (backend-agnostic!)
            from sirf_simind_connection.utils.stir_utils import (
                extract_attributes_from_stir,
            )

            attributes = extract_attributes_from_stir(template_sinogram_path)

            # Template correction 1: Set acquisition time (projections × time per
            # projection)
            if "number_of_projections" in attributes and "image_duration" in attributes:
                time_per_projection = (
                    attributes["image_duration"] / attributes["number_of_projections"]
                )
                total_duration = (
                    attributes["number_of_projections"] * time_per_projection
                )
                self.converter.edit_parameter(
                    str(h00_file), "!image duration (sec)[1]", total_duration
                )
                self.logger.debug(f"Set image duration: {total_duration} s")

            # Template correction 2: Check and correct radius from template sinogram
            if "height_to_detector_surface" in attributes:
                expected_radius = attributes[
                    "height_to_detector_surface"
                ]  # Already in mm from STIR
                current_radius = self.converter.read_parameter(
                    str(h00_file), ";# Radius"
                )

                if (
                    current_radius is None
                    or abs(float(current_radius) - expected_radius) > 0.1
                ):
                    self.logger.info(
                        f"Correcting radius from template: {expected_radius:.4f} mm"
                    )
                    self.converter.edit_parameter(
                        str(h00_file), ";#Radius", expected_radius
                    )

            # Template correction 3: Handle non-circular orbits if present
            if attributes.get("orbit") == "non-circular" and "radii" in attributes:
                radii = attributes["radii"]  # Already in mm from STIR
                orbits_string = "{" + ",".join([f"{r:.1f}" for r in radii]) + "}"
                self.converter.add_parameter(
                    str(h00_file),
                    "Radii",
                    orbits_string,
                    59,  # line number to insert at
                )
                self.logger.debug("Added non-circular orbit radii from template")

        except Exception as e:
            self.logger.warning(
                f"Failed to apply template corrections to {h00_file}: {e}"
            )

    def _validate_scaling_factors(self, h00_file: Path, source: ImageData) -> None:
        """Validate and fix scaling factors against source image."""
        try:
            # Get voxel size from source image (in mm)
            voxel_size = source.voxel_sizes()[2]  # Get voxel size in z-direction
            self.logger.debug(f"Source voxel size: {voxel_size:.3f} mm")

            # Validate and fix scaling factors using the converter
            scaling_ok = self.converter.validate_and_fix_scaling_factors(
                str(h00_file), source, tolerance=0.00001
            )

            if not scaling_ok:
                self.logger.info(f"Corrected scaling factors in {h00_file.name}")
            else:
                self.logger.debug(f"Scaling factors validated for {h00_file.name}")

        except Exception as e:
            self.logger.warning(
                f"Failed to validate scaling factors for {h00_file}: {e}"
            )

    def _load_converted_scattwin_files(
        self, output_prefix: str
    ) -> Dict[str, AcquisitionData]:
        """Load all converted scattwin .hs files."""
        output = {}
        hs_files = list(self.output_dir.glob(f"*{output_prefix}*.hs"))

        for hs_file in hs_files:
            try:
                # Extract scatter type and window from filename
                key = self._extract_output_key(hs_file.name)
                if BACKEND_AVAILABLE:
                    # Return wrapped backend-agnostic object
                    output[key] = create_acquisition_data(str(hs_file))
                else:
                    output[key] = AcquisitionData(str(hs_file))
            except Exception as e:
                self.logger.error(f"Failed to load {hs_file}: {e}")
                continue

        if not output:
            raise OutputError("No valid scattwin output files could be loaded")

        self.logger.info(f"Loaded {len(output)} scattwin output files")
        return output

    def _extract_output_key(self, filename: str) -> str:
        """Extract scatter type and window from filename."""
        # Parse filename to extract scatter type and window number
        parts = filename.split("_")
        if len(parts) >= 2:
            scatter_type = parts[-2]
            window = parts[-1].split(".")[0]
            return f"{scatter_type}_{window}"
        return filename

    def _get_penetrate_output_name(self, component: PenetrateOutputType) -> str:
        """Get descriptive name for penetrate output component."""
        name_mapping = {
            PenetrateOutputType.ALL_INTERACTIONS: "all_interactions",
            PenetrateOutputType.GEOM_COLL_PRIMARY_ATT: "geom_coll_primary",
            PenetrateOutputType.SEPTAL_PENETRATION_PRIMARY_ATT: "septal_pen_primary",
            PenetrateOutputType.COLL_SCATTER_PRIMARY_ATT: "coll_scatter_primary",
            PenetrateOutputType.COLL_XRAY_PRIMARY_ATT: "coll_xray_primary",
            PenetrateOutputType.GEOM_COLL_SCATTERED: "geom_coll_scattered",
            PenetrateOutputType.SEPTAL_PENETRATION_SCATTERED: "septal_pen_scattered",
            PenetrateOutputType.COLL_SCATTER_SCATTERED: "coll_scatter_scattered",
            PenetrateOutputType.COLL_XRAY_SCATTERED: "coll_xray_scattered",
            PenetrateOutputType.GEOM_COLL_PRIMARY_ATT_BACK: "geom_coll_primary_back",
            PenetrateOutputType.SEPTAL_PENETRATION_PRIMARY_ATT_BACK: (
                "septal_pen_primary_back"
            ),
            PenetrateOutputType.COLL_SCATTER_PRIMARY_ATT_BACK: (
                "coll_scatter_primary_back"
            ),
            PenetrateOutputType.COLL_XRAY_PRIMARY_ATT_BACK: "coll_xray_primary_back",
            PenetrateOutputType.GEOM_COLL_SCATTERED_BACK: "geom_coll_scattered_back",
            PenetrateOutputType.SEPTAL_PENETRATION_SCATTERED_BACK: (
                "septal_pen_scattered_back"
            ),
            PenetrateOutputType.COLL_SCATTER_SCATTERED_BACK: (
                "coll_scatter_scattered_back"
            ),
            PenetrateOutputType.COLL_XRAY_SCATTERED_BACK: "coll_xray_scattered_back",
            PenetrateOutputType.ALL_UNSCATTERED_UNATTENUATED: (
                "unscattered_unattenuated"
            ),
            PenetrateOutputType.ALL_UNSCATTERED_UNATTENUATED_GEOM_COLL: (
                "unscattered_unattenuated_geom_coll"
            ),
        }
        return name_mapping.get(component, f"component_{component.value}")

    def get_penetrate_component_description(
        self, component: PenetrateOutputType
    ) -> str:
        """Get detailed description for penetrate output component."""
        descriptions = {
            PenetrateOutputType.ALL_INTERACTIONS: "All type of interactions",
            PenetrateOutputType.GEOM_COLL_PRIMARY_ATT: (
                "Geometrically collimated primary attenuated photons from phantom"
            ),
            PenetrateOutputType.SEPTAL_PENETRATION_PRIMARY_ATT: (
                "Septal penetration from primary attenuated photons from phantom"
            ),
            PenetrateOutputType.COLL_SCATTER_PRIMARY_ATT: (
                "Collimator scatter from primary attenuated photons from phantom"
            ),
            PenetrateOutputType.COLL_XRAY_PRIMARY_ATT: (
                "X-rays from collimator from primary attenuated photons from phantom"
            ),
            PenetrateOutputType.GEOM_COLL_SCATTERED: (
                "Geometrically collimated from scattered photons from phantom"
            ),
            PenetrateOutputType.SEPTAL_PENETRATION_SCATTERED: (
                "Septal penetration from scattered photons from phantom"
            ),
            PenetrateOutputType.COLL_SCATTER_SCATTERED: (
                "Collimator scatter from scattered photons from phantom"
            ),
            PenetrateOutputType.COLL_XRAY_SCATTERED: (
                "X-rays from collimator from scattered photons from phantom"
            ),
            PenetrateOutputType.GEOM_COLL_PRIMARY_ATT_BACK: (
                "Geometrically collimated primary attenuated photons (with backscatter)"
            ),
            PenetrateOutputType.SEPTAL_PENETRATION_PRIMARY_ATT_BACK: (
                "Septal penetration from primary attenuated photons (with backscatter)"
            ),
            PenetrateOutputType.COLL_SCATTER_PRIMARY_ATT_BACK: (
                "Collimator scatter from primary attenuated photons (with backscatter)"
            ),
            PenetrateOutputType.COLL_XRAY_PRIMARY_ATT_BACK: (
                "X-rays from collimator from primary attenuated photons "
                "(with backscatter)"
            ),
            PenetrateOutputType.GEOM_COLL_SCATTERED_BACK: (
                "Geometrically collimated scattered photons (with backscatter)"
            ),
            PenetrateOutputType.SEPTAL_PENETRATION_SCATTERED_BACK: (
                "Septal penetration from scattered photons (with backscatter)"
            ),
            PenetrateOutputType.COLL_SCATTER_SCATTERED_BACK: (
                "Collimator scatter from scattered photons (with backscatter)"
            ),
            PenetrateOutputType.COLL_XRAY_SCATTERED_BACK: (
                "X-rays from collimator from scattered photons (with backscatter)"
            ),
            PenetrateOutputType.ALL_UNSCATTERED_UNATTENUATED: (
                "Photons without scattering and attenuation in phantom"
            ),
            PenetrateOutputType.ALL_UNSCATTERED_UNATTENUATED_GEOM_COLL: (
                "Photons without scattering and attenuation, geometrically collimated"
            ),
        }
        return descriptions.get(
            component, f"Component {component.value} - see SIMIND manual for details"
        )

    def list_expected_files(
        self, output_prefix: str, scoring_routine: ScoringRoutine
    ) -> List[str]:
        """List expected output files for a given scoring routine."""
        if scoring_routine == ScoringRoutine.SCATTWIN:
            # Scattwin files for window 1 (most common case)
            return [
                f"{output_prefix}_tot_w1.a00",
                f"{output_prefix}_sca_w1.a00",
                f"{output_prefix}_pri_w1.a00",
                f"{output_prefix}_air_w1.a00",
            ]
        elif scoring_routine == ScoringRoutine.PENETRATE:
            # All possible penetrate files
            return [f"{output_prefix}.b{i:02d}" for i in range(1, 20)]
        else:
            return []

    def cleanup_temp_files(self) -> None:
        """Clean up any temporary files created during processing."""
        # This could be extended to clean up converter temporary files
        # For now, just log that cleanup was called
        self.logger.debug("Output processor cleanup completed")

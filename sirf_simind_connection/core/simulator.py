"""
Simind Simulator Class.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# Import backend factory and interfaces using centralized access
from sirf_simind_connection.utils.backend_access import get_backend_interfaces
from sirf_simind_connection.utils.sirf_stir_utils import register_and_enforce_backend

BACKEND_AVAILABLE, _backends = get_backend_interfaces()

# Unpack interfaces needed by simulator
ensure_acquisition_interface = _backends['wrappers']['ensure_acquisition_interface']
ensure_image_interface = _backends['wrappers']['ensure_image_interface']
to_native_acquisition = _backends['wrappers']['to_native_acquisition']
detect_image_backend = _backends['detection']['detect_image_backend']
detect_acquisition_backend = _backends['detection']['detect_acquisition_backend']
detect_backend_from_interface = _backends['detection']['detect_backend_from_interface']
get_backend = _backends['detection']['get_backend']
set_backend = _backends['detection']['set_backend']
AcquisitionDataInterface = _backends['types']['AcquisitionDataInterface']
ImageDataInterface = _backends['types']['ImageDataInterface']

from sirf_simind_connection.converters.simind_to_stir import SimindToStirConverter
from sirf_simind_connection.utils.stir_utils import extract_attributes_from_stir

from .components import (  # Exceptions; Data classes; Managers and processors;
    # Constants
    SIMIND_VOXEL_UNIT_CONVERSION,
    AcquisitionManager,
    DataFileManager,
    EnergyWindow,
    GeometryManager,
    ImageGeometry,
    ImageValidator,
    OrbitFileManager,
    OutputError,
    OutputProcessor,
    PenetrateOutputType,
    RotationDirection,
    RotationParameters,
    ScoringRoutine,
    SimindExecutor,
    ValidationError,
)
from .config import RuntimeSwitches, SimulationConfig


class SimindSimulator:
    """
    Enhanced SIMIND simulator with support for both scattwin and penetrate
    scoring routines.
    """

    def __init__(
        self,
        config_source: Union[str, SimulationConfig],
        output_dir: str,
        output_prefix: str = "output",
        photon_multiplier: int = 1,
        scoring_routine: Union[ScoringRoutine, int] = ScoringRoutine.SCATTWIN,
    ):
        """
        Initialize the simulator with flexible configuration source.

        Args:
            config_source: Can be string path to .smc/.yaml file or
                SimulationConfig object
            output_dir: Directory for simulation outputs
            output_prefix: Prefix for output files
            photon_multiplier: Photon multiplier for NN runtime switch
            scoring_routine: Scoring routine to use (SCATTWIN or PENETRATE)
        """
        self.logger = logging.getLogger(__name__)

        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_prefix = output_prefix

        # Handle scoring routine
        if isinstance(scoring_routine, int):
            self.scoring_routine = ScoringRoutine(scoring_routine)
        else:
            self.scoring_routine = scoring_routine

        # Initialize configuration based on source type
        self.config, self.template_path = self._initialize_config(config_source)

        # Initialize runtime switches
        self.runtime_switches = RuntimeSwitches()
        self.runtime_switches.set_switch("NN", photon_multiplier)

        # Initialize components with enhanced output processor
        self.converter = SimindToStirConverter()
        self.geometry_manager = GeometryManager(self.config)
        self.acquisition_manager = AcquisitionManager(
            self.config, self.runtime_switches
        )
        self.file_manager = DataFileManager(self.output_dir)
        self.orbit_manager = OrbitFileManager(self.output_dir)
        self.executor = SimindExecutor()
        self.output_processor = OutputProcessor(self.converter, self.output_dir)

        # Set up for voxelised phantom simulation
        self._configure_voxelised_phantom()

        # Configure scoring routine
        self._configure_scoring_routine()

        # Simulation state
        self.source: Optional[ImageDataInterface] = None
        self.mu_map: Optional[ImageDataInterface] = None
        # Template sinogram: store both filepath (backend-agnostic) and wrapped object
        self.template_sinogram: Optional[AcquisitionDataInterface] = None
        self.template_sinogram_path: Optional[str] = None
        self.attributes: Dict = {}
        self.energy_windows: List[EnergyWindow] = []
        self.rotation_params: Optional[RotationParameters] = None
        self.non_circular_orbit = False
        self.orbit_radii: List[float] = []

        # Results
        self._outputs: Optional[Dict[str, AcquisitionDataInterface]] = None
        self._preferred_backend: Optional[str] = None

        self.logger.info(
            f"Simulator initialized with {self.scoring_routine.name} scoring routine"
        )

    def _configure_scoring_routine(self) -> None:
        """Configure the scoring routine in the simulation config."""
        self.config.set_value(
            84, self.scoring_routine.value
        )  # Index 84 = scoring routine

        # Set appropriate flags based on scoring routine
        if self.scoring_routine == ScoringRoutine.PENETRATE:
            # Penetrate routine may need specific configuration
            self.logger.info("Configured for penetrate scoring routine")
        else:
            # Default scattwin configuration
            self.logger.info("Configured for scattwin scoring routine")

    def _register_backend_hint(self, backend: Optional[str]) -> None:
        """Record backend preference, ensuring we don't mix SIRF and STIR.

        This method now delegates to the centralized register_and_enforce_backend
        helper from sirf_stir_utils, eliminating duplicate backend enforcement logic.
        """
        if BACKEND_AVAILABLE and register_and_enforce_backend is not None:
            self._preferred_backend = register_and_enforce_backend(
                backend, self._preferred_backend
            )

    def _initialize_config(
        self, config_source: Union[str, SimulationConfig]
    ) -> tuple[SimulationConfig, Optional[Path]]:
        """
        Initialize configuration from various source types.

        Returns:
            tuple: (SimulationConfig object, template_path if applicable)
        """
        if isinstance(config_source, SimulationConfig):
            # Direct SimulationConfig object
            self.logger.info("Using provided SimulationConfig object")
            return config_source, None

        elif isinstance(config_source, str):
            config_path = Path(config_source).resolve()

            if not config_path.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {config_source}"
                )

            if config_path.suffix.lower() == ".smc":
                # SMC template file (original behavior)
                self.logger.info(f"Loading SMC template file: {config_path}")
                return SimulationConfig(str(config_path)), config_path

            elif config_path.suffix.lower() in [".yaml", ".yml"]:
                # YAML configuration file
                self.logger.info(f"Loading YAML configuration file: {config_path}")
                return SimulationConfig(str(config_path)), config_path

            else:
                raise ValueError(
                    f"Unsupported configuration file type: {config_path.suffix}"
                )

        else:
            raise TypeError(
                f"config_source must be string path or SimulationConfig object, "
                f"got {type(config_source)}"
            )

    def _configure_voxelised_phantom(self) -> None:
        """Configure settings for voxelised phantom simulation."""
        self.config.set_flag(5, True)  # SPECT study
        self.config.set_value(15, -1)  # source type
        self.config.set_value(14, -1)  # phantom type
        self.config.set_flag(14, True)  # write to interfile header

    # =============================================================================
    # CONFIGURATION ACCESS METHODS
    # =============================================================================

    def get_config(self) -> SimulationConfig:
        """Get the current simulation configuration."""
        return self.config

    def save_config_as_yaml(self, yaml_path: str) -> None:
        """Save current configuration to YAML file."""
        yaml_data = {
            "config_values": self.config.get_all_values(),
            "config_flags": self.config.get_all_flags(),
            "data_files": self.config.get_all_data_files(),
        }

        if hasattr(self.config, "get_photon_energy"):
            yaml_data["photon_energy"] = self.config.get_photon_energy()

        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, indent=2)

        self.logger.info(f"Configuration saved to YAML: {yaml_path}")

    # =============================================================================
    # INPUT CONFIGURATION METHODS (UNCHANGED)
    # =============================================================================

    def set_source(self, source: Union[str, ImageDataInterface]) -> None:
        """Set the source image."""
        if not BACKEND_AVAILABLE or ensure_image_interface is None:
            raise ImportError(
                "SIRF/STIR backend wrappers are not available to load image data"
            )

        try:
            # Detect backend from the provided value
            if not isinstance(source, str):
                backend = detect_image_backend(source)
                if backend is None and isinstance(source, ImageDataInterface):
                    backend = detect_backend_from_interface(source)
                self._register_backend_hint(backend)

            # Use preferred backend if we have one, otherwise let the backend module decide
            self.source = ensure_image_interface(
                source, preferred_backend=self._preferred_backend
            )

            # Register the backend from the wrapped result
            backend = detect_backend_from_interface(self.source)
            self._register_backend_hint(backend)
        except Exception as exc:
            raise TypeError(
                "source must be a string path or backend-compatible image object"
            ) from exc

        # Validate and configure geometry
        ImageValidator.validate_square_pixels(self.source)
        geometry = ImageGeometry.from_image(self.source)
        self.geometry_manager.configure_source_geometry(geometry)

        self.logger.info(
            f"Source configured: {geometry.dim_x}×{geometry.dim_y}×{geometry.dim_z}"
        )

    def set_mu_map(self, mu_map: Union[str, ImageDataInterface]) -> None:
        """Set the attenuation map."""
        if not BACKEND_AVAILABLE or ensure_image_interface is None:
            raise ImportError(
                "SIRF/STIR backend wrappers are not available to load attenuation data"
            )

        try:
            # Detect backend from the provided value
            if not isinstance(mu_map, str):
                backend = detect_image_backend(mu_map)
                if backend is None and isinstance(mu_map, ImageDataInterface):
                    backend = detect_backend_from_interface(mu_map)
                self._register_backend_hint(backend)

            # Use preferred backend if we have one
            self.mu_map = ensure_image_interface(mu_map, preferred_backend=self._preferred_backend)

            # Register the backend from the wrapped result
            backend = detect_backend_from_interface(self.mu_map)
            self._register_backend_hint(backend)
        except Exception as exc:
            raise TypeError(
                "mu_map must be a string path or backend-compatible image object"
            ) from exc

        # Validate and configure geometry
        ImageValidator.validate_square_pixels(self.mu_map)
        geometry = ImageGeometry.from_image(self.mu_map)
        self.geometry_manager.configure_attenuation_geometry(geometry)

        self.logger.info(
            f"Attenuation map configured: {geometry.dim_x}×{geometry.dim_y}"
            f"×{geometry.dim_z}"
        )

    def set_energy_windows(
        self,
        lower_bounds: Union[float, List[float]],
        upper_bounds: Union[float, List[float]],
        scatter_orders: Union[int, List[int]],
    ) -> None:
        """Set energy windows for the simulation."""

        if self.scoring_routine == ScoringRoutine.PENETRATE:
            self.logger.warning(
                "Energy windows configuration is not applicable for penetrate routine"
            )
            self.logger.warning(
                "Penetrate routine analyzes all interactions regardless of "
                "energy windows"
            )

        # Convert single values to lists
        if not isinstance(lower_bounds, list):
            lower_bounds = [lower_bounds]
        if not isinstance(upper_bounds, list):
            upper_bounds = [upper_bounds]
        if not isinstance(scatter_orders, list):
            scatter_orders = [scatter_orders]

        # Create EnergyWindow objects
        self.energy_windows = [
            EnergyWindow(lb, ub, so, i + 1)
            for i, (lb, ub, so) in enumerate(
                zip(lower_bounds, upper_bounds, scatter_orders)
            )
        ]

        self.logger.info(f"Configured {len(self.energy_windows)} energy windows")

    def set_collimator_routine(self, enabled: bool) -> None:
        """
        Enable or disable collimator modeling (penetration routine).

        This sets index 53 in the SIMIND configuration:
        - 0: No collimator modeling (geometric only)
        - 1: Full collimator modeling (penetration, scatter, etc.)

        Args:
            enabled (bool): True to enable collimator modeling, False for
                geometric only.
        """
        self.config.set_value(53, 1 if enabled else 0)
        mode_str = "enabled" if enabled else "disabled"
        self.logger.info(
            f"Collimator routine {mode_str} (index 53 = {1 if enabled else 0})"
        )

    def set_template_sinogram(
        self, template_sinogram: Union[str, AcquisitionDataInterface]
    ) -> None:
        """Set template sinogram and extract acquisition parameters.

        Args:
            template_sinogram: Filepath to .hs header, backend interface, or native
                acquisition object compatible with the wrapper factory.
        """
        import tempfile

        if not BACKEND_AVAILABLE or create_acquisition_data is None:
            raise ImportError(
                "SIRF/STIR backend wrappers are not available to load acquisition data"
            )

        if isinstance(template_sinogram, str):
            # Store filepath directly (backend-agnostic)
            self.template_sinogram_path = template_sinogram
            # Load wrapped object
            self.template_sinogram = ensure_acquisition_interface(
                template_sinogram, preferred_backend=self._preferred_backend
            )
        else:
            # Detect backend from the provided value
            backend = detect_acquisition_backend(template_sinogram)
            if backend is None and isinstance(template_sinogram, AcquisitionDataInterface):
                backend = detect_backend_from_interface(template_sinogram)
            self._register_backend_hint(backend)

            wrapped = ensure_acquisition_interface(
                template_sinogram, preferred_backend=self._preferred_backend
            )

            # Object provided - write to temp file to get filepath
            temp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".hs", delete=False, dir=str(self.output_dir)
            )
            temp_path = temp_file.name
            temp_file.close()

            # Write object to file
            wrapped.write(temp_path)
            self.template_sinogram_path = temp_path

            # Clone the object
            self.template_sinogram = wrapped.clone()

        # Register the backend from the wrapped result
        backend = detect_backend_from_interface(self.template_sinogram)
        self._register_backend_hint(backend)

        # Extract parameters from template using filepath (backend-agnostic!)
        self.attributes = extract_attributes_from_stir(self.template_sinogram_path)

        # Set up rotation parameters
        direction = (
            RotationDirection.CCW
            if self.attributes["direction_of_rotation"].lower() == "ccw"
            else RotationDirection.CW
        )
        self.rotation_params = RotationParameters(
            direction=direction,
            rotation_angle=self.attributes["extent_of_rotation"],
            start_angle=self.attributes["start_angle"],
            num_projections=self.attributes["number_of_projections"],
        )

        # Configure acquisition
        detector_distance = self.attributes["height_to_detector_surface"]
        self.acquisition_manager.configure_rotation(
            self.rotation_params, detector_distance
        )

        # Handle non-circular orbits
        if (
            self.attributes.get("orbit") == "non-circular"
            and "radii" in self.attributes
        ):
            self.non_circular_orbit = True
            self.orbit_radii = self.attributes["radii"]
            self.logger.info("Non-circular orbit detected")

        self.logger.info("Template sinogram configured")

    def set_rotation_parameters(
        self,
        direction: str,
        rotation_angle: float,
        start_angle: float,
        num_projections: int,
        detector_distance: float,
    ) -> None:
        """Manually set rotation parameters."""
        direction_enum = (
            RotationDirection.CCW
            if direction.lower() == "ccw"
            else RotationDirection.CW
        )

        self.rotation_params = RotationParameters(
            direction=direction_enum,
            rotation_angle=rotation_angle,
            start_angle=start_angle,
            num_projections=num_projections,
        )

        self.acquisition_manager.configure_rotation(
            self.rotation_params, detector_distance
        )
        self.logger.info(
            f"Rotation configured: {direction} {rotation_angle}° from {start_angle}°"
        )

    # =============================================================================
    # CONFIGURATION METHODS (UNCHANGED)
    # =============================================================================

    def add_config_value(self, index: int, value) -> None:
        """Add a configuration index value."""
        self.config.set_value(index, value)

    def add_config_flag(self, flag: int, value: bool) -> None:
        """Add a configuration flag."""
        self.config.set_flag(flag, value)

    def set_scoring_routine(self, scoring_routine: Union[ScoringRoutine, int]) -> None:
        """Update scoring routine and reconfigure related settings."""

        if isinstance(scoring_routine, int):
            scoring = ScoringRoutine(scoring_routine)
        else:
            scoring = scoring_routine

        if scoring == self.scoring_routine:
            return

        self.scoring_routine = scoring
        self._configure_scoring_routine()
        self.logger.info("Scoring routine updated to %s", self.scoring_routine.name)

    def add_runtime_switch(self, switch: str, value) -> None:
        """
        Add a runtime switch.

        Special handling for CC (collimator): also updates text_variables[1]
        in the .smc file so SIMIND can find the collimator file.
        """
        self.runtime_switches.set_switch(switch, value)

        # TODO: improve handling of regularly used switches
        # Sync collimator to .smc file text_variables
        if switch == "CC":
            self.config.text_variables[1] = str(value)

    # =============================================================================
    # SIMULATION EXECUTION
    # =============================================================================

    def run_simulation(self) -> None:
        """Run the complete SIMIND simulation."""
        self.logger.info("Starting SIMIND simulation")

        # Reset previous results
        self._outputs = None

        try:
            # Validate inputs
            self._validate_simulation_inputs()

            # Prepare simulation
            self._prepare_simulation()

            # Execute SIMIND
            self._execute_simulation()

        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            raise
        finally:
            # Cleanup temporary files
            self.file_manager.cleanup_temp_files()

    def _validate_simulation_inputs(self) -> None:
        """Validate all inputs before simulation."""

        # Common validation
        if self.source is None or self.mu_map is None:
            raise ValidationError("Both source and mu_map must be set")

        # Check image compatibility
        ImageValidator.validate_compatibility(self.source, self.mu_map)

        # Routine-specific validation
        if self.scoring_routine == ScoringRoutine.SCATTWIN:
            if not self.energy_windows:
                raise ValidationError("Energy windows must be set for scattwin routine")
        elif self.scoring_routine == ScoringRoutine.PENETRATE:
            # Penetrate routine doesn't need energy windows but may have other
            # requirements
            pass

    def _prepare_simulation(self) -> None:
        """Prepare all files and configuration for simulation."""

        # Configure energy windows only for scattwin
        if self.scoring_routine == ScoringRoutine.SCATTWIN:
            if not self.energy_windows:
                raise ValidationError("Energy windows must be set for scattwin routine")

            self.acquisition_manager.configure_energy_windows(
                self.energy_windows, str(self.output_dir / self.output_prefix)
            )

        # Prepare data files (same for both routines)
        source_file = self.file_manager.prepare_source_file(
            self.source, self.output_prefix
        )
        attenuation_file = self.file_manager.prepare_attenuation_file(
            self.mu_map,
            self.output_prefix,
            self.config.get_flag(11),  # use attenuation
            self.config.get_value("photon_energy"),
            self.template_path.parent if self.template_path else Path.cwd(),
        )

        # Set data files in configuration
        self.config.set_data_file(6, source_file)  # source file
        self.config.set_data_file(5, attenuation_file)  # attenuation file

        # Add PX runtime switch - required for voxelised phantoms
        if self.source:
            voxel_size = self.source.voxel_sizes()[-1]
            self.runtime_switches.set_switch(
                "PX", voxel_size / SIMIND_VOXEL_UNIT_CONVERSION
            )
            self.logger.info(f"Set PX runtime switch to {voxel_size} cm")
        else:
            raise ValidationError("Source image must be set to determine voxel size")

        # Save configuration as .smc file for SIMIND execution
        output_path = self.output_dir / self.output_prefix
        self.config.save_file(output_path)
        self.logger.info(f"Configuration saved as SMC file: {output_path}.smc")

    def _execute_simulation(self) -> None:
        """Execute the SIMIND simulation."""
        # Change to output directory (SIMIND requirement)
        original_cwd = os.getcwd()

        try:
            os.chdir(self.output_dir)

            print(f"Running SIMIND simulation with output prefix: {self.output_prefix}")
            print(f"in directory: {self.output_dir}")

            # Prepare orbit file if needed
            orbit_file = None
            self.logger.debug(
                f"Orbit file check: non_circular={self.non_circular_orbit}, "
                f"radii_count={len(self.orbit_radii) if self.orbit_radii else 0}"
            )
            if self.non_circular_orbit and self.orbit_radii:
                center_of_rotation = (
                    self.source.dimensions()[1] / 2 if self.source else None
                )
                orbit_file = self.orbit_manager.write_orbit_file(
                    self.orbit_radii, self.output_prefix, center_of_rotation
                )
            else:
                self.logger.warning(
                    "Skipping orbit file creation: "
                    f"non_circular_orbit={self.non_circular_orbit}, "
                    f"orbit_radii={'empty' if not self.orbit_radii else f'{len(self.orbit_radii)} values'}"
                )

            # Execute simulation
            self.executor.run_simulation(
                self.output_prefix, orbit_file, self.runtime_switches.switches
            )

        finally:
            os.chdir(original_cwd)

    # =============================================================================
    # OUTPUT METHODS (UNCHANGED)
    # =============================================================================

    def get_outputs(
        self,
        native: bool = False,
        preferred_backend: Optional[str] = None,
    ) -> Dict[str, Union[AcquisitionDataInterface, Any]]:
        """Get all simulation outputs.

        Args:
            native: When True, return native SIRF/STIR acquisition objects.
            preferred_backend: Optional backend to enforce when returning native
                objects. Useful when reading from file paths and you want a
                specific toolkit representation.
        """
        if native:
            if not BACKEND_AVAILABLE or to_native_acquisition is None:
                raise ImportError(
                    "Requesting native outputs requires SIRF/STIR backends."
                )

            if preferred_backend:
                preferred_backend = preferred_backend.lower()
                self._register_backend_hint(preferred_backend)

        if self._outputs is None:
            self._outputs = self.output_processor.process_outputs(
                self.output_prefix,
                self.template_sinogram_path,
                self.source,
                self.scoring_routine,
                preferred_backend=self._preferred_backend,
            )
        if not native:
            return self._outputs

        target_backend = preferred_backend or self._preferred_backend

        return {
            key: to_native_acquisition(
                value,
                preferred_backend=target_backend,
                ensure_interface=False,
            )
            for key, value in self._outputs.items()
        }

    # Scattwin-specific output methods (existing)
    def get_total_output(
        self,
        window: int = 1,
        native: bool = False,
        preferred_backend: Optional[str] = None,
    ) -> Union[AcquisitionDataInterface, Any]:
        """Get total output for specified window (scattwin only).

        Returns:
            Backend-agnostic acquisition data or native object if requested
        """
        if self.scoring_routine != ScoringRoutine.SCATTWIN:
            raise OutputError(
                "get_total_output() is only available for scattwin routine"
            )

        outputs = self.get_outputs(native=native, preferred_backend=preferred_backend)
        key = f"tot_w{window}"
        if key not in outputs:
            raise OutputError(f"Total output for window {window} not found")
        return outputs[key]

    def get_scatter_output(
        self,
        window: int = 1,
        native: bool = False,
        preferred_backend: Optional[str] = None,
    ) -> Union[AcquisitionDataInterface, Any]:
        """Get scatter output for specified window (scattwin only).

        Returns:
            Backend-agnostic acquisition data (wrapped) or native if requested
        """
        if self.scoring_routine != ScoringRoutine.SCATTWIN:
            raise OutputError(
                "get_scatter_output() is only available for scattwin routine"
            )

        outputs = self.get_outputs(native=native, preferred_backend=preferred_backend)
        key = f"sca_w{window}"
        if key not in outputs:
            raise OutputError(f"Scatter output for window {window} not found")
        return outputs[key]

    def get_primary_output(
        self,
        window: int = 1,
        native: bool = False,
        preferred_backend: Optional[str] = None,
    ) -> Union[AcquisitionDataInterface, Any]:
        """Get primary output for specified window (scattwin only).

        Returns:
            Backend-agnostic acquisition data or native object if requested
        """
        if self.scoring_routine != ScoringRoutine.SCATTWIN:
            raise OutputError(
                "get_primary_output() is only available for scattwin routine"
            )

        outputs = self.get_outputs(native=native, preferred_backend=preferred_backend)
        key = f"pri_w{window}"
        if key not in outputs:
            raise OutputError(f"Primary output for window {window} not found")
        return outputs[key]

    def get_air_output(
        self,
        window: int = 1,
        native: bool = False,
        preferred_backend: Optional[str] = None,
    ) -> Union[AcquisitionDataInterface, Any]:
        """Get air output for specified window (scattwin only)."""
        if self.scoring_routine != ScoringRoutine.SCATTWIN:
            raise OutputError("get_air_output() is only available for scattwin routine")

        outputs = self.get_outputs(native=native, preferred_backend=preferred_backend)
        key = f"air_w{window}"
        if key not in outputs:
            raise OutputError(f"Air output for window {window} not found")
        return outputs[key]

    # Penetrate-specific output methods (new)
    def get_penetrate_output(
        self,
        component: Union[PenetrateOutputType, str],
        native: bool = False,
        preferred_backend: Optional[str] = None,
    ) -> Union[AcquisitionDataInterface, Any]:
        """Get penetrate output for specified component.

        Returns:
            Backend-agnostic acquisition data (wrapped) or native if requested
        """
        if self.scoring_routine != ScoringRoutine.PENETRATE:
            raise OutputError(
                "get_penetrate_output() is only available for penetrate routine"
            )

        outputs = self.get_outputs(native=native, preferred_backend=preferred_backend)

        if isinstance(component, PenetrateOutputType):
            component_name = self.output_processor._get_penetrate_output_name(component)
        else:
            component_name = component

        if component_name not in outputs:
            available = list(outputs.keys())
            raise OutputError(
                f"Penetrate component '{component_name}' not found. "
                f"Available: {available}"
            )

        return outputs[component_name]

    def get_all_interactions(
        self, native: bool = False, preferred_backend: Optional[str] = None
    ) -> Union[AcquisitionDataInterface, Any]:
        """Get all interactions output (penetrate routine)."""
        return self.get_penetrate_output(
            PenetrateOutputType.ALL_INTERACTIONS,
            native=native,
            preferred_backend=preferred_backend,
        )

    def get_geometrically_collimated_primary(
        self,
        with_backscatter: bool = False,
        native: bool = False,
        preferred_backend: Optional[str] = None,
    ) -> Union[AcquisitionDataInterface, Any]:
        """Get geometrically collimated primary photons."""
        component = (
            PenetrateOutputType.GEOM_COLL_PRIMARY_ATT_BACK
            if with_backscatter
            else PenetrateOutputType.GEOM_COLL_PRIMARY_ATT
        )
        return self.get_penetrate_output(
            component, native=native, preferred_backend=preferred_backend
        )

    def get_septal_penetration(
        self,
        primary: bool = True,
        with_backscatter: bool = False,
        native: bool = False,
        preferred_backend: Optional[str] = None,
    ) -> Union[AcquisitionDataInterface, Any]:
        """Get septal penetration component."""
        if primary:
            component = (
                PenetrateOutputType.SEPTAL_PENETRATION_PRIMARY_ATT_BACK
                if with_backscatter
                else PenetrateOutputType.SEPTAL_PENETRATION_PRIMARY_ATT
            )
        else:
            component = (
                PenetrateOutputType.SEPTAL_PENETRATION_SCATTERED_BACK
                if with_backscatter
                else PenetrateOutputType.SEPTAL_PENETRATION_SCATTERED
            )
        return self.get_penetrate_output(
            component, native=native, preferred_backend=preferred_backend
        )

    def get_collimator_scatter(
        self,
        primary: bool = True,
        with_backscatter: bool = False,
        native: bool = False,
        preferred_backend: Optional[str] = None,
    ) -> Union[AcquisitionDataInterface, Any]:
        """Get collimator scatter component."""
        if primary:
            component = (
                PenetrateOutputType.COLL_SCATTER_PRIMARY_ATT_BACK
                if with_backscatter
                else PenetrateOutputType.COLL_SCATTER_PRIMARY_ATT
            )
        else:
            component = (
                PenetrateOutputType.COLL_SCATTER_SCATTERED_BACK
                if with_backscatter
                else PenetrateOutputType.COLL_SCATTER_SCATTERED
            )
        return self.get_penetrate_output(
            component, native=native, preferred_backend=preferred_backend
        )

    def list_available_outputs(self) -> List[str]:
        """List all available output components for the current scoring routine."""
        outputs = self.get_outputs()
        return list(outputs.keys())

    def get_scoring_routine(self) -> ScoringRoutine:
        """Get the current scoring routine."""
        return self.scoring_routine


# =============================================================================
# CONVENIENCE FACTORY FUNCTIONS
# =============================================================================


def create_simulator_from_template(
    config_source: Union[str, SimulationConfig],
    output_dir: str,
    template_sinogram: Union[str, AcquisitionDataInterface],
    source: Union[str, ImageDataInterface],
    mu_map: Union[str, ImageDataInterface],
    scoring_routine: Union[ScoringRoutine, int] = ScoringRoutine.SCATTWIN,
    energy_windows: Optional[Dict] = None,
    **kwargs,
) -> SimindSimulator:
    """Factory function to create a fully configured simulator from SMC template."""

    simulator = SimindSimulator(
        config_source,
        output_dir,
        output_prefix=kwargs.get("output_prefix", "output"),
        photon_multiplier=kwargs.get("photon_multiplier", 1),
        scoring_routine=scoring_routine,
    )

    # Set all inputs
    simulator.set_source(source)
    simulator.set_mu_map(mu_map)
    simulator.set_template_sinogram(template_sinogram)

    # Set energy windows only for scattwin
    if scoring_routine == ScoringRoutine.SCATTWIN and energy_windows:
        simulator.set_energy_windows(**energy_windows)

    return simulator


def create_penetrate_simulator(
    config_source: Union[str, SimulationConfig],
    output_dir: str,
    template_sinogram: Union[str, AcquisitionDataInterface],
    source: Union[str, ImageDataInterface],
    mu_map: Union[str, ImageDataInterface],
    **kwargs,
) -> SimindSimulator:
    """Factory function to create a simulator specifically for penetrate routine."""

    return create_simulator_from_template(
        config_source,
        output_dir,
        template_sinogram,
        source,
        mu_map,
        scoring_routine=ScoringRoutine.PENETRATE,
        **kwargs,
    )


def create_scattwin_simulator(
    config_source: Union[str, SimulationConfig],
    output_dir: str,
    template_sinogram: Union[str, AcquisitionDataInterface],
    source: Union[str, ImageDataInterface],
    mu_map: Union[str, ImageDataInterface],
    energy_windows: Dict,
    **kwargs,
) -> SimindSimulator:
    """Factory function to create a simulator specifically for scattwin routine."""

    return create_simulator_from_template(
        config_source,
        output_dir,
        template_sinogram,
        source,
        mu_map,
        scoring_routine=ScoringRoutine.SCATTWIN,
        energy_windows=energy_windows,
        **kwargs,
    )

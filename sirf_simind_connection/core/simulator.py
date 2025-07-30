"""
Main SimindSimulator class - now much simpler and focused on orchestration.
Enhanced to accept different configuration sources.
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union

from sirf.STIR import AcquisitionData, ImageData

from sirf_simind_connection.converters.simind_to_stir import SimindToStirConverter
from sirf_simind_connection.utils.stir_utils import extract_attributes_from_stir

from .config import RuntimeSwitches, SimulationConfig
from .components import (
    # Exceptions
    SimindError, ValidationError, SimulationError, OutputError,
    # Data classes
    ImageGeometry, EnergyWindow, RotationParameters, RotationDirection,
    # Managers and processors
    ImageValidator, GeometryManager, AcquisitionManager,
    DataFileManager, OrbitFileManager, SimindExecutor, OutputProcessor,
    # Constants
    SIMIND_VOXEL_UNIT_CONVERSION
)


class SimindSimulator:
    """
    Simplified SIMIND simulator that orchestrates the simulation process.
    
    This class now focuses on high-level coordination rather than low-level details.
    Individual responsibilities are delegated to specialized components.
    
    The simulator can be initialized with:
    - An existing .smc template file path
    - A SimulationConfig object directly
    - A YAML configuration file path
    """
    
    def __init__(self, config_source: Union[str, SimulationConfig], output_dir: str, 
                 output_prefix: str = "output", photon_multiplier: int = 1):
        """
        Initialize the simulator with flexible configuration source.
        
        Args:
            config_source: Can be:
                - String path to .smc template file
                - String path to .yaml/.yml configuration file  
                - SimulationConfig object directly
            output_dir: Directory for simulation outputs
            output_prefix: Prefix for output files
            photon_multiplier: Photon multiplier for NN runtime switch
        """
        self.logger = logging.getLogger(__name__)
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_prefix = output_prefix
        
        # Initialize configuration based on source type
        self.config, self.template_path = self._initialize_config(config_source)
        
        # Initialize runtime switches
        self.runtime_switches = RuntimeSwitches()
        self.runtime_switches.set_switch("NN", photon_multiplier)
        
        # Initialize components
        self.converter = SimindToStirConverter()
        self.geometry_manager = GeometryManager(self.config)
        self.acquisition_manager = AcquisitionManager(self.config, self.runtime_switches)
        self.file_manager = DataFileManager(self.output_dir)
        self.orbit_manager = OrbitFileManager(self.output_dir)
        self.executor = SimindExecutor()
        self.output_processor = OutputProcessor(self.converter, self.output_dir)
        
        # Set up for voxelised phantom simulation
        self._configure_voxelised_phantom()
        
        # Simulation state
        self.source: Optional[ImageData] = None
        self.mu_map: Optional[ImageData] = None
        self.template_sinogram: Optional[AcquisitionData] = None
        self.energy_windows: List[EnergyWindow] = []
        self.rotation_params: Optional[RotationParameters] = None
        self.non_circular_orbit = False
        self.orbit_radii: List[float] = []
        
        # Results
        self._outputs: Optional[Dict[str, AcquisitionData]] = None
    
    def _initialize_config(self, config_source: Union[str, SimulationConfig]) -> tuple[SimulationConfig, Optional[Path]]:
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
                raise FileNotFoundError(f"Configuration file not found: {config_source}")
            
            if config_path.suffix.lower() == '.smc':
                # SMC template file (original behavior)
                self.logger.info(f"Loading SMC template file: {config_path}")
                return SimulationConfig(str(config_path)), config_path
                
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                # YAML configuration file
                self.logger.info(f"Loading YAML configuration file: {config_path}")
                return self._load_config_from_yaml(config_path), config_path
                
            else:
                raise ValueError(f"Unsupported configuration file type: {config_path.suffix}")
        
        else:
            raise TypeError(f"config_source must be string path or SimulationConfig object, got {type(config_source)}")
    
    def _load_config_from_yaml(self, yaml_path: Path) -> SimulationConfig:
        """
        Load configuration from YAML file.
        
        Expected YAML structure:
        ```yaml
        template_smc: path/to/template.smc  # Optional base template
        config_values:
          1: some_value
          2: another_value
        config_flags:
          5: true
          11: false
        data_files:
          5: path/to/attenuation.dat
          6: path/to/source.dat
        ```
        """
        try:
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Create base config from template if specified
            if 'template_smc' in yaml_config:
                template_path = yaml_path.parent / yaml_config['template_smc']
                if not template_path.exists():
                    raise FileNotFoundError(f"Template SMC file not found: {template_path}")
                config = SimulationConfig(str(template_path))
                self.logger.info(f"Base configuration loaded from template: {template_path}")
            else:
                # Create empty config (would need a default constructor)
                config = SimulationConfig()
                self.logger.info("Creating configuration from YAML without template")
            
            # Apply config values
            if 'config_values' in yaml_config:
                for index, value in yaml_config['config_values'].items():
                    config.set_value(int(index), value)
                    self.logger.debug(f"Set config value {index} = {value}")
            
            # Apply config flags
            if 'config_flags' in yaml_config:
                for flag, value in yaml_config['config_flags'].items():
                    config.set_flag(int(flag), bool(value))
                    self.logger.debug(f"Set config flag {flag} = {value}")
            
            # Apply data files
            if 'data_files' in yaml_config:
                for index, filepath in yaml_config['data_files'].items():
                    # Resolve path relative to YAML file location
                    if not Path(filepath).is_absolute():
                        filepath = yaml_path.parent / filepath
                    config.set_data_file(int(index), str(filepath))
                    self.logger.debug(f"Set data file {index} = {filepath}")
            
            # Apply any other specific configurations
            if 'photon_energy' in yaml_config:
                config.set_value('photon_energy', yaml_config['photon_energy'])
            
            return config
            
        except Exception as e:
            raise ValueError(f"Failed to load YAML configuration from {yaml_path}: {e}")
    
    def _configure_voxelised_phantom(self) -> None:
        """Configure settings for voxelised phantom simulation."""
        self.config.set_flag(5, True)   # SPECT study
        self.config.set_value(15, -1)   # source type
        self.config.set_value(14, -1)   # phantom type
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
            'config_values': self.config.get_all_values(),
            'config_flags': self.config.get_all_flags(),
            'data_files': self.config.get_all_data_files(),
        }
        
        if hasattr(self.config, 'get_photon_energy'):
            yaml_data['photon_energy'] = self.config.get_photon_energy()
        
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Configuration saved to YAML: {yaml_path}")
    
    # =============================================================================
    # INPUT CONFIGURATION METHODS (UNCHANGED)
    # =============================================================================
    
    def set_source(self, source: Union[str, ImageData]) -> None:
        """Set the source image."""
        if isinstance(source, str):
            self.source = ImageData(source)
        elif isinstance(source, ImageData):
            self.source = source
        else:
            raise TypeError("source must be a string path or ImageData object")
        
        # Validate and configure geometry
        ImageValidator.validate_square_pixels(self.source)
        geometry = ImageGeometry.from_image(self.source)
        self.geometry_manager.configure_source_geometry(geometry)
        
        self.logger.info(f"Source configured: {geometry.dim_x}×{geometry.dim_y}×{geometry.dim_z}")
    
    def set_mu_map(self, mu_map: Union[str, ImageData]) -> None:
        """Set the attenuation map."""
        if isinstance(mu_map, str):
            self.mu_map = ImageData(mu_map)
        elif isinstance(mu_map, ImageData):
            self.mu_map = mu_map
        else:
            raise TypeError("mu_map must be a string path or ImageData object")
        
        # Validate and configure geometry
        ImageValidator.validate_square_pixels(self.mu_map)
        geometry = ImageGeometry.from_image(self.mu_map)
        self.geometry_manager.configure_attenuation_geometry(geometry)
        
        self.logger.info(f"Attenuation map configured: {geometry.dim_x}×{geometry.dim_y}×{geometry.dim_z}")
    
    def set_energy_windows(self, lower_bounds: Union[float, List[float]], 
                          upper_bounds: Union[float, List[float]],
                          scatter_orders: Union[int, List[int]]) -> None:
        """Set energy windows for the simulation."""
        # Convert single values to lists
        if not isinstance(lower_bounds, list):
            lower_bounds = [lower_bounds]
        if not isinstance(upper_bounds, list):
            upper_bounds = [upper_bounds]
        if not isinstance(scatter_orders, list):
            scatter_orders = [scatter_orders]
        
        # Create EnergyWindow objects
        self.energy_windows = [
            EnergyWindow(lb, ub, so, i+1) 
            for i, (lb, ub, so) in enumerate(zip(lower_bounds, upper_bounds, scatter_orders))
        ]
        
        self.logger.info(f"Configured {len(self.energy_windows)} energy windows")
    
    def set_template_sinogram(self, template_sinogram: Union[str, AcquisitionData]) -> None:
        """Set template sinogram and extract acquisition parameters."""
        if isinstance(template_sinogram, str):
            self.template_sinogram = AcquisitionData(template_sinogram)
        elif isinstance(template_sinogram, AcquisitionData):
            self.template_sinogram = template_sinogram.clone()
        else:
            raise TypeError("template_sinogram must be a string path or AcquisitionData object")
        
        # Extract parameters from template
        attributes = extract_attributes_from_stir(self.template_sinogram)
        
        # Set up rotation parameters
        direction = RotationDirection.CCW if attributes["direction_of_rotation"].lower() == "ccw" else RotationDirection.CW
        self.rotation_params = RotationParameters(
            direction=direction,
            rotation_angle=attributes["extent_of_rotation"],
            start_angle=attributes["start_angle"],
            num_projections=attributes["number_of_projections"]
        )
        
        # Configure acquisition
        detector_distance = attributes["height_to_detector_surface"]
        self.acquisition_manager.configure_rotation(self.rotation_params, detector_distance)
        
        # Handle non-circular orbits
        if attributes.get("orbit") == "non-circular" and "radii" in attributes:
            self.non_circular_orbit = True
            self.orbit_radii = attributes["radii"]
            self.logger.info("Non-circular orbit detected")
        
        self.logger.info("Template sinogram configured")
    
    def set_rotation_parameters(self, direction: str, rotation_angle: float, 
                               start_angle: float, num_projections: int,
                               detector_distance: float) -> None:
        """Manually set rotation parameters."""
        direction_enum = RotationDirection.CCW if direction.lower() == "ccw" else RotationDirection.CW
        
        self.rotation_params = RotationParameters(
            direction=direction_enum,
            rotation_angle=rotation_angle,
            start_angle=start_angle,
            num_projections=num_projections
        )
        
        self.acquisition_manager.configure_rotation(self.rotation_params, detector_distance)
        self.logger.info(f"Rotation configured: {direction} {rotation_angle}° from {start_angle}°")
    
    # =============================================================================
    # CONFIGURATION METHODS (UNCHANGED)
    # =============================================================================
    
    def add_config_value(self, index: int, value) -> None:
        """Add a configuration index value."""
        self.config.set_value(index, value)
    
    def add_config_flag(self, flag: int, value: bool) -> None:
        """Add a configuration flag."""
        self.config.set_flag(flag, value)
    
    def add_runtime_switch(self, switch: str, value) -> None:
        """Add a runtime switch."""
        self.runtime_switches.set_switch(switch, value)
    
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
        if not self.energy_windows:
            raise ValidationError("Energy windows must be set before simulation")
        
        if self.source is None or self.mu_map is None:
            raise ValidationError("Both source and mu_map must be set")
        
        # Check image compatibility
        ImageValidator.validate_compatibility(self.source, self.mu_map)
    
    def _prepare_simulation(self) -> None:
        """Prepare all files and configuration for simulation."""
        # Configure energy windows
        self.acquisition_manager.configure_energy_windows(
            self.energy_windows, str(self.output_dir / self.output_prefix)
        )
        
        # Prepare data files
        source_file = self.file_manager.prepare_source_file(self.source, self.output_prefix)
        attenuation_file = self.file_manager.prepare_attenuation_file(
            self.mu_map, self.output_prefix, 
            self.config.get_flag(11),  # use attenuation
            self.config.get_value("photon_energy"),
            self.template_path.parent if self.template_path else Path.cwd()
        )
        
        # Set data files in configuration
        self.config.set_data_file(6, source_file)      # source file
        self.config.set_data_file(5, attenuation_file) # attenuation file

        # need to add PX runtime switch with source voxel size
        if self.source:
            voxel_size = self.source.voxel_sizes()[-1]
            self.runtime_switches.set_switch("PX", voxel_size / SIMIND_VOXEL_UNIT_CONVERSION)
            self.logger.info(f"Set PX runtime switch to {voxel_size} cm")

        
        # CRITICAL: Always save configuration as .smc file for SIMIND execution
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
            if self.non_circular_orbit and self.orbit_radii:
                center_of_rotation = self.source.dimensions()[1] / 2 if self.source else None
                orbit_file = self.orbit_manager.write_orbit_file(
                    self.orbit_radii, self.output_prefix, center_of_rotation
                )
            
            # Execute simulation
            self.executor.run_simulation(
                self.output_prefix, 
                orbit_file,
                self.runtime_switches.switches
            )
            
        finally:
            os.chdir(original_cwd)
    
    # =============================================================================
    # OUTPUT METHODS (UNCHANGED)
    # =============================================================================
    
    def get_outputs(self) -> Dict[str, AcquisitionData]:
        """Get all simulation outputs."""
        if self._outputs is None:
            self._outputs = self.output_processor.process_outputs(
                self.output_prefix, self.template_sinogram, self.source
            )
        return self._outputs
    
    def get_total_output(self, window: int = 1) -> AcquisitionData:
        """Get total output for specified window."""
        outputs = self.get_outputs()
        key = f"tot_w{window}"
        if key not in outputs:
            raise OutputError(f"Total output for window {window} not found")
        return outputs[key]
    
    def get_scatter_output(self, window: int = 1) -> AcquisitionData:
        """Get scatter output for specified window."""
        outputs = self.get_outputs()
        key = f"sca_w{window}"
        if key not in outputs:
            raise OutputError(f"Scatter output for window {window} not found")
        return outputs[key]
    
    def get_primary_output(self, window: int = 1) -> AcquisitionData:
        """Get primary output for specified window."""
        outputs = self.get_outputs()
        key = f"pri_w{window}"
        if key not in outputs:
            raise OutputError(f"Primary output for window {window} not found")
        return outputs[key]
    
    def get_air_output(self, window: int = 1) -> AcquisitionData:
        """Get air output for specified window."""
        outputs = self.get_outputs()
        key = f"air_w{window}"
        if key not in outputs:
            raise OutputError(f"Air output for window {window} not found")
        return outputs[key]


# =============================================================================
# CONVENIENCE FACTORY FUNCTIONS
# =============================================================================

def create_simulator_from_template(template_smc_path: str, output_dir: str, 
                                  template_sinogram: Union[str, AcquisitionData],
                                  source: Union[str, ImageData],
                                  mu_map: Union[str, ImageData],
                                  energy_windows: Dict,
                                  **kwargs) -> SimindSimulator:
    """Factory function to create a fully configured simulator from SMC template."""
    
    simulator = SimindSimulator(
        template_smc_path, output_dir, 
        output_prefix=kwargs.get("output_prefix", "output"),
        photon_multiplier=kwargs.get("photon_multiplier", 1)
    )
    
    # Set all inputs
    simulator.set_source(source)
    simulator.set_mu_map(mu_map)
    simulator.set_template_sinogram(template_sinogram)
    simulator.set_energy_windows(**energy_windows)
    
    return simulator


def create_simulator_from_yaml(yaml_config_path: str, output_dir: str,
                                template_sinogram: Union[str, AcquisitionData],
                                source: Union[str, ImageData],
                                mu_map: Union[str, ImageData],
                                energy_windows: Dict,
                                **kwargs) -> SimindSimulator:
    """Factory function to create a simulator from YAML configuration."""
    
    simulator = SimindSimulator(
        yaml_config_path, output_dir,
        output_prefix=kwargs.get("output_prefix", "output"),
        photon_multiplier=kwargs.get("photon_multiplier", 1)
    )

    # Set all inputs
    simulator.set_source(source)
    simulator.set_mu_map(mu_map)
    simulator.set_template_sinogram(template_sinogram)
    simulator.set_energy_windows(**energy_windows)

    return simindulator


def create_simulator_from_config(config: SimulationConfig, output_dir: str,
                                    template_sinogram: Union[str, AcquisitionData],
                                    source: Union[str, ImageData],
                                    mu_map: Union[str, ImageData],
                                    energy_windows: Dict,
                                    **kwargs) -> SimindSimulator:
        """Factory function to create a simulator from SimulationConfig object."""
        
        simulator = SimindSimulator(
            config, output_dir,
            output_prefix=kwargs.get("output_prefix", "output"),
            photon_multiplier=kwargs.get("photon_multiplier", 1)
        )
    
        # Set all inputs
        simulator.set_source(source)
        simulator.set_mu_map(mu_map)
        simulator.set_template_sinogram(template_sinogram)
        simulator.set_energy_windows(**energy_windows)
    
        return simulator
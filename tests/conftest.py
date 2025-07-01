"""
conftest.py - pytest configuration and shared fixtures for SIRF-SIMIND-Connection tests

This module provides shared fixtures and utilities for testing the package.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import yaml
import os

# Try to import SIRF and package modules
try:
    from sirf.STIR import ImageData, AcquisitionData
    SIRF_AVAILABLE = True
except ImportError:
    SIRF_AVAILABLE = False

try:
    import sirf_simind_connection
    PACKAGE_AVAILABLE = True
except ImportError:
    PACKAGE_AVAILABLE = False


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "requires_simind: mark test as requiring SIMIND installation"
    )
    config.addinivalue_line(
        "markers", "requires_data: mark test as requiring test data files"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid or hasattr(item.function, "_slow_test"):
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.nodeid or "test_complete" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark tests requiring SIMIND
        if "simind" in item.nodeid.lower() or "simulation" in item.nodeid:
            item.add_marker(pytest.mark.requires_simind)


def pytest_runtest_setup(item):
    """Skip tests based on availability of dependencies."""
    # Skip SIRF tests if SIRF not available
    if not SIRF_AVAILABLE and any(mark.name in ['requires_sirf', 'integration'] 
                                  for mark in item.iter_markers()):
        pytest.skip("SIRF not available")
    
    # Skip package tests if package not available
    if not PACKAGE_AVAILABLE:
        pytest.skip("sirf_simind_connection package not available")
    
    # Skip SIMIND tests if SIMIND not installed (unless mocked)
    if item.get_closest_marker("requires_simind") and not _simind_available():
        if not item.get_closest_marker("mock"):
            pytest.skip("SIMIND not available")


def _simind_available():
    """Check if SIMIND is available in the system."""
    import shutil
    return shutil.which("simind") is not None


# Fixtures for temporary directories and files
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    fd, temp_path = tempfile.mkstemp()
    os.close(fd)
    temp_path = Path(temp_path)
    yield temp_path
    temp_path.unlink(missing_ok=True)


# Configuration fixtures
@pytest.fixture
def sample_smc_content():
    """Sample SIMIND control file content."""
    return """
! SIMIND Control File - Test Configuration
TITLE Test Simulation for SIRF-SIMIND-Connection
PHOTONS 100000
SPECTRUM MONO 140.0
DETECTOR NaI 0.95
MATRIX 64 64
PIXEL_SIZE 0.4 0.4
PROJECTIONS 60
ORBIT CIRCULAR
START_ANGLE 0.0
STOP_ANGLE 360.0
! Energy windows
ENERGY_WINDOW 1 126.0 154.0
ENERGY_WINDOW 2 100.0 126.0
ENERGY_WINDOW 3 154.0 180.0
! Output settings
OUTPUT_FORMAT STIR
SAVE_SCATTER_ESTIMATES 1
"""


@pytest.fixture
def sample_yaml_config():
    """Sample YAML configuration for scanner setup."""
    return {
        'scanner': {
            'name': 'test_scanner',
            'type': 'dual_head_spect',
            'detector': {
                'material': 'NaI',
                'thickness': 0.95,
                'crystal_x': 40.0,
                'crystal_y': 30.0,
                'pixel_size': [0.4, 0.4],
                'matrix_size': [128, 128]
            },
            'collimator': {
                'type': 'LEHR',
                'hole_diameter': 0.11,
                'septal_thickness': 0.016,
                'hole_length': 2.4
            },
            'gantry': {
                'radius': 15.0,
                'angular_range': 360.0,
                'num_projections': 120
            }
        },
        'simulation': {
            'number_of_photons': 1000000,
            'random_seed': 12345,
            'detector_binning': [128, 128],
            'voxel_size': [0.4, 0.4, 0.4],
            'output_format': 'STIR'
        },
        'physics': {
            'scatter_modeling': True,
            'attenuation_correction': True,
            'detector_response': True
        }
    }


@pytest.fixture
def smc_file(temp_dir, sample_smc_content):
    """Create a temporary SMC file."""
    smc_path = temp_dir / "test_config.smc"
    smc_path.write_text(sample_smc_content)
    return smc_path


@pytest.fixture
def yaml_config_file(temp_dir, sample_yaml_config):
    """Create a temporary YAML configuration file."""
    yaml_path = temp_dir / "test_config.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(sample_yaml_config, f)
    return yaml_path


# Image data fixtures
@pytest.fixture
def test_phantom_3d():
    """Create a simple 3D test phantom."""
    if not SIRF_AVAILABLE:
        pytest.skip("SIRF not available")
    
    # Create a simple 3D phantom
    dimensions = (64, 64, 32)
    phantom_array = np.zeros(dimensions)
    
    # Create a cylindrical object
    x, y, z = np.meshgrid(
        np.linspace(-dimensions[0]//2, dimensions[0]//2, dimensions[0]),
        np.linspace(-dimensions[1]//2, dimensions[1]//2, dimensions[1]),
        np.linspace(-dimensions[2]//2, dimensions[2]//2, dimensions[2]),
        indexing='ij'
    )
    
    # Main cylinder
    radius = 20
    cylinder = (x**2 + y**2) <= radius**2
    phantom_array[cylinder] = 1.0
    
    # Hot spots
    for center in [(10, 0, 0), (-10, 0, 0), (0, 15, 0), (0, -15, 0)]:
        sphere = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) <= 25
        phantom_array[sphere] = 5.0
    
    # Convert to SIRF ImageData
    phantom = ImageData()
    phantom.initialise(dimensions)
    phantom.fill(phantom_array)
    phantom.set_voxel_spacing((0.4, 0.4, 0.4))
    
    return phantom


@pytest.fixture
def test_attenuation_map_3d():
    """Create a simple 3D attenuation map."""
    if not SIRF_AVAILABLE:
        pytest.skip("SIRF not available")
    
    dimensions = (64, 64, 32)
    mu_array = np.zeros(dimensions)
    
    x, y, z = np.meshgrid(
        np.linspace(-dimensions[0]//2, dimensions[0]//2, dimensions[0]),
        np.linspace(-dimensions[1]//2, dimensions[1]//2, dimensions[1]),
        np.linspace(-dimensions[2]//2, dimensions[2]//2, dimensions[2]),
        indexing='ij'
    )
    
    # Soft tissue background
    radius = 20
    tissue = (x**2 + y**2) <= radius**2
    mu_array[tissue] = 0.15  # cm^-1 for soft tissue at 140 keV
    
    # Bone inserts
    bone_centers = [(0, 0, 0)]
    for center in bone_centers:
        bone = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) <= 36
        mu_array[bone] = 0.4  # cm^-1 for bone at 140 keV
    
    # Convert to SIRF ImageData
    mu_map = ImageData()
    mu_map.initialise(dimensions)
    mu_map.fill(mu_array)
    mu_map.set_voxel_spacing((0.4, 0.4, 0.4))
    
    return mu_map


@pytest.fixture
def mock_projection_data():
    """Create mock projection data for testing."""
    if not SIRF_AVAILABLE:
        pytest.skip("SIRF not available")
    
    # Create mock projection data with realistic dimensions
    num_projections = 60
    detector_size = (128, 128)
    
    # Generate synthetic projection data
    proj_array = np.random.poisson(1000, (num_projections, *detector_size))
    
    # Add some structure (sinusoidal pattern simulating object projection)
    for i in range(num_projections):
        angle = i * 2 * np.pi / num_projections
        for j in range(detector_size[0]):
            proj_array[i, j, :] += np.random.poisson(
                500 * np.exp(-0.1 * (j - detector_size[0]//2)**2) * 
                (1 + 0.5 * np.sin(angle + j * 0.1))
            )
    
    # Create AcquisitionData (this would need proper initialization in real use)
    # For testing, we'll return the array and let individual tests handle SIRF objects
    return proj_array


# Mock fixtures for SIMIND simulator
@pytest.fixture
def mock_simind_simulator(temp_dir, test_phantom_3d, test_attenuation_map_3d):
    """Create a mock SIMIND simulator for testing."""
    if not PACKAGE_AVAILABLE:
        pytest.skip("Package not available")
    
    from unittest.mock import Mock, patch
    
    with patch('sirf_simind_connection.SimindSimulator') as MockSimulator:
        # Create mock instance
        mock_instance = Mock()
        MockSimulator.return_value = mock_instance
        
        # Configure mock methods
        mock_instance.set_windows = Mock()
        mock_instance.run_simulation = Mock()
        mock_instance.get_output_total = Mock()
        mock_instance.get_output_scatter = Mock()
        
        # Set up mock return values for outputs
        mock_total_data = Mock()
        mock_total_data.as_array.return_value = np.random.poisson(1000, (60, 128, 128))
        mock_instance.get_output_total.return_value = mock_total_data
        
        mock_scatter_data = Mock()
        mock_scatter_data.as_array.return_value = np.random.poisson(200, (60, 128, 128))
        mock_instance.get_output_scatter.return_value = mock_scatter_data
        
        yield mock_instance


# Utility fixtures for testing specific scenarios
@pytest.fixture
def energy_windows_tc99m():
    """Standard energy windows for Tc-99m imaging."""
    return {
        'photopeak': {'lower': 126, 'upper': 154},    # 140 ± 10%
        'scatter_low': {'lower': 100, 'upper': 126},   # Lower scatter
        'scatter_high': {'lower': 154, 'upper': 180},  # Upper scatter
        'triple_energy_window': {
            'lower': {'lower': 100, 'upper': 126},
            'photopeak': {'lower': 126, 'upper': 154},
            'upper': {'lower': 154, 'upper': 180}
        }
    }


@pytest.fixture
def reconstruction_parameters():
    """Standard reconstruction parameters for testing."""
    return {
        'num_subsets': 8,
        'num_iterations': 10,
        'algorithm': 'OSMAPOSL',
        'prior_weight': 0.1,
        'filter_fwhm': 5.0,
        'attenuation_correction': True,
        'scatter_correction': True
    }


# Performance testing fixtures
@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time
    
    return Timer()


# Error injection fixtures for robustness testing
@pytest.fixture
def error_injection():
    """Fixture for injecting controlled errors during testing."""
    
    class ErrorInjector:
        def __init__(self):
            self.errors = {}
        
        def inject_error(self, function_name, error_type, error_message=None):
            """Inject an error for a specific function."""
            self.errors[function_name] = (error_type, error_message)
        
        def should_raise_error(self, function_name):
            """Check if an error should be raised for a function."""
            return function_name in self.errors
        
        def get_error(self, function_name):
            """Get the error to raise for a function."""
            if function_name in self.errors:
                error_type, message = self.errors[function_name]
                return error_type(message or f"Injected error in {function_name}")
            return None
        
        def clear_errors(self):
            """Clear all injected errors."""
            self.errors.clear()
    
    return ErrorInjector()


# Data validation fixtures
@pytest.fixture
def data_validator():
    """Fixture for validating test data integrity."""
    
    class DataValidator:
        def validate_image_data(self, image_data):
            """Validate SIRF ImageData object."""
            if not SIRF_AVAILABLE:
                return True
            
            assert hasattr(image_data, 'as_array'), "Image data must have as_array method"
            array = image_data.as_array()
            assert array.ndim in [2, 3], "Image must be 2D or 3D"
            assert array.size > 0, "Image must not be empty"
            assert not np.isnan(array).any(), "Image must not contain NaN values"
            assert not np.isinf(array).any(), "Image must not contain infinite values"
            return True
        
        def validate_projection_data(self, proj_data):
            """Validate projection data."""
            if hasattr(proj_data, 'as_array'):
                array = proj_data.as_array()
            else:
                array = proj_data
            
            assert array.ndim == 3, "Projection data must be 3D (proj, y, x)"
            assert array.shape[0] > 0, "Must have at least one projection"
            assert array.shape[1] > 0 and array.shape[2] > 0, "Detector dimensions must be positive"
            assert (array >= 0).all(), "Projection data must be non-negative"
            return True
        
        def validate_config_file(self, config_path):
            """Validate configuration file."""
            config_path = Path(config_path)
            assert config_path.exists(), f"Config file does not exist: {config_path}"
            
            if config_path.suffix.lower() == '.yaml':
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                assert isinstance(config, dict), "YAML config must be a dictionary"
            elif config_path.suffix.lower() == '.smc':
                content = config_path.read_text()
                assert len(content.strip()) > 0, "SMC file must not be empty"
            
            return True
    
    return DataValidator()


# Skip markers for conditional tests
skip_if_no_sirf = pytest.mark.skipif(not SIRF_AVAILABLE, reason="SIRF not available")
skip_if_no_package = pytest.mark.skipif(not PACKAGE_AVAILABLE, reason="Package not available")
skip_if_no_simind = pytest.mark.skipif(not _simind_available(), reason="SIMIND not available")
"""
Test suite for SIRF-SIMIND-Connection package.

This module contains comprehensive tests for all major functionality
of the SIRF-SIMIND-Connection package.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import yaml

# Import the package modules
try:
    from sirf_simind_connection import SimindSimulator, SimulationConfig
    from sirf_simind_connection.dicom_converter import DicomConverter
    from sirf_simind_connection.utils import density_conversion
    from sirf.STIR import ImageData, AcquisitionData
    SIRF_AVAILABLE = True
except ImportError:
    SIRF_AVAILABLE = False


class TestSimulationConfig:
    """Test the SimulationConfig class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_yaml_config(self):
        """Sample YAML configuration for testing."""
        return {
            'scanner': {
                'name': 'test_scanner',
                'detector': {
                    'material': 'NaI',
                    'thickness': 0.95,
                    'crystal_x': 59.8,
                    'crystal_y': 44.8
                },
                'collimator': {
                    'type': 'LEHR',
                    'hole_diameter': 0.11,
                    'septal_thickness': 0.016
                }
            },
            'simulation': {
                'number_of_photons': 1000000,
                'detector_binning': [128, 128],
                'voxel_size': [0.4, 0.4, 0.4]
            }
        }
    
    @pytest.fixture
    def sample_smc_content(self):
        """Sample SMC file content for testing."""
        return """
        ! SIMIND Control File
        TITLE Test Simulation
        PHOTONS 1000000
        SPECTRUM MONO 140.0
        DETECTOR NaI 0.95
        MATRIX 128 128
        PIXEL_SIZE 0.4 0.4
        """
    
    def test_config_creation(self, temp_dir, sample_smc_content):
        """Test creating a SimulationConfig object."""
        if not SIRF_AVAILABLE:
            pytest.skip("SIRF not available")
        
        smc_file = temp_dir / "test.smc"
        smc_file.write_text(sample_smc_content)
        
        config = SimulationConfig(str(smc_file))
        assert config is not None
    
    def test_yaml_import(self, temp_dir, sample_yaml_config, sample_smc_content):
        """Test importing YAML configuration."""
        if not SIRF_AVAILABLE:
            pytest.skip("SIRF not available")
        
        # Create temporary files
        smc_file = temp_dir / "test.smc"
        yaml_file = temp_dir / "test.yaml"
        
        smc_file.write_text(sample_smc_content)
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_yaml_config, f)
        
        config = SimulationConfig(str(smc_file))
        config.import_yaml(str(yaml_file))
    
    def test_save_file(self, temp_dir, sample_smc_content):
        """Test saving configuration to file."""
        if not SIRF_AVAILABLE:
            pytest.skip("SIRF not available")
        
        smc_file = temp_dir / "test.smc"
        smc_file.write_text(sample_smc_content)
        
        config = SimulationConfig(str(smc_file))
        output_file = config.save_file(str(temp_dir / "output.smc"))
        
        assert Path(output_file).exists()


class TestSimindSimulator:
    """Test the SimindSimulator class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_image_data(self):
        """Create mock ImageData objects."""
        if not SIRF_AVAILABLE:
            pytest.skip("SIRF not available")
        
        # Create mock source image
        source = Mock(spec=ImageData)
        source.dimensions.return_value = (64, 64, 64)
        source.voxel_sizes.return_value = (0.4, 0.4, 0.4)
        source.as_array.return_value = np.random.rand(64, 64, 64)
        
        # Create mock attenuation map
        mu_map = Mock(spec=ImageData)
        mu_map.dimensions.return_value = (64, 64, 64)
        mu_map.voxel_sizes.return_value = (0.4, 0.4, 0.4)
        mu_map.as_array.return_value = np.random.rand(64, 64, 64) * 0.1
        
        return source, mu_map
    
    def test_simulator_initialization(self, temp_dir, mock_image_data):
        """Test initializing the SimindSimulator."""
        if not SIRF_AVAILABLE:
            pytest.skip("SIRF not available")
        
        source, mu_map = mock_image_data
        
        # Create a mock SMC file
        smc_content = "TITLE Test\nPHOTONS 1000\n"
        smc_file = temp_dir / "test.smc"
        smc_file.write_text(smc_content)
        
        simulator = SimindSimulator(
            template_smc_file_path=str(smc_file),
            output_dir=str(temp_dir),
            source=source,
            mu_map=mu_map
        )
        
        assert simulator is not None
        assert simulator.output_dir == str(temp_dir)
    
    def test_set_windows(self, temp_dir, mock_image_data):
        """Test setting energy windows."""
        if not SIRF_AVAILABLE:
            pytest.skip("SIRF not available")
        
        source, mu_map = mock_image_data
        
        smc_content = "TITLE Test\nPHOTONS 1000\n"
        smc_file = temp_dir / "test.smc"
        smc_file.write_text(smc_content)
        
        simulator = SimindSimulator(
            template_smc_file_path=str(smc_file),
            output_dir=str(temp_dir),
            source=source,
            mu_map=mu_map
        )
        
        # Test setting multiple windows
        simulator.set_windows(
            lower_bounds=[126, 100, 154],
            upper_bounds=[154, 126, 180],
            scatter_orders=[0, 1, 0]
        )
        
        assert len(simulator.windows) == 3
    
    @patch('subprocess.run')
    def test_run_simulation(self, mock_subprocess, temp_dir, mock_image_data):
        """Test running the simulation."""
        if not SIRF_AVAILABLE:
            pytest.skip("SIRF not available")
        
        source, mu_map = mock_image_data
        
        smc_content = "TITLE Test\nPHOTONS 1000\n"
        smc_file = temp_dir / "test.smc"
        smc_file.write_text(smc_content)
        
        # Mock successful subprocess run
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Simulation completed successfully"
        
        simulator = SimindSimulator(
            template_smc_file_path=str(smc_file),
            output_dir=str(temp_dir),
            source=source,
            mu_map=mu_map
        )
        
        simulator.set_windows(
            lower_bounds=[126],
            upper_bounds=[154],
            scatter_orders=[0]
        )
        
        # Create mock output files
        (temp_dir / "tot001.h").write_text("dummy")
        (temp_dir / "tot001.a").write_text("dummy binary data")
        
        simulator.run_simulation()
        
        # Verify subprocess was called
        mock_subprocess.assert_called()


class TestDicomConverter:
    """Test the DICOM conversion functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_dicom_to_stir_conversion(self, temp_dir):
        """Test DICOM to STIR conversion."""
        if not SIRF_AVAILABLE:
            pytest.skip("SIRF not available")
        
        # This would require actual DICOM files for full testing
        # For now, test the interface
        converter = DicomConverter()
        
        # Test with mock data
        with pytest.raises((FileNotFoundError, ImportError)):
            converter.convert_to_stir("nonexistent.dcm", str(temp_dir))


class TestDensityConversion:
    """Test density and attenuation conversion utilities."""
    
    def test_hounsfield_to_density(self):
        """Test Hounsfield units to density conversion."""
        if not SIRF_AVAILABLE:
            pytest.skip("SIRF not available")
        
        # Standard conversion: water = 0 HU = 1.0 g/cm³
        hu_values = np.array([0, 1000, -1000])  # water, bone, air
        densities = density_conversion.hounsfield_to_density(hu_values)
        
        assert densities[0] == pytest.approx(1.0, rel=1e-3)  # water
        assert densities[1] > 1.0  # bone (denser than water)
        assert densities[2] < 1.0  # air (less dense than water)
    
    def test_density_to_attenuation(self):
        """Test density to attenuation coefficient conversion."""
        if not SIRF_AVAILABLE:
            pytest.skip("SIRF not available")
        
        densities = np.array([1.0, 1.8, 0.001])  # water, bone, air
        energy_kev = 140  # Tc-99m photopeak
        
        mu_values = density_conversion.density_to_attenuation(densities, energy_kev)
        
        assert len(mu_values) == len(densities)
        assert all(mu_values >= 0)  # attenuation coefficients should be positive
        assert mu_values[1] > mu_values[0]  # bone > water
        assert mu_values[2] < mu_values[0]  # air < water


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_smc_file(self):
        """Test handling of invalid SMC files."""
        if not SIRF_AVAILABLE:
            pytest.skip("SIRF not available")
        
        with pytest.raises(FileNotFoundError):
            SimulationConfig("nonexistent.smc")
    
    def test_invalid_energy_windows(self, temp_dir):
        """Test handling of invalid energy window specifications."""
        if not SIRF_AVAILABLE:
            pytest.skip("SIRF not available")
        
        smc_content = "TITLE Test\nPHOTONS 1000\n"
        smc_file = temp_dir / "test.smc"
        smc_file.write_text(smc_content)
        
        source = Mock(spec=ImageData)
        mu_map = Mock(spec=ImageData)
        
        simulator = SimindSimulator(
            template_smc_file_path=str(smc_file),
            output_dir=str(temp_dir),
            source=source,
            mu_map=mu_map
        )
        
        # Test mismatched window bounds
        with pytest.raises(ValueError):
            simulator.set_windows(
                lower_bounds=[126, 100],
                upper_bounds=[154],  # Mismatched length
                scatter_orders=[0, 1]
            )
    
    def test_missing_simind_executable(self):
        """Test handling when SIMIND executable is not found."""
        if not SIRF_AVAILABLE:
            pytest.skip("SIRF not available")
        
        # This test would check for proper error handling when SIMIND is not installed
        # The actual implementation would depend on how the package checks for SIMIND
        pass


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_complete_simulation_workflow(self, temp_dir):
        """Test a complete simulation workflow."""
        if not SIRF_AVAILABLE:
            pytest.skip("SIRF not available")
        
        # This test would run through the complete workflow:
        # 1. Load configuration
        # 2. Set up simulator
        # 3. Run simulation
        # 4. Retrieve results
        # For now, this is a placeholder for the structure
        pass
    
    def test_multi_window_reconstruction(self, temp_dir):
        """Test multi-window simulation and reconstruction."""
        if not SIRF_AVAILABLE:
            pytest.skip("SIRF not available")
        
        # This test would verify:
        # 1. Multi-window simulation setup
        # 2. Scatter correction generation
        # 3. Reconstruction with scatter correction
        # 4. Comparison with ground truth
        pass


# Utility functions for test data generation
def create_test_phantom(dimensions=(64, 64, 64), voxel_size=(0.4, 0.4, 0.4)):
    """Create a simple test phantom for simulation testing."""
    if not SIRF_AVAILABLE:
        return None
    
    # Create a simple cylindrical phantom with hot spots
    x, y, z = np.meshgrid(
        np.linspace(-dimensions[0]//2, dimensions[0]//2, dimensions[0]),
        np.linspace(-dimensions[1]//2, dimensions[1]//2, dimensions[1]),
        np.linspace(-dimensions[2]//2, dimensions[2]//2, dimensions[2]),
        indexing='ij'
    )
    
    # Cylindrical background
    radius = min(dimensions[0], dimensions[1]) // 3
    cylinder = (x**2 + y**2) <= radius**2
    
    # Add hot spots
    phantom = np.zeros(dimensions)
    phantom[cylinder] = 1.0
    
    # Add some hot spots
    for i in range(3):
        center_x = np.random.randint(20, dimensions[0]-20)
        center_y = np.random.randint(20, dimensions[1]-20)
        center_z = np.random.randint(20, dimensions[2]-20)
        
        sphere = ((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2) <= 25
        phantom[sphere] = 5.0
    
    return phantom


def create_test_attenuation_map(dimensions=(64, 64, 64)):
    """Create a simple attenuation map for testing."""
    if not SIRF_AVAILABLE:
        return None
    
    # Create a simple attenuation map
    # Background: soft tissue (mu ~ 0.15 cm^-1 at 140 keV)
    # Bone inserts: bone (mu ~ 0.4 cm^-1 at 140 keV)
    
    x, y, z = np.meshgrid(
        np.linspace(-dimensions[0]//2, dimensions[0]//2, dimensions[0]),
        np.linspace(-dimensions[1]//2, dimensions[1]//2, dimensions[1]),
        np.linspace(-dimensions[2]//2, dimensions[2]//2, dimensions[2]),
        indexing='ij'
    )
    
    # Soft tissue background
    mu_map = np.ones(dimensions) * 0.15
    
    # Add bone-like regions
    radius = min(dimensions[0], dimensions[1]) // 4
    bone_region = (x**2 + y**2) <= radius**2
    mu_map[bone_region] = 0.4
    
    return mu_map


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
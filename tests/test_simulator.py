# test_simulator.py - Unit tests for SimindSimulator class
"""
test_simulator.py - Unit tests for SimindSimulator class
"""

import pytest
import numpy as np
import subprocess
from unittest.mock import Mock, patch, MagicMock

try:
    from sirf_simind_connection import SimindSimulator
    from sirf.STIR import ImageData, AcquisitionData
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not available")
class TestSimindSimulator:
    """Unit tests for SimindSimulator class."""
    
    def test_simulator_initialization(self, smc_file, temp_dir, test_phantom_3d, test_attenuation_map_3d):
        """Test simulator initialization with valid inputs."""
        simulator = SimindSimulator(
            template_smc_file_path=str(smc_file),
            output_dir=str(temp_dir),
            source=test_phantom_3d,
            mu_map=test_attenuation_map_3d
        )
        
        assert simulator.output_dir == str(temp_dir)
        assert simulator.source is not None
        assert simulator.mu_map is not None
    
    def test_set_single_window(self, smc_file, temp_dir, test_phantom_3d, test_attenuation_map_3d):
        """Test setting a single energy window."""
        simulator = SimindSimulator(
            template_smc_file_path=str(smc_file),
            output_dir=str(temp_dir),
            source=test_phantom_3d,
            mu_map=test_attenuation_map_3d
        )
        
        simulator.set_windows(
            lower_bounds=[126],
            upper_bounds=[154],
            scatter_orders=[0]
        )
        
        assert len(simulator.windows) == 1
    
    def test_set_multiple_windows(self, smc_file, temp_dir, test_phantom_3d, test_attenuation_map_3d):
        """Test setting multiple energy windows."""
        simulator = SimindSimulator(
            template_smc_file_path=str(smc_file),
            output_dir=str(temp_dir),
            source=test_phantom_3d,
            mu_map=test_attenuation_map_3d
        )
        
        simulator.set_windows(
            lower_bounds=[100, 126, 154],
            upper_bounds=[126, 154, 180],
            scatter_orders=[1, 0, 1]
        )
        
        assert len(simulator.windows) == 3
    
    def test_invalid_window_configuration(self, smc_file, temp_dir, test_phantom_3d, test_attenuation_map_3d):
        """Test invalid energy window configuration."""
        simulator = SimindSimulator(
            template_smc_file_path=str(smc_file),
            output_dir=str(temp_dir),
            source=test_phantom_3d,
            mu_map=test_attenuation_map_3d
        )
        
        # Mismatched array lengths
        with pytest.raises(ValueError):
            simulator.set_windows(
                lower_bounds=[126, 100],
                upper_bounds=[154],  # Different length
                scatter_orders=[0, 1]
            )
    
    @patch('subprocess.run')
    def test_successful_simulation_run(self, mock_subprocess, smc_file, temp_dir, 
                                     test_phantom_3d, test_attenuation_map_3d):
        """Test successful simulation execution."""
        # Mock successful subprocess run
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Simulation completed"
        
        simulator = SimindSimulator(
            template_smc_file_path=str(smc_file),
            output_dir=str(temp_dir),
            source=test_phantom_3d,
            mu_map=test_attenuation_map_3d
        )
        
        simulator.set_windows(
            lower_bounds=[126],
            upper_bounds=[154],
            scatter_orders=[0]
        )
        
        # Create mock output files
        (temp_dir / "tot001.h").write_text("dummy header")
        (temp_dir / "tot001.a").write_bytes(b"dummy binary data")
        
        simulator.run_simulation()
        
        # Verify subprocess was called
        mock_subprocess.assert_called()
    
    @patch('subprocess.run')
    def test_failed_simulation_run(self, mock_subprocess, smc_file, temp_dir,
                                 test_phantom_3d, test_attenuation_map_3d):
        """Test handling of failed simulation."""
        # Mock failed subprocess run
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "SIMIND error"
        
        simulator = SimindSimulator(
            template_smc_file_path=str(smc_file),
            output_dir=str(temp_dir),
            source=test_phantom_3d,
            mu_map=test_attenuation_map_3d
        )
        
        simulator.set_windows(
            lower_bounds=[126],
            upper_bounds=[154],
            scatter_orders=[0]
        )
        
        with pytest.raises(RuntimeError):
            simulator.run_simulation()
    
    def test_get_output_without_simulation(self, smc_file, temp_dir,
                                         test_phantom_3d, test_attenuation_map_3d):
        """Test getting output before running simulation."""
        simulator = SimindSimulator(
            template_smc_file_path=str(smc_file),
            output_dir=str(temp_dir),
            source=test_phantom_3d,
            mu_map=test_attenuation_map_3d
        )
        
        simulator.set_windows(
            lower_bounds=[126],
            upper_bounds=[154],
            scatter_orders=[0]
        )
        
        with pytest.raises(FileNotFoundError):
            simulator.get_output_total(window=1)
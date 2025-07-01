"""
test_simulation_config.py - Unit tests for SimulationConfig class
"""

import pytest
import numpy as np
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

try:
    from sirf_simind_connection import SimulationConfig
    PACKAGE_AVAILABLE = True
except ImportError:
    PACKAGE_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not PACKAGE_AVAILABLE, reason="Package not available")
class TestSimulationConfig:
    """Unit tests for SimulationConfig class."""
    
    def test_config_initialization_with_valid_smc(self, smc_file):
        """Test config initialization with valid SMC file."""
        config = SimulationConfig(str(smc_file))
        assert config is not None
    
    def test_config_initialization_with_invalid_path(self):
        """Test config initialization with invalid file path."""
        with pytest.raises(FileNotFoundError):
            SimulationConfig("nonexistent_file.smc")
    
    def test_yaml_import_valid_config(self, smc_file, yaml_config_file):
        """Test importing valid YAML configuration."""
        config = SimulationConfig(str(smc_file))
        config.import_yaml(str(yaml_config_file))
        # Should not raise any exceptions
    
    def test_yaml_import_invalid_file(self, smc_file):
        """Test importing invalid YAML file."""
        config = SimulationConfig(str(smc_file))
        with pytest.raises(FileNotFoundError):
            config.import_yaml("nonexistent.yaml")
    
    def test_save_file_creates_output(self, smc_file, temp_dir):
        """Test that save_file creates output file."""
        config = SimulationConfig(str(smc_file))
        output_path = str(temp_dir / "output.smc")
        result_path = config.save_file(output_path)
        
        assert Path(result_path).exists()
        assert result_path == output_path
    
    def test_config_validation_with_required_fields(self, temp_dir):
        """Test configuration validation with required fields."""
        # Create minimal valid SMC content
        smc_content = """
        TITLE Test Configuration
        PHOTONS 1000000
        SPECTRUM MONO 140.0
        DETECTOR NaI 0.95
        MATRIX 128 128
        """
        smc_file = temp_dir / "valid.smc"
        smc_file.write_text(smc_content)
        
        config = SimulationConfig(str(smc_file))
        assert config is not None
    
    def test_config_modification_and_save(self, smc_file, temp_dir):
        """Test modifying config and saving."""
        config = SimulationConfig(str(smc_file))
        
        # Modify configuration (this would depend on actual API)
        # For now, just test the save functionality
        output_path = str(temp_dir / "modified.smc")
        saved_path = config.save_file(output_path)
        
        assert Path(saved_path).exists()
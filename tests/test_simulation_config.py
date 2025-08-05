import os
import tempfile
from pathlib import Path

import pytest

from sirf_simind_connection import SimulationConfig
from sirf_simind_connection.configs import get

# Most tests here are unit tests that don't require SIRF
pytestmark = pytest.mark.unit


def test_simulation_config_loading():
    """Test loading SimulationConfig from file."""
    config = SimulationConfig(get("AnyScan.yaml"))
    assert config is not None


def test_config_yaml_loading():
    """Test loading configuration from YAML file."""
    config = SimulationConfig(get("AnyScan.yaml"))

    # Test basic parameter access
    photon_energy = config.get_value("photon_energy")
    assert photon_energy > 0

    # Test flag access
    spect_study = config.get_flag("simulate_spect_study")
    assert isinstance(spect_study, bool)


@pytest.mark.skipif(
    "CI" in os.environ or "GITHUB_ACTIONS" in os.environ,
    reason="SMC config files not available in CI environment",
)
def test_config_smc_loading():
    """Test loading configuration from SMC file."""
    config = SimulationConfig(get("input.smc"))
    assert config is not None

    # Test parameter access
    photon_energy = config.get_value(1)  # Access by index
    assert photon_energy >= 0


def test_config_modification():
    """Test modifying configuration values."""
    config = SimulationConfig(get("AnyScan.yaml"))

    # Test value modification
    config.get_value("photon_energy")
    config.set_value("photon_energy", 150.0)
    assert config.get_value("photon_energy") == 150.0

    # Test flag modification
    config.set_flag("write_results_to_screen", True)
    assert config.get_flag("write_results_to_screen")


def test_config_yaml_export():
    """Test exporting configuration to YAML."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = SimulationConfig(get("AnyScan.yaml"))

        # Modify a parameter
        config.set_value("photon_energy", 140.0)

        # Export to YAML
        yaml_path = Path(temp_dir) / "test_config.yaml"
        config.export_yaml(yaml_path)

        # Verify file was created
        assert yaml_path.exists()

        # Load the exported config and verify
        new_config = SimulationConfig(yaml_path)
        assert new_config.get_value("photon_energy") == 140.0


def test_config_validation():
    """Test configuration parameter validation."""
    config = SimulationConfig(get("AnyScan.yaml"))

    # Test that validation runs without error
    # Note: validation returns True if no warnings, False if there are warnings
    result = config.validate_parameters()
    assert isinstance(result, bool)

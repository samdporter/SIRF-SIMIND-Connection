import tempfile

import pytest

from sirf_simind_connection import SimindSimulator, SimulationConfig
from sirf_simind_connection.configs import get
from sirf_simind_connection.core.components import ScoringRoutine
from sirf_simind_connection.utils.stir_utils import (
    create_attenuation_map,
    create_simple_phantom,
)

# All tests in this file require SIRF since they use SimindSimulator with SIRF objects
pytestmark = pytest.mark.requires_sirf


def test_simulator_initialization():
    """Test initialization of SimindSimulator."""
    config = SimulationConfig(get("AnyScan.yaml"))
    simulator = SimindSimulator(config, "output_dir")
    assert simulator is not None


def test_simulator_configuration():
    """Test simulator configuration methods."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = SimulationConfig(get("AnyScan.yaml"))
        simulator = SimindSimulator(config, temp_dir)

        # Test configuration access
        assert simulator.get_config() is not None
        assert simulator.get_scoring_routine() == ScoringRoutine.SCATTWIN

        # Test configuration modification
        simulator.add_config_value(1, 150.0)  # photon energy
        assert simulator.get_config().get_value(1) == 150.0


def test_simulator_with_phantom():
    """Test simulator with phantom and attenuation map."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = SimulationConfig(get("AnyScan.yaml"))
        simulator = SimindSimulator(config, temp_dir)

        # Create phantom and attenuation map
        phantom = create_simple_phantom()
        mu_map = create_attenuation_map(phantom)

        # Set inputs
        simulator.set_source(phantom)
        simulator.set_mu_map(mu_map)

        # Verify inputs are set
        assert simulator.source is not None
        assert simulator.mu_map is not None

        # Set energy windows
        simulator.set_energy_windows(
            lower_bounds=[126], upper_bounds=[154], scatter_orders=[0]
        )

        assert len(simulator.energy_windows) == 1

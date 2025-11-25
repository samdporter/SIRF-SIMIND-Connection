"""
SIMIND simulator setup helpers.
"""

import logging

from sirf_simind_connection import SimindSimulator, SimulationConfig
from sirf_simind_connection.configs import get
from sirf_simind_connection.core.components import ScoringRoutine


def create_simind_simulator(config, spect_data, output_dir):
    """
    Create and configure SIMIND simulator.

    Uses PENETRATE scoring routine with energy windows set via indices 20 and 21
    (not via .win files). Collimator modeling (index 53) is controlled by the
    coordinator based on correction mode.
    """
    simind_config = SimulationConfig(get(config["simind"]["config"]))

    simulator = SimindSimulator(
        simind_config,
        output_dir=output_dir,
        scoring_routine=ScoringRoutine.PENETRATE,
    )

    simulator.set_source(spect_data["initial_image"])
    simulator.set_mu_map(spect_data["attenuation"])
    simulator.set_template_sinogram(spect_data["acquisition_data"])

    # Set energy windows using indices 20 (upper) and 21 (lower)
    simulator.config.set_value(20, config["simind"]["energy_upper"])
    simulator.config.set_value(21, config["simind"]["energy_lower"])

    # Set photon energy
    if "photon_energy" in config["simind"]:
        photon_energy = config["simind"]["photon_energy"]
        simulator.add_config_value("photon_energy", photon_energy)
        logging.info("Configured photon energy: %s keV", photon_energy)

    # Configure collimator (CC runtime switch)
    if "collimator" in config["simind"]:
        collimator = config["simind"]["collimator"]
        simulator.add_runtime_switch("CC", collimator)
        logging.info("Configured collimator: CC=%s", collimator)

    # Configure source type (FI runtime switch)
    if "source_type" in config["simind"]:
        source_type = config["simind"]["source_type"]
        simulator.add_runtime_switch("FI", source_type)
        logging.info("Configured source type: FI=%s", source_type)

    # Configure photon multiplier (NN runtime switch) if specified
    if "photon_multiplier" in config["simind"]:
        photon_multiplier = config["simind"]["photon_multiplier"]
        simulator.runtime_switches.set_switch("NN", photon_multiplier)
        logging.info("Configured photon multiplier: NN=%s", photon_multiplier)

    # Configure MPI if requested
    if config["simind"].get("use_mpi", False):
        num_cores = config["simind"].get("num_mpi_cores", 6)
        simulator.runtime_switches.set_switch("MP", num_cores)
        logging.info("Configured SIMIND for MPI with %d cores", num_cores)

    logging.info(
        "Created SIMIND simulator (PENETRATE): energy window [%s, %s] keV (indices 20-21)",
        config["simind"]["energy_lower"],
        config["simind"]["energy_upper"],
    )

    return simulator

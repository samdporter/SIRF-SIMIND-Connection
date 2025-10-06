#!/usr/bin/env python3
"""
Compare SPECT PSF modeling approaches using RDP-regularized SVRG reconstruction.

Tests 7 reconstruction approaches:
1. Fast SPECT (no res model)
2. Accurate SPECT (with res model only - no Gaussian)
3. Accurate SPECT (with res model + image-based Gaussian)
4. Fast + Geometric residual correction
5. Accurate + Geometric residual correction
6. Fast + Full residual correction (additive + residual)
7. Accurate + Full residual correction

Uses STIR's CudaRelativeDifferencePrior with SVRG optimization.

IMPORTANT: For efficiency, partitions data ONCE per mode, then loops over beta values.
"""

import argparse
import logging
import os
import time

import numpy as np
import yaml

# CIL imports
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import IndicatorBox
from cil.optimisation.utilities import Sampler, StepSizeRule
from setr.cil_extensions.callbacks import (
    PrintObjectiveCallback,
    SaveImageCallback,
    SaveObjectiveCallback,
)

# SETR imports
from setr.cil_extensions.preconditioners import BSREMPreconditioner
from setr.utils import get_spect_am, get_spect_data
from setr.utils.sirf import get_array
from sirf.STIR import (
    AcquisitionData,
    CudaRelativeDifferencePrior,
    MessageRedirector,
)

# SIRF-SIMIND imports
from sirf_simind_connection import SimindSimulator, SimulationConfig
from sirf_simind_connection.configs import get
from sirf_simind_connection.core.components import ScoringRoutine
from sirf_simind_connection.core.coordinator import SimindCoordinator
from sirf_simind_connection.utils.cil_partitioner import (
    create_svrg_objective_with_rdp,
    partition_data_with_cil_objectives,
)


def configure_logging(verbose=False):
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare SPECT PSF modeling approaches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config
  python compare_psf_models.py --config config_default.yaml --output_path /path/to/output

  # Use quick test config
  python compare_psf_models.py --config config_quick_test.yaml --output_path /tmp/test

  # Override config values
  python compare_psf_models.py --config config_default.yaml --output_path /path/to/output \
      --override svrg.num_epochs=10 --override rdp.beta_values=[0.01,0.1]
        """,
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for reconstructions",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values (e.g., --override svrg.num_epochs=10)",
    )

    return parser.parse_args()


def apply_overrides(config, overrides):
    """Apply command-line overrides to config."""
    for override in overrides:
        if "=" not in override:
            logging.warning(f"Invalid override format: {override}")
            continue

        key_path, value = override.split("=", 1)
        keys = key_path.split(".")

        # Navigate to the right place in config
        target = config
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]

        # Parse value
        try:
            # Try to evaluate as Python literal
            import ast

            parsed_value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Keep as string if eval fails
            parsed_value = value

        target[keys[-1]] = parsed_value
        logging.info(f"Override: {key_path} = {parsed_value}")

    return config


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
    # This works with PENETRATE scoring routine
    simulator.config.set_value(20, config["simind"]["energy_upper"])
    simulator.config.set_value(21, config["simind"]["energy_lower"])

    # Set photon energy
    if "photon_energy" in config["simind"]:
        photon_energy = config["simind"]["photon_energy"]
        simulator.add_config_value("photon_energy", photon_energy)
        logging.info(f"Configured photon energy: {photon_energy} keV")

    # Configure collimator (CC runtime switch)
    if "collimator" in config["simind"]:
        collimator = config["simind"]["collimator"]
        simulator.add_runtime_switch("CC", collimator)
        logging.info(f"Configured collimator: CC={collimator}")

    # Configure source type (FI runtime switch)
    if "source_type" in config["simind"]:
        source_type = config["simind"]["source_type"]
        simulator.add_runtime_switch("FI", source_type)
        logging.info(f"Configured source type: FI={source_type}")

    # Configure photon multiplier (NN runtime switch) if specified
    if "photon_multiplier" in config["simind"]:
        photon_multiplier = config["simind"]["photon_multiplier"]
        simulator.runtime_switches.set_switch("NN", photon_multiplier)
        logging.info(f"Configured photon multiplier: NN={photon_multiplier}")

    # Configure MPI if requested
    if config["simind"].get("use_mpi", False):
        num_cores = config["simind"].get("num_mpi_cores", 6)
        simulator.runtime_switches.set_switch("MP", num_cores)
        logging.info(f"Configured SIMIND for MPI with {num_cores} cores")

    logging.info(
        f"Created SIMIND simulator (PENETRATE): energy window "
        f"[{config['simind']['energy_lower']}, {config['simind']['energy_upper']}] keV "
        f"(indices 20-21)"
    )

    return simulator


class LinearDecayStepSize(StepSizeRule):
    """Linear decay step size: step_size / (1 + eta * k)"""

    def __init__(self, initial_step_size, eta):
        super().__init__()
        self.initial_step_size = initial_step_size
        self.eta = eta

    def get_step_size(self, algorithm):
        k = algorithm.iteration
        return self.initial_step_size / (1 + self.eta * k)


class UpdateEtaCallback:
    """
    Callback to update eta in KL functions after SIMIND simulations.

    For residual_correction modes, eta must be updated after each SIMIND simulation
    to reflect the updated additive term. This callback checks the coordinator's
    cache version and updates eta in all KL data functions when a new simulation
    has completed.

    Args:
        coordinator: SimindCoordinator instance managing SIMIND simulations.
        kl_data_functions: List of CIL KullbackLeibler functions (one per subset).
        partition_indices: List of view indices for each subset.
        epsilon: Small constant added to eta for numerical stability.
    """

    def __init__(self, coordinator, kl_data_functions, partition_indices, epsilon=1e-5):
        self.coordinator = coordinator
        self.kl_data_functions = kl_data_functions
        self.partition_indices = partition_indices
        self.epsilon = epsilon
        self.last_cache_version = 0  # Track which coordinator cache we've processed

    def __call__(self, algorithm):
        """Update eta if new SIMIND simulation results are available."""
        # Check if coordinator has new simulation results
        if self.coordinator.cache_version > self.last_cache_version:
            # Get full additive term from coordinator
            full_additive = self.coordinator.get_full_additive_term()

            if full_additive is not None:
                logging.info(
                    f"Updating eta in KL functions (cache version {self.coordinator.cache_version})"
                )

                # Update eta for each subset
                for i, (kl_func, subset_indices) in enumerate(
                    zip(self.kl_data_functions, self.partition_indices)
                ):
                    # Extract subset views from full additive
                    additive_subset = full_additive.get_subset(subset_indices)

                    # Update eta = additive + epsilon
                    eta_subset = additive_subset + additive_subset.get_uniform_copy(
                        self.epsilon
                    )
                    kl_func.eta = eta_subset

                    if i == 0:  # Log only first subset to avoid spam
                        logging.info(f"  Subset 0 eta sum: {eta_subset.sum():.2e}")

                # Update cache version tracker
                self.last_cache_version = self.coordinator.cache_version


def compute_sensitivity_inverse(projectors):
    """
    Compute the inverse of the sensitivity image for BSREM preconditioner.
    The sensitivity image is the sum of back-projections of ones from each subset.
    S = sum_i( A_i^T * 1_i )
    Args:
        projectors (list): List of acquisition model projectors for all subsets.
    Returns:
        sirf.STIR.ImageData: Image containing the inverse of the sensitivity.
    """
    # Get a template for the sensitivity image from the first projector
    sens = projectors[0].domain_geometry().get_uniform_copy(0)

    for proj in projectors:
        # Create a sinogram of ones and back-project it
        ones = proj.range_geometry().get_uniform_copy(1)
        sens += proj.backward(ones)

    sens = sens.maximum(0)
    sens_arr = get_array(sens)
    s_inv = np.reciprocal(sens_arr, where=sens_arr > 0)
    sens.fill(s_inv)
    return sens


def partition_data_once_cil(
    acquisition_data,
    additive_data,
    acq_model_func,
    initial_image,
    num_subsets,
    coordinator=None,
):
    """
    Partition data ONCE and return CIL objective functions.

    Uses CIL KL objectives composed with LINEAR acquisition models (optionally wrapped
    with SimindSubsetProjector for Monte Carlo corrections).

    IMPORTANT: Returns LINEAR acquisition models and KL functions with eta parameter
    for proper CIL LinearOperator compatibility.

    Args:
        acquisition_data: Measured SPECT data.
        additive_data: Initial additive term (scatter estimate).
        acq_model_func: Factory function returning STIR AcquisitionModel.
        initial_image: Initial image for setup.
        num_subsets: Number of subsets.
        coordinator: SimindCoordinator instance (optional).

    Returns:
        tuple: (kl_objectives, projectors, kl_data_functions, partition_indices)
    """
    logging.info(f"Partitioning data into {num_subsets} subsets (CIL mode)...")

    # Create uniform normalisation for SPECT
    normalisation = acquisition_data.get_uniform_copy(1)

    # Partition using CIL-compatible function
    # Returns LINEAR acquisition models and unwrapped KL functions for eta updates
    kl_objectives, projectors, partition_indices, kl_data_functions = (
        partition_data_with_cil_objectives(
            acquisition_data,
            additive_data,
            normalisation,
            num_subsets,
            initial_image,
            acq_model_func,
            simind_coordinator=coordinator,
            mode="staggered",
        )
    )

    logging.info(f"Created {len(kl_objectives)} CIL KL objectives with LINEAR models")

    return kl_objectives, projectors, kl_data_functions, partition_indices


def run_svrg_with_prior_cil(
    kl_objectives,
    projectors,
    initial_image,
    beta,
    config,
    output_prefix,
    output_dir,
    coordinator=None,
    kl_data_functions=None,
    partition_indices=None,
):
    """
    Run SVRG reconstruction with CIL objectives and RDP prior.

    Args:
        kl_objectives: List of CIL KL objective functions.
        projectors: List of SimindSubsetProjector or STIR AcquisitionModel (LINEAR).
        initial_image: Initial image for reconstruction.
        beta: RDP penalization factor.
        config: Configuration dictionary.
        output_prefix: Prefix for output files.
        output_dir: Output directory.
        coordinator: SimindCoordinator instance (optional, for eta updates).
        kl_data_functions: List of unwrapped KL functions (optional, for eta updates).
        partition_indices: List of subset view indices (optional, for eta updates).

    Returns:
        Reconstructed image.
    """
    logging.info(f"SVRG (CIL): {output_prefix}, beta={beta}")

    num_subsets = len(kl_objectives)

    if beta > 0:
        # Create RDP prior for this beta
        prior = CudaRelativeDifferencePrior()
        prior.set_penalisation_factor(beta)
        prior.set_gamma(config["rdp"]["gamma"])
        prior.set_epsilon(config["rdp"]["epsilon"])
        prior.set_up(initial_image)
    else:
        prior = False

    # Create SVRG function with RDP prior
    sampler = Sampler.random_without_replacement(num_subsets)
    update_interval = num_subsets
    snapshot_update_interval = 2 * update_interval

    # Combine SVRG and RDP using CIL's composition
    total_objective = create_svrg_objective_with_rdp(
        kl_objectives,
        prior,
        sampler,
        snapshot_update_interval,
        initial_image,
    )

    # Constraint: positivity
    g = IndicatorBox(lower=0, upper=np.inf)

    # Step size rule
    step_size_rule = LinearDecayStepSize(
        config["svrg"]["initial_step_size"],
        config["svrg"]["relaxation_eta"],
    )

    # Preconditioner (compute from first projector)
    # For BSREM, we need sensitivity image (backprojection of ones)
    s_inv = compute_sensitivity_inverse(projectors)
    bsrem_precond = BSREMPreconditioner(s_inv, 1, np.inf, epsilon=1e-6, smooth=False)

    callbacks = [
        PrintObjectiveCallback(interval=update_interval),
        SaveObjectiveCallback(
            os.path.join(output_dir, f"{output_prefix}objective.csv"),
            interval=update_interval,
        ),
        SaveImageCallback(
            os.path.join(output_dir, f"{output_prefix}image"), interval=update_interval
        ),
    ]

    # Add UpdateEtaCallback if residual_correction is enabled
    if coordinator is not None and kl_data_functions is not None:
        eta_callback = UpdateEtaCallback(
            coordinator, kl_data_functions, partition_indices
        )
        callbacks.append(eta_callback)
        logging.info("Added UpdateEtaCallback for eta updates after SIMIND simulations")

    # Create ISTA algorithm
    algo = ISTA(
        initial=initial_image.clone(),
        f=total_objective,
        g=g,
        step_size=step_size_rule,
        update_objective_interval=update_interval,
        preconditioner=bsrem_precond,
    )

    # Run reconstruction
    num_iterations = config["svrg"]["num_epochs"] * num_subsets
    logging.info(
        f"Running SVRG for {num_iterations} iterations "
        f"({config['svrg']['num_epochs']} epochs)"
    )

    algo.run(num_iterations, verbose=True, callbacks=callbacks)

    # Save final result
    output_fname = os.path.join(output_dir, f"{output_prefix}_final.hv")
    algo.solution.write(output_fname)
    logging.info(f"Saved: {output_fname}")

    return algo.solution


# ---------------------------
# Refactoring helpers
# ---------------------------


def _log_mode_banner(title: str):
    logging.info("=" * 80)
    logging.info(title)
    logging.info("=" * 80)


def _make_stir_acq_model_factory(
    spect_data, config, *, use_psf: bool, use_gaussian: bool
):
    """Return a factory function that builds STIR acquisition models (no SIMIND wrapper)."""

    def get_am():
        # Extract PSF parameters from config if use_psf=True
        res = None
        gauss_fwhm = None
        if use_psf:
            res = (
                config["resolution"]["stir_psf_params"][0],  # collimator_sigma_0
                config["resolution"]["stir_psf_params"][1],  # collimator_slope
                False,  # full_3D
            )
        if use_gaussian:
            gauss_fwhm = config["resolution"]["psf_fwhm"]

        stir_am = get_spect_am(
            spect_data,
            res=res,
            keep_all_views_in_cache=True,
            gauss_fwhm=gauss_fwhm,
            attenuation=True,
        )
        return stir_am

    return get_am


def _run_mode_core(
    spect_data,
    config,
    output_dir,
    *,
    mode_no: int,
    mode_title: str,
    use_psf: bool,
    use_gaussian: bool,
    residual_correction: bool,
    update_additive: bool,
    simind_dir_suffix: str,
):
    """Shared implementation for all seven modes using CIL objectives."""
    _log_mode_banner(f"MODE {mode_no}: {mode_title}")

    num_subsets = config["svrg"]["num_subsets"]

    # Build STIR acquisition model factory (no SIMIND wrapper)
    get_am = _make_stir_acq_model_factory(
        spect_data,
        config,
        use_psf=use_psf,
        use_gaussian=use_gaussian,
    )

    # Optional SIMIND coordinator setup
    coordinator = None
    if residual_correction or update_additive:
        simind_dir = os.path.join(output_dir, f"simind_{simind_dir_suffix}")
        os.makedirs(simind_dir, exist_ok=True)
        simind_sim = create_simind_simulator(config, spect_data, simind_dir)
        correction_update_interval = (
            config["projector"]["correction_update_epochs"] * num_subsets
        )

        # Create full-data STIR acquisition models for coordinator
        full_acq_data = spect_data["acquisition_data"]  # Full data (all views)
        initial_image = spect_data["initial_image"]

        # Create linear acquisition model (no additive) for coordinator
        linear_am = get_am()  # Use the factory
        linear_am.set_up(full_acq_data, initial_image)

        # For mode_both, also create full model with additive
        stir_am = None
        if residual_correction and update_additive:
            stir_am = get_am()
            stir_am.set_up(full_acq_data, initial_image)

        # Create coordinator
        coordinator = SimindCoordinator(
            simind_simulator=simind_sim,
            num_subsets=num_subsets,
            correction_update_interval=correction_update_interval,
            residual_correction=residual_correction,
            update_additive=update_additive,
            linear_acquisition_model=linear_am,  # Full-data model for projections
            stir_acquisition_model=stir_am,  # Full-data model for mode_both
            output_dir=simind_dir,  # Save intermediate files
        )

        # Initialize coordinator with existing additive term (skip first simulation)
        coordinator.initialize_with_additive(spect_data["additive"])

    # Partition once using CIL objectives
    # Returns LINEAR models and KL data functions for eta updates
    kl_objectives, projectors, kl_data_functions, partition_indices = (
        partition_data_once_cil(
            spect_data["acquisition_data"],
            spect_data["additive"],
            get_am,
            spect_data["initial_image"],
            num_subsets,
            coordinator=coordinator,
        )
    )

    # Loop over beta values
    results = []
    for beta in config["rdp"]["beta_values"]:
        logging.info(f"--- Mode: {mode_no}, Beta: {beta} ---")
        output_dir_for_run = os.path.join(output_dir, f"mode{mode_no}/b{beta}")
        os.makedirs(output_dir_for_run, exist_ok=True)
        output_prefix = ""

        # Reset coordinator iteration counter for each beta
        if coordinator is not None:
            coordinator.reset_iteration_counter()

        recon = run_svrg_with_prior_cil(
            kl_objectives,
            projectors,
            spect_data["initial_image"],
            beta,
            config,
            output_prefix,
            output_dir_for_run,
            coordinator=coordinator,
            kl_data_functions=kl_data_functions,
            partition_indices=partition_indices,
        )
        results.append(recon)

    return results


# ---------------------------
# Public mode functions (thin wrappers)
# ---------------------------


def run_mode_1(spect_data, config, output_dir):
    """Mode 1: Fast SPECT (no resolution model) - all beta values."""
    return _run_mode_core(
        spect_data,
        config,
        output_dir,
        mode_no=1,
        mode_title="Fast SPECT (no res)",
        use_psf=False,
        use_gaussian=False,
        residual_correction=False,
        update_additive=False,
        simind_dir_suffix="mode1",
    )


def run_mode_2(spect_data, config, output_dir):
    """Mode 2: Accurate SPECT (with resolution model only - no Gaussian) - all beta values."""
    return _run_mode_core(
        spect_data,
        config,
        output_dir,
        mode_no=2,
        mode_title="Accurate SPECT (with res only)",
        use_psf=True,
        use_gaussian=False,
        residual_correction=False,
        update_additive=False,
        simind_dir_suffix="mode2",
    )


def run_mode_3(spect_data, config, output_dir):
    """Mode 3: Accurate SPECT (with resolution model + Gaussian) - all beta values."""
    return _run_mode_core(
        spect_data,
        config,
        output_dir,
        mode_no=3,
        mode_title="Accurate SPECT (with res + Gaussian)",
        use_psf=True,
        use_gaussian=True,
        residual_correction=False,
        update_additive=False,
        simind_dir_suffix="mode3",
    )


def run_mode_4(spect_data, config, output_dir):
    """Mode 4: Fast + Geometric residual correction - all beta values."""
    return _run_mode_core(
        spect_data,
        config,
        output_dir,
        mode_no=4,
        mode_title="Fast + Geometric residual",
        use_psf=False,
        use_gaussian=False,
        residual_correction=True,
        update_additive=False,
        simind_dir_suffix="mode4",
    )


def run_mode_5(spect_data, config, output_dir):
    """Mode 5: Accurate + Geometric residual correction - all beta values."""
    return _run_mode_core(
        spect_data,
        config,
        output_dir,
        mode_no=5,
        mode_title="Accurate + Geometric residual",
        use_psf=True,
        use_gaussian=True,
        residual_correction=True,
        update_additive=False,
        simind_dir_suffix="mode5",
    )


def run_mode_6(spect_data, config, output_dir):
    """Mode 6: Fast + Full residual correction - all beta values."""
    return _run_mode_core(
        spect_data,
        config,
        output_dir,
        mode_no=6,
        mode_title="Fast + Full residual",
        use_psf=False,
        use_gaussian=False,
        residual_correction=True,
        update_additive=True,
        simind_dir_suffix="mode6",
    )


def run_mode_7(spect_data, config, output_dir):
    """Mode 7: Accurate + Full residual correction - all beta values."""
    return _run_mode_core(
        spect_data,
        config,
        output_dir,
        mode_no=7,
        mode_title="Accurate + Full residual",
        use_psf=True,
        use_gaussian=True,
        residual_correction=True,
        update_additive=True,
        simind_dir_suffix="mode7",
    )


# ---------------------------
# Main
# ---------------------------


def main():
    """Main entry point."""
    args = parse_args()
    AcquisitionData.set_storage_scheme("memory")

    # Load config
    config = load_config(args.config)

    # Apply overrides
    config = apply_overrides(config, args.override)

    # Configure logging
    configure_logging(config["output"]["verbose"])
    msg = MessageRedirector()

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Log configuration
    logging.info("=" * 80)
    logging.info("SPECT PSF Model Comparison")
    logging.info("=" * 80)
    logging.info(f"Config: {args.config}")
    logging.info(f"Data: {config['data_path']}")
    logging.info(f"Output: {args.output_path}")
    logging.info(f"Modes: {config['reconstruction']['modes']}")
    logging.info(f"Beta values: {config['rdp']['beta_values']}")
    logging.info(
        f"SVRG: {config['svrg']['num_epochs']} epochs, {config['svrg']['num_subsets']} subsets"
    )
    logging.info(
        f"Energy window: "
        f"[{config['simind']['energy_lower']}, {config['simind']['energy_upper']}] keV"
    )
    logging.info("=" * 80)

    # Load data ONCE
    start_time = time.time()
    spect_data = get_spect_data(config["data_path"])
    logging.info(f"Data loading: {time.time() - start_time:.2f}s")

    # Reconstruction mode dispatch
    mode_funcs = {
        1: run_mode_1,
        2: run_mode_2,
        3: run_mode_3,
        4: run_mode_4,
        5: run_mode_5,
        6: run_mode_6,
        7: run_mode_7,
    }

    # Run reconstructions - each mode handles all beta values internally
    total_start = time.time()

    for mode in sorted(config["reconstruction"]["modes"]):
        mode_start = time.time()
        try:
            # Run all beta values for this mode
            mode_funcs[mode](spect_data, config, args.output_path)
            logging.info(f"Mode {mode} (all betas): {time.time() - mode_start:.2f}s")
        except Exception as e:
            logging.error(f"Mode {mode} failed: {e}", exc_info=True)

    # Summary
    total_time = time.time() - total_start
    logging.info("=" * 80)
    logging.info(f"Completed in {total_time:.2f}s")
    logging.info(f"Results: {args.output_path}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

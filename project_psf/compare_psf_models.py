#!/usr/bin/env python3
"""
Compare SPECT PSF modeling approaches using RDP-regularized SVRG reconstruction.

Tests 8 reconstruction approaches:
1. Fast SPECT (no res model)
2. Accurate SPECT (with res model only - no Gaussian)
3. Accurate SPECT (with res model + image-based Gaussian)
4. STIR PSF residual correction (no SIMIND)
5. Fast + SIMIND Geometric residual correction
6. Accurate + SIMIND Geometric residual correction
7. Fast + SIMIND Full residual correction (additive + residual)
8. Accurate + SIMIND Full residual correction

Uses SETR's RelativeDifferencePrior with SVRG optimization.

IMPORTANT: For efficiency, partitions data ONCE per mode, then loops over beta values.
"""

import argparse
import csv
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

# CIL imports
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import IndicatorBox, ScaledFunction
from cil.optimisation.utilities import Sampler
from setr.cil_extensions.callbacks import (
    PrintObjectiveCallback,
    SaveImageCallback,
    SavePreconditionerCallback,
)

# SETR imports
from setr.cil_extensions.preconditioners import (
    BSREMPreconditioner,
    ImageFunctionPreconditioner,
    LehmerMeanPreconditioner,
    PoissonHessianPreconditioner,
    PreconditionerWithInterval,
    SubsetPoissonHessianPreconditioner,
)
from setr.priors import RelativeDifferencePrior
from setr.utils import get_spect_am, get_spect_data
from sirf.STIR import AcquisitionData, MessageRedirector

# Local imports
from step_size_rules import (
    ArmijoAfterCorrectionStepSize,
    ArmijoTriggerCallback,
    LinearDecayStepSizeRule,
    SaveStepSizeHistoryCallback,
)

# SIRF-SIMIND imports
from sirf_simind_connection import SimindSimulator, SimulationConfig
from sirf_simind_connection.configs import get
from sirf_simind_connection.core.components import ScoringRoutine
from sirf_simind_connection.core.coordinator import (
    SimindCoordinator,
    StirPsfCoordinator,
)
from sirf_simind_connection.utils import get_array
from sirf_simind_connection.utils.cil_partitioner import (
    create_full_objective_with_rdp,
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
      --override stochastic.num_epochs=10 --override rdp.beta_values=[0.01,0.1]
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
        help=(
            "Override config values (e.g., --override stochastic.num_epochs=10). "
            "Legacy keys (svrg.*) remain supported."
        ),
    )

    parser.add_argument(
        "--execution",
        choices=["local", "cluster"],
        default="local",
        help="Run locally (default) or stage/submit an SGE cluster sweep.",
    )

    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit jobs immediately in cluster mode (otherwise stage only).",
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


class SaveObjectiveWithIterationCallback:
    """Log objective values with iteration numbers to a CSV file."""

    def __init__(self, csv_path, interval, iteration_offset=0):
        self.csv_path = csv_path
        self._interval = max(int(interval), 1)
        self.iteration_offset = int(iteration_offset)
        self.records = []

    def __call__(self, algorithm):
        # CIL reports iteration=-1 before the first update; skip in that case.
        if algorithm.iteration < 0:
            return

        global_iteration = self.iteration_offset + algorithm.iteration
        if global_iteration % self._interval != 0:
            return

        objective_value = algorithm.f(algorithm.solution) + algorithm.g(
            algorithm.solution
        )
        self.records.append(
            {
                "iteration": global_iteration,
                "objective": float(objective_value),
            }
        )
        self._flush()

    def increment_iteration_offset(self, increment):
        """Advance the iteration offset between sequential algorithm runs."""
        self.iteration_offset += int(increment)

    def set_interval(self, interval):
        """Update the logging interval (must be >= 1)."""
        self._interval = max(int(interval), 1)

    def _flush(self):
        """Persist the collected objective history to CSV."""
        output_dir = os.path.dirname(self.csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            with open(self.csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["iteration", "objective"])
                writer.writeheader()
                writer.writerows(self.records)
        except OSError as exc:
            logging.error(
                "Failed to write objective history to %s: %s", self.csv_path, exc
            )


def _format_literal(value):
    """Return a string representation safe for literal evaluation."""
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    return repr(value)


def _generate_mode_beta_combinations(config):
    """Return sorted list of (mode, beta_literal) pairs for cluster runs."""
    modes = sorted(int(mode) for mode in config["reconstruction"]["modes"])
    betas = config["rdp"]["beta_values"]
    combinations = []
    for mode in modes:
        for beta in betas:
            combinations.append((mode, _format_literal(beta)))
    return combinations


def run_cluster_sweep(args, config):
    """Launch or stage an SGE array sweep over (mode, beta) combinations."""

    combinations = _generate_mode_beta_combinations(config)
    if not combinations:
        logging.error("No (mode, beta) combinations available for cluster run")
        return

    cluster_cfg = config.get("cluster", {})
    job_name = cluster_cfg.get("job_name", "psf_mode_beta")
    runtime = cluster_cfg.get("runtime", "48:00:00")
    memory = cluster_cfg.get("memory", "60G")
    queue = cluster_cfg.get("queue")
    cores = int(cluster_cfg.get("cores", 1))
    gpu = bool(cluster_cfg.get("gpu", False))
    submit_jobs = bool(args.submit or cluster_cfg.get("submit", False))
    python_exec = cluster_cfg.get("python_executable", sys.executable)
    env_setup = cluster_cfg.get("env_setup_script")
    logs_subdir = cluster_cfg.get("logs_subdir", "_logs")
    qsub_extra = cluster_cfg.get("qsub_extra_args", [])

    if isinstance(qsub_extra, str):
        qsub_extra = shlex.split(qsub_extra)
    elif not isinstance(qsub_extra, (list, tuple)):
        qsub_extra = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    job_script_default = os.path.join(script_dir, "cluster", "psf_sweep_job.sh")
    job_script = os.path.abspath(cluster_cfg.get("job_script", job_script_default))

    if not os.path.isfile(job_script):
        raise FileNotFoundError(
            f"Cluster job script not found: {job_script}. Check config['cluster']['job_script']."
        )

    config_path = os.path.abspath(args.config)
    output_root = os.path.abspath(args.output_path)
    os.makedirs(output_root, exist_ok=True)

    cluster_root = os.path.join(output_root, "_cluster")
    os.makedirs(cluster_root, exist_ok=True)

    task_file = os.path.join(cluster_root, f"{job_name}_tasks.csv")
    with open(task_file, "w", encoding="utf-8") as f:
        for mode, beta_literal in combinations:
            f.write(f"{mode},{beta_literal}\n")

    overrides_file = None
    if args.override:
        overrides_file = os.path.join(cluster_root, f"{job_name}_overrides.txt")
        with open(overrides_file, "w", encoding="utf-8") as f:
            for override in args.override:
                f.write(f"{override}\n")

    if env_setup:
        env_setup = os.path.abspath(env_setup)

    logs_dir = os.path.join(cluster_root, logs_subdir)
    os.makedirs(logs_dir, exist_ok=True)

    env_exports = {
        "PSF_SCRIPT": os.path.abspath(__file__),
        "CONFIG_PATH": config_path,
        "OUTPUT_ROOT": output_root,
        "TASK_FILE": task_file,
        "PYTHON_EXECUTABLE": python_exec,
    }

    if overrides_file:
        env_exports["OVERRIDES_FILE"] = overrides_file
    if env_setup:
        env_exports["ENV_SETUP_SCRIPT"] = env_setup
    env_exports["CLUSTER_CORES"] = str(cores)

    def _escape_env_value(value):
        text = str(value)
        text = text.replace("\\", "\\\\")
        text = text.replace(",", "\\,")
        text = text.replace(" ", "\\ ")
        return text

    env_export_str = ",".join(
        f"{key}={_escape_env_value(value)}" for key, value in env_exports.items()
    )

    total_tasks = len(combinations)
    task_range = f"1-{total_tasks}"

    qsub_cmd = [
        "qsub",
        "-t",
        task_range,
        "-N",
        job_name,
        "-l",
        f"h_rt={runtime}",
    ]

    if gpu:
        qsub_cmd.extend(["-l", "gpu=true", "-l", f"tmem={memory}"])
    else:
        qsub_cmd.extend(["-l", f"h_vmem={memory}"])

    if queue:
        qsub_cmd.extend(["-q", queue])

    if cores > 1:
        qsub_cmd.extend(["-pe", "smp", str(cores)])

    qsub_cmd.extend(["-o", logs_dir, "-e", logs_dir])

    qsub_cmd.extend(qsub_extra)

    qsub_cmd.extend(["-v", env_export_str, job_script])

    logging.info("Prepared %d SGE tasks (modes × betas).", total_tasks)
    logging.info("Task file: %s", task_file)
    logging.info("Logs directory: %s", logs_dir)
    logging.info("Job script: %s", job_script)
    logging.info("Sample command: %s", " ".join(shlex.quote(arg) for arg in qsub_cmd))

    if submit_jobs:
        if shutil.which("qsub") is None:
            logging.error("qsub command not found in PATH; cannot submit jobs.")
            return

        logging.info("Submitting SGE array job...")
        try:
            subprocess.run(qsub_cmd, check=True)
        except subprocess.CalledProcessError as exc:
            logging.error("qsub submission failed: %s", exc)
            return

        logging.info("SGE submission successful. Monitor with qstat -u $USER")
    else:
        logging.info(
            "Cluster run staged only. Re-run with --submit or set cluster.submit=true to submit."
        )


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


# LinearDecayStepSize now imported from step_size_rules.py


class UpdateEtaCallback:
    """
    Callback to update additive/residual terms in KL functions after SIMIND simulations.

    This callback refreshes the additive component (`eta` for classical KL, `additive`
    for the residual-aware variant) and, when available, the residual correction.
    Updates are triggered whenever the coordinator publishes a new cache version.

    Args:
        coordinator: SimindCoordinator instance managing SIMIND simulations.
        kl_data_functions: List of KL-like data functions (one per subset).
        partition_indices: List of view indices for each subset.
        eta_floor: Non-negative floor applied to additive terms for stability.
    """

    def __init__(
        self, coordinator, kl_data_functions, partition_indices, eta_floor=1e-8
    ):
        self.coordinator = coordinator
        self.kl_data_functions = kl_data_functions
        self.partition_indices = partition_indices
        self.eta_floor = float(max(eta_floor, 0.0))
        self.last_cache_version = 0  # Track which coordinator cache we've processed

    def __call__(self, algorithm):
        """Update eta if new SIMIND simulation results are available."""
        # Check if coordinator has new simulation results
        if self.coordinator.cache_version > self.last_cache_version:
            # Get full additive term from coordinator
            full_additive = self.coordinator.get_full_additive_term()

            if full_additive is not None:
                logging.info(
                    f"UpdateEtaCallback: Updating KL at iteration "
                    f"{algorithm.iteration} "
                    f"(cache version {self.last_cache_version} -> "
                    f"{self.coordinator.cache_version})"
                )

                # Update eta for each subset
                for i, (kl_func, subset_indices) in enumerate(
                    zip(self.kl_data_functions, self.partition_indices)
                ):
                    # Extract subset views from full additive
                    additive_subset = full_additive.get_subset(subset_indices)

                    if self.eta_floor > 0.0:
                        additive_base = additive_subset.maximum(self.eta_floor)
                    else:
                        additive_base = additive_subset

                    residual_subset = self.coordinator.get_subset_residual(
                        subset_indices, current_additive_subset=additive_base
                    )
                    if residual_subset is None:
                        residual_subset = additive_base.get_uniform_copy(0)

                    eta_subset = additive_base + residual_subset

                    if hasattr(kl_func, "eta"):
                        kl_func.eta = eta_subset
                    else:
                        raise TypeError(
                            "UpdateEtaCallback requires KL functions with an 'eta' attribute"
                        )

                    eta_min_val = float(get_array(eta_subset).min())
                    eta_cap_val = eta_min_val
                    if eta_cap_val <= 0.0:
                        floor_val = self.eta_floor if self.eta_floor > 0.0 else 1e-12
                        eta_cap_val = floor_val
                    logging.info(
                        "  Subset %d base additive sum: %.2e, residual sum: %.2e, "
                        "combined min: %.2e",
                        i,
                        additive_base.sum(),
                        residual_subset.sum(),
                        eta_min_val,
                    )

                    algo = getattr(self.coordinator, "algorithm", None)
                    if algo is not None and hasattr(
                        algo, "positivity_eta_min_per_subset"
                    ):
                        if i < len(algo.positivity_eta_min_per_subset):
                            algo.positivity_eta_min_per_subset[i] = eta_cap_val
                            algo.positivity_eta_min_global = min(
                                algo.positivity_eta_min_per_subset
                            )

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


def _traverse_preconditioners(root):
    """Yield preconditioner nodes reachable from root (following .preconds if present)."""
    stack = [root]
    seen = set()
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        children = getattr(node, "preconds", None)
        if children:
            stack.extend(children)


def _set_preconditioner_update_interval(preconditioner, interval):
    """
    Force all nested PreconditionerWithInterval instances to use a specific update interval.

    Returns:
        dict: Mapping of id(node) -> original interval for restoration.
    """
    originals = {}
    for node in _traverse_preconditioners(preconditioner):
        if isinstance(node, PreconditionerWithInterval):
            originals[id(node)] = node.update_interval
            node.update_interval = interval
    return originals


def _restore_preconditioner_update_interval(preconditioner, originals):
    """Restore update_interval values from a snapshot produced by _set_preconditioner_update_interval."""
    if not originals:
        return
    for node in _traverse_preconditioners(preconditioner):
        key = id(node)
        if key in originals:
            node.update_interval = originals[key]


def _reset_preconditioner_runtime_state(preconditioner):
    """
    Clear cached state (precond/freeze/accumulators) so fresh estimates are built next time.
    """
    for node in _traverse_preconditioners(preconditioner):
        if hasattr(node, "precond"):
            node.precond = None
        if hasattr(node, "freeze"):
            node.freeze = None
        if isinstance(node, SubsetPoissonHessianPreconditioner):
            node._accumulator = None
            node._sum_weights = 0.0
            node._contributions = 0


def _normalise_preconditioner_cfg(precond_cfg):
    if precond_cfg is None:
        return {}
    if isinstance(precond_cfg, str):
        return {"type": precond_cfg}
    return dict(precond_cfg)


def _ensure_numeric(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"inf", "+inf", "infinity"}:
            return np.inf
        if lowered in {"-inf", "minf"}:
            return -np.inf
        if lowered in {"none", "null"}:
            return default
    return value


def build_preconditioner(
    config,
    projectors,
    kl_objectives,
    initial_image,
    num_subsets,
    beta,
    rdp_prior,
):
    """
    Create the data-term preconditioner (BSREM or Poisson Hessian) and optionally
    combine it with the RDP prior preconditioner using a Lehmer mean.
    """
    precond_cfg = _normalise_preconditioner_cfg(config.get("preconditioner"))
    precond_type = precond_cfg.get("type", "poisson_hessian").lower()

    if precond_type not in {"bsrem", "poisson_hessian", "hessian", "both"}:
        raise ValueError(
            f"Unknown preconditioner type '{precond_type}'. "
            "Valid options: 'bsrem', 'poisson_hessian', 'both'."
        )

    def _build_bsrem():
        bsrem_cfg = precond_cfg.get("bsrem", {})
        update_interval = bsrem_cfg.get(
            "update_interval", precond_cfg.get("update_interval", num_subsets)
        )
        freeze_iter = bsrem_cfg.get(
            "freeze_iter", precond_cfg.get("freeze_iter", np.inf)
        )
        epsilon = bsrem_cfg.get("epsilon", precond_cfg.get("epsilon", 0.0))
        smooth = bsrem_cfg.get("smooth", False)
        max_value = _ensure_numeric(
            bsrem_cfg.get(
                "max_value", precond_cfg.get("max_value", initial_image.max())
            ),
            initial_image.max(),
        )

        s_inv = compute_sensitivity_inverse(projectors)
        precond = BSREMPreconditioner(
            s_inv,
            update_interval=update_interval,
            freeze_iter=freeze_iter,
            epsilon=epsilon,
            smooth=smooth,
            max_val=max_value,
        )

        fwhm = bsrem_cfg.get("gaussian_fwhm", precond_cfg.get("gaussian_fwhm"))
        if smooth and fwhm is not None and hasattr(precond, "gaussian"):
            try:
                precond.gaussian.set_fwhms(tuple(fwhm))
            except Exception as exc:  # pragma: no cover - defensive logging
                logging.warning(
                    "Unable to set BSREM Gaussian FWHM to %s: %s", fwhm, exc
                )

        return precond, update_interval, freeze_iter

    def _build_poisson_hessian():
        hess_cfg = precond_cfg.get("poisson_hessian", {})
        update_interval = hess_cfg.get(
            "update_interval", precond_cfg.get("update_interval", 1)
        )
        freeze_iter = hess_cfg.get(
            "freeze_iter", precond_cfg.get("freeze_iter", np.inf)
        )
        epsilon = hess_cfg.get("epsilon", precond_cfg.get("epsilon", 0.0))
        max_value = _ensure_numeric(
            hess_cfg.get("max_value", precond_cfg.get("max_value", np.inf)),
            np.inf,
        )
        hessian_floor = hess_cfg.get("hessian_floor", 1e-12)
        mode = hess_cfg.get("mode", "sum")
        ema_decay = hess_cfg.get("ema_decay", 0.9)
        reset_interval = hess_cfg.get("reset_interval")
        weights = hess_cfg.get("weights")
        scale_to_full = hess_cfg.get("scale_to_full", True)
        use_full = hess_cfg.get("use_full", hess_cfg.get("full_batch", False))

        if use_full or num_subsets <= 1:
            precond = PoissonHessianPreconditioner(
                kl_objectives,
                update_interval=update_interval,
                freeze_iter=freeze_iter,
                epsilon=epsilon,
                max_value=max_value,
                hessian_floor=hessian_floor,
            )
        else:
            precond = SubsetPoissonHessianPreconditioner(
                kl_objectives,
                update_interval=update_interval,
                freeze_iter=freeze_iter,
                epsilon=epsilon,
                max_value=max_value,
                hessian_floor=hessian_floor,
                mode=mode,
                ema_decay=ema_decay,
                reset_interval=reset_interval,
                weights=weights,
                scale_to_full=scale_to_full,
            )
        return precond, update_interval, freeze_iter

    data_precond = None
    data_update_interval = None
    data_freeze_iter = None

    if precond_type == "bsrem":
        data_precond, data_update_interval, data_freeze_iter = _build_bsrem()
    elif precond_type in {"poisson_hessian", "hessian"}:
        data_precond, data_update_interval, data_freeze_iter = _build_poisson_hessian()
    else:  # both
        bsrem_precond, bsrem_update_interval, bsrem_freeze_iter = _build_bsrem()
        hess_precond, hess_update_interval, hess_freeze_iter = _build_poisson_hessian()
        both_cfg = precond_cfg.get("both", {})
        global_lehmer_cfg = precond_cfg.get("lehmer", {})
        data_scales = both_cfg.get("scales", [0.5, 0.5])
        data_p = both_cfg.get("p", global_lehmer_cfg.get("p", 1e-1))
        data_epsilon = both_cfg.get("epsilon", global_lehmer_cfg.get("epsilon", 1e-8))
        combined_update_interval = max(bsrem_update_interval, hess_update_interval)
        combined_freeze_iter = min(bsrem_freeze_iter, hess_freeze_iter)
        data_precond = LehmerMeanPreconditioner(
            [bsrem_precond, hess_precond],
            p=data_p,
            epsilon=data_epsilon,
            update_interval=combined_update_interval,
            freeze_iter=combined_freeze_iter,
            scales=data_scales,
        )
        data_update_interval = combined_update_interval
        data_freeze_iter = combined_freeze_iter

    # If no prior active, return data preconditioner directly
    if rdp_prior is None or beta <= 0:
        return data_precond

    rdp_cfg = config.get("rdp", {})
    prior_cfg = precond_cfg.get("prior", {})
    prior_update_interval = prior_cfg.get(
        "update_interval", precond_cfg.get("prior_update_interval", num_subsets)
    )
    prior_freeze_iter = prior_cfg.get(
        "freeze_iter", precond_cfg.get("prior_freeze_iter", np.inf)
    )
    prior_epsilon = prior_cfg.get(
        "epsilon", rdp_cfg.get("preconditioner_epsilon", 1e-8)
    )
    prior_max_value = _ensure_numeric(
        prior_cfg.get("max_value", np.inf),
        np.inf,
    )

    def rdp_inv_hessian(image):
        inv_diag = rdp_prior.inv_hessian_diag(image)
        inv_diag.abs(out=inv_diag)
        inv_diag.divide(max(beta, 1e-12), out=inv_diag)
        return inv_diag

    prior_precond = ImageFunctionPreconditioner(
        rdp_inv_hessian,
        update_interval=prior_update_interval,
        freeze_iter=prior_freeze_iter,
        epsilon=prior_epsilon,
        max_value=prior_max_value,
    )

    combine_mode = prior_cfg.get("combine", precond_cfg.get("combine", "lehmer"))
    if combine_mode is None or str(combine_mode).lower() == "none":
        return data_precond

    combine_mode = str(combine_mode).lower()
    if combine_mode != "lehmer":
        raise ValueError(
            f"Unsupported preconditioner combination '{combine_mode}'. "
            "Currently only 'lehmer' is supported."
        )

    lehmer_cfg = precond_cfg.get("lehmer", {})
    p_value = lehmer_cfg.get("p", 1e-1)
    lehmer_epsilon = lehmer_cfg.get("epsilon", rdp_cfg.get("lehmer_epsilon", 1e-8))
    combined_update_interval = lehmer_cfg.get(
        "update_interval", max(data_update_interval, prior_update_interval)
    )
    combined_freeze_iter = lehmer_cfg.get(
        "freeze_iter", min(data_freeze_iter, prior_freeze_iter)
    )

    return LehmerMeanPreconditioner(
        [data_precond, prior_precond],
        p=p_value,
        epsilon=lehmer_epsilon,
        update_interval=combined_update_interval,
        freeze_iter=combined_freeze_iter,
    )


def partition_data_once_cil(
    acquisition_data,
    additive_data,
    acq_model_func,
    initial_image,
    num_subsets,
    coordinator=None,
    eta_floor=1e-5,
    count_floor=1e-8,
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
        eta_floor: Minimum additive value to maintain positivity.
        count_floor: Minimum measured count used inside the log term.

    Returns:
        tuple: (
            kl_objectives,
            projectors,
            kl_data_functions,
            partition_indices,
            subset_sensitivity_max,
            subset_eta_min,
        )
    """
    logging.info(f"Partitioning data into {num_subsets} subsets (CIL mode)...")

    # Create uniform normalisation for SPECT
    normalisation = acquisition_data.get_uniform_copy(1)

    # Partition using CIL-compatible function
    # Returns LINEAR acquisition models and unwrapped KL functions for eta updates
    (
        kl_objectives,
        projectors,
        partition_indices,
        kl_data_functions,
        subset_sensitivity_max,
        subset_eta_min,
    ) = partition_data_with_cil_objectives(
        acquisition_data,
        additive_data,
        normalisation,
        num_subsets,
        initial_image,
        acq_model_func,
        coordinator=coordinator,
        mode="staggered",
        eta_floor=eta_floor,
        count_floor=count_floor,
    )

    logging.info(f"Created {len(kl_objectives)} CIL KL objectives with LINEAR models")

    return (
        kl_objectives,
        projectors,
        kl_data_functions,
        partition_indices,
        subset_sensitivity_max,
        subset_eta_min,
    )


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
    subset_sensitivity_max=None,
    subset_eta_min=None,
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
    solver_cfg = _get_solver_config(config)
    algorithm = solver_cfg.get("algorithm", "SVRG").upper()
    max_update_scale = solver_cfg.get("max_update_scale")
    data_fidelity_cfg = config.get("data_fidelity", {})
    eta_floor = data_fidelity_cfg.get("eta_floor", 1e-5)
    count_floor = data_fidelity_cfg.get("counts_floor", 1e-8)
    count_floor = data_fidelity_cfg.get("counts_floor", 1e-8)

    logging.info(f"{algorithm} (CIL): {output_prefix}, beta={beta}")

    num_subsets = len(kl_objectives)

    rdp_prior = None
    if beta > 0:
        rdp_prior = RelativeDifferencePrior(
            domain_geometry=initial_image,
            gamma=config["rdp"].get("gamma", 1.0),
            epsilon=config["rdp"].get("epsilon", 1e-6),
            stencil=str(config["rdp"].get("stencil", "18")),
            both_directions=config["rdp"].get("both_directions", True),
            bnd_cond=config["rdp"].get("boundary_condition", "Periodic"),
        )
        prior = ScaledFunction(rdp_prior, -beta)
    else:
        prior = False

    # Create SVRG function with RDP prior
    sampler = Sampler.random_without_replacement(num_subsets)
    update_interval = num_subsets
    if algorithm == "SVRG":
        snapshot_update_interval = solver_cfg.get(
            "snapshot_update_interval", update_interval
        )
    else:
        snapshot_update_interval = None

    # Combine SVRG and RDP using CIL's composition
    total_objective = create_svrg_objective_with_rdp(
        kl_objectives,
        prior,
        sampler,
        snapshot_update_interval,
        initial_image,
        algorithm=algorithm,
    )

    # Constraint: positivity
    g = IndicatorBox(lower=0, upper=np.inf)

    # Step size rule
    use_armijo_after_correction = coordinator is not None and solver_cfg.get(
        "use_armijo_after_correction", True
    )
    use_armijo_periodic = solver_cfg.get("use_armijo_periodic", False)
    use_armijo = use_armijo_after_correction or use_armijo_periodic

    warmup_iterations = 0

    if use_armijo:
        correction_update_interval = (
            config["projector"]["correction_update_epochs"] * num_subsets
        )
        periodic_interval = 0
        if use_armijo_periodic:
            periodic_epochs = solver_cfg.get("armijo_periodic_epochs")
            if periodic_epochs is None:
                # Fall back to previous behaviour when no coordinator is available
                if coordinator is None:
                    periodic_interval = correction_update_interval
            elif periodic_epochs > 0:
                periodic_interval = max(int(periodic_epochs * num_subsets), 1)

        warmup_iterations = max(int(solver_cfg.get("armijo_initial_iterations", 0)), 0)

        step_size_rule = ArmijoAfterCorrectionStepSize(
            initial_step_size=solver_cfg["initial_step_size"],
            beta=solver_cfg.get("armijo_beta", 0.5),
            decay_rate=solver_cfg["relaxation_eta"],
            max_iter=solver_cfg.get("armijo_max_iter", 20),
            tol=solver_cfg.get("armijo_tol", 1e-4),
            update_interval=periodic_interval,
            initial_armijo_iterations=warmup_iterations,
        )
        logging.info(
            "Using ArmijoAfterCorrectionStepSize "
            f"(periodic_interval={periodic_interval}, "
            f"warmup_iterations={warmup_iterations})"
        )
    else:
        # No coordinator or Armijo disabled: use simple linear decay
        step_size_rule = LinearDecayStepSizeRule(
            solver_cfg["initial_step_size"],
            solver_cfg["relaxation_eta"],
        )
        logging.info("Using LinearDecayStepSizeRule")

    preconditioner = build_preconditioner(
        config=config,
        projectors=projectors,
        kl_objectives=kl_objectives,
        initial_image=initial_image,
        num_subsets=num_subsets,
        beta=beta,
        rdp_prior=rdp_prior,
    )

    objective_csv_path = os.path.join(output_dir, f"{output_prefix}objective.csv")
    objective_logger = SaveObjectiveWithIterationCallback(
        objective_csv_path, interval=update_interval
    )

    step_size_logger = None
    if use_armijo:
        step_csv_path = os.path.join(output_dir, f"{output_prefix}step_sizes.csv")
        step_size_logger = SaveStepSizeHistoryCallback(
            step_csv_path, coordinator=coordinator
        )
        logging.info(
            "Configured SaveStepSizeHistoryCallback (logging to %s)", step_csv_path
        )

    # Optional Armijo warm-up with full-gradient updates
    if use_armijo and warmup_iterations > 0:
        warmup_objective = create_full_objective_with_rdp(kl_objectives, prior)
        objective_logger.set_interval(1)
        warmup_algo = ISTA(
            initial=initial_image.clone(),
            f=warmup_objective,
            g=g,
            step_size=step_size_rule,
            update_objective_interval=update_interval,
            preconditioner=preconditioner,
        )
        _configure_update_clipping(warmup_algo, max_update_scale)
        _configure_positivity_cap(
            warmup_algo,
            solver_cfg,
            subset_sensitivity_max,
            subset_eta_min,
        )
        warmup_callbacks = [
            PrintObjectiveCallback(interval=update_interval),
            objective_logger,
        ]
        if step_size_logger is not None:
            warmup_callbacks.append(step_size_logger)

        warmup_epochs = warmup_iterations / max(num_subsets, 1)
        logging.info(
            "Running Armijo warm-up: %d iterations (%.2f epochs)",
            warmup_iterations,
            warmup_epochs,
        )
        precond_interval_snapshot = None
        if preconditioner is not None:
            precond_interval_snapshot = _set_preconditioner_update_interval(
                preconditioner, 1
            )
            _reset_preconditioner_runtime_state(preconditioner)
        previous_algorithm = None
        if coordinator is not None:
            previous_algorithm = coordinator.algorithm
            coordinator.algorithm = warmup_algo
        try:
            warmup_algo.run(warmup_iterations, verbose=True, callbacks=warmup_callbacks)
        finally:
            if coordinator is not None:
                coordinator.algorithm = previous_algorithm
            if preconditioner is not None:
                _restore_preconditioner_update_interval(
                    preconditioner, precond_interval_snapshot
                )
                _reset_preconditioner_runtime_state(preconditioner)
        warmup_clip_iters = getattr(warmup_algo, "_clip_iterations", 0)
        if warmup_clip_iters:
            logging.info(
                "Warm-up clipping active on %d iterations (scale %.3g, last limit %.3e)",
                warmup_clip_iters,
                getattr(warmup_algo, "max_update_scale", float("nan")),
                getattr(warmup_algo, "_last_clip_limit", float("nan")),
            )
        warmup_pos_hits = getattr(warmup_algo, "_positivity_cap_hits", 0)
        if warmup_pos_hits:
            logging.info(
                "Warm-up positivity cap active on %d iterations (theta %.3g)",
                warmup_pos_hits,
                getattr(warmup_algo, "positivity_theta", float("nan")),
            )
        initial_image = warmup_algo.solution.clone()

        # Reset warm-up configuration before stochastic updates
        step_size_rule.initial_armijo_iterations = 0
        capped_step = step_size_rule.apply_warmup_cap()
        step_size_rule.reinitialize_decay(start_iteration=0)
        if step_size_logger is not None:
            step_size_logger.increment_iteration_offset(warmup_iterations)
        objective_logger.increment_iteration_offset(warmup_iterations)
        objective_logger.set_interval(update_interval)
        logging.info(
            "Completed Armijo warm-up phase (max Armijo step size now %.6f)",
            capped_step,
        )

    callbacks = [
        PrintObjectiveCallback(interval=update_interval),
        objective_logger,
        SaveImageCallback(
            os.path.join(output_dir, f"{output_prefix}image"), interval=update_interval
        ),
    ]

    output_cfg = config.get("output", {})
    if output_cfg.get("save_preconditioner", False) and preconditioner is not None:
        precond_interval = max(
            int(output_cfg.get("preconditioner_interval", update_interval)), 1
        )
        callbacks.append(
            SavePreconditionerCallback(
                os.path.join(output_dir, f"{output_prefix}preconditioner"),
                interval=precond_interval,
            )
        )
        logging.info("Preconditioner snapshots enabled (interval=%d)", precond_interval)

    # Add UpdateEtaCallback if residual_correction is enabled
    if coordinator is not None and kl_data_functions is not None:
        eta_callback = UpdateEtaCallback(
            coordinator,
            kl_data_functions,
            partition_indices,
            eta_floor=eta_floor,
        )
        callbacks.append(eta_callback)
        logging.info("Added UpdateEtaCallback for eta updates after SIMIND simulations")

    if step_size_logger is not None:
        callbacks.append(step_size_logger)

    if use_armijo and coordinator is not None:
        armijo_trigger = ArmijoTriggerCallback(coordinator)
        callbacks.append(armijo_trigger)
        logging.info("Added ArmijoTriggerCallback to trigger Armijo after corrections")

    # Create ISTA algorithm
    algo = ISTA(
        initial=initial_image.clone(),
        f=total_objective,
        g=g,
        step_size=step_size_rule,
        update_objective_interval=update_interval,
        preconditioner=preconditioner,
    )
    _configure_update_clipping(algo, max_update_scale)
    _configure_positivity_cap(
        algo,
        solver_cfg,
        subset_sensitivity_max,
        subset_eta_min,
    )

    if coordinator is not None:
        coordinator.algorithm = algo

    # Run reconstruction
    num_iterations = solver_cfg["num_epochs"] * num_subsets
    logging.info(
        f"Running {algorithm} for {num_iterations} iterations "
        f"({solver_cfg['num_epochs']} epochs)"
    )

    algo.run(num_iterations, verbose=True, callbacks=callbacks)
    clip_iters = getattr(algo, "_clip_iterations", 0)
    if clip_iters:
        logging.info(
            "Clipped voxel updates on %d iterations (scale %.3g, last limit %.3e)",
            clip_iters,
            getattr(algo, "max_update_scale", float("nan")),
            getattr(algo, "_last_clip_limit", float("nan")),
        )

    positivity_hits = getattr(algo, "_positivity_cap_hits", 0)
    if positivity_hits:
        logging.info(
            "Positivity cap active on %d iterations (theta %.3g)",
            positivity_hits,
            getattr(algo, "positivity_theta", float("nan")),
        )

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


def _get_solver_config(config):
    """Return stochastic solver configuration, supporting legacy 'svrg' blocks."""
    solver_cfg = config.get("stochastic")
    if solver_cfg is None:
        solver_cfg = config.get("svrg")
    if solver_cfg is None:
        raise KeyError(
            "Missing 'stochastic' configuration. "
            "Provide config['stochastic'] or legacy config['svrg']."
        )
    return solver_cfg


def _configure_update_clipping(algo, scale):
    """Attach clipping configuration to an ISTA instance."""
    if scale is None:
        algo.max_update_scale = None
        return

    try:
        scale = float(scale)
    except (TypeError, ValueError):
        logging.warning(
            "Ignoring invalid max_update_scale=%r (expected numeric).", scale
        )
        algo.max_update_scale = None
        return

    if scale <= 0:
        algo.max_update_scale = None
        return

    algo.max_update_scale = scale
    algo._clip_iterations = 0
    algo._last_clip_limit = None
    algo._last_clip_step = None


def _configure_positivity_cap(algo, solver_cfg, s_max_list, eta_min_list):
    """Attach positivity-cap parameters to an ISTA instance."""

    cap_cfg = (solver_cfg or {}).get("positivity_cap", {})
    enabled = bool(cap_cfg.get("enabled", True))
    theta = cap_cfg.get("theta", 0.1)

    s_list = list(s_max_list or [])
    eta_list = list(eta_min_list or [])

    if not enabled or not s_list or not eta_list:
        algo.positivity_cap_enabled = False
        algo.positivity_theta = float(theta)
        algo.positivity_s_max_per_subset = s_list
        algo.positivity_eta_min_per_subset = eta_list
        algo.positivity_s_max_global = max(s_list) if s_list else None
        algo.positivity_eta_min_global = min(eta_list) if eta_list else None
        algo._positivity_cap_hits = 0
        algo._last_positivity_cap = None
        algo._last_positivity_subset = None
        return

    algo.positivity_cap_enabled = True
    algo.positivity_theta = float(theta)
    algo.positivity_s_max_per_subset = s_list
    algo.positivity_eta_min_per_subset = eta_list
    algo.positivity_s_max_global = float(max(s_list)) if s_list else None
    algo.positivity_eta_min_global = float(min(eta_list)) if eta_list else None
    algo._positivity_cap_hits = 0
    algo._last_positivity_cap = None
    algo._last_positivity_subset = None


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

    solver_cfg = _get_solver_config(config)
    num_subsets = solver_cfg["num_subsets"]

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
        total_iterations = solver_cfg["num_epochs"] * num_subsets

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
            total_iterations=total_iterations,
        )

        # Initialize coordinator with existing additive term (skip first simulation)
        coordinator.initialize_with_additive(spect_data["additive"])

    data_fidelity_cfg = config.get("data_fidelity", {})
    eta_floor = data_fidelity_cfg.get("eta_floor", 1e-5)
    count_floor = data_fidelity_cfg.get("count_floor", 1e-5)

    # Partition once using CIL objectives
    # Returns LINEAR models and KL data functions for eta updates
    (
        kl_objectives,
        projectors,
        kl_data_functions,
        partition_indices,
        subset_sensitivity_max,
        subset_eta_min,
    ) = partition_data_once_cil(
        spect_data["acquisition_data"],
        spect_data["additive"],
        get_am,
        spect_data["initial_image"],
        num_subsets,
        coordinator=coordinator,
        eta_floor=eta_floor,
        count_floor=count_floor,
    )

    # Loop over beta values
    results = []
    for beta in config["rdp"]["beta_values"]:
        logging.info(f"--- Mode: {mode_no}, Beta: {beta} ---")
        output_dir_for_run = os.path.join(output_dir, f"mode{mode_no}/b{beta}")
        os.makedirs(output_dir_for_run, exist_ok=True)
        output_prefix = ""

        # Reset coordinator for this beta value to ensure a clean state.
        # This resets the iteration counter and re-initializes the cumulative
        # additive term from the original data before the reconstruction starts.
        if coordinator is not None:
            coordinator.initialize_with_additive(spect_data["additive"])
            coordinator.reset_iteration_counter()
            logging.info(f"Coordinator reset for beta={beta}")

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
            subset_sensitivity_max=subset_sensitivity_max,
            subset_eta_min=subset_eta_min,
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
    """Mode 4: STIR PSF residual correction (no SIMIND) - all beta values."""
    _log_mode_banner("MODE 4: STIR PSF residual correction")

    solver_cfg = _get_solver_config(config)
    num_subsets = solver_cfg["num_subsets"]

    # Build STIR acquisition model factories
    # Fast: no PSF
    get_am_fast = _make_stir_acq_model_factory(
        spect_data, config, use_psf=False, use_gaussian=False
    )
    # Accurate: with PSF
    get_am_psf = _make_stir_acq_model_factory(
        spect_data, config, use_psf=True, use_gaussian=False
    )

    # Create full-data STIR acquisition models for coordinator
    full_acq_data = spect_data["acquisition_data"]
    initial_image = spect_data["initial_image"]

    # Fast projector (no PSF)
    stir_fast_am = get_am_fast()
    stir_fast_am.set_up(full_acq_data, initial_image)

    # PSF projector (accurate)
    stir_psf_am = get_am_psf()
    stir_psf_am.set_up(full_acq_data, initial_image)

    # Create StirPsfCoordinator
    stir_dir = os.path.join(output_dir, "stir_psf_mode4")
    os.makedirs(stir_dir, exist_ok=True)

    correction_update_interval = (
        config["projector"]["correction_update_epochs"] * num_subsets
    )
    total_iterations = solver_cfg["num_epochs"] * num_subsets

    coordinator = StirPsfCoordinator(
        stir_psf_projector=stir_psf_am,
        stir_fast_projector=stir_fast_am,
        correction_update_interval=correction_update_interval,
        initial_additive=spect_data["additive"],
        output_dir=stir_dir,
        total_iterations=total_iterations,
    )

    # Initialize coordinator with existing additive term
    coordinator.initialize_with_additive(spect_data["additive"])

    data_fidelity_cfg = config.get("data_fidelity", {})
    eta_floor = data_fidelity_cfg.get("eta_floor", 1e-5)
    count_floor = data_fidelity_cfg.get("count_floor", 1e-5)

    # Partition once using CIL objectives with fast projector
    (
        kl_objectives,
        projectors,
        kl_data_functions,
        partition_indices,
        subset_sensitivity_max,
        subset_eta_min,
    ) = partition_data_once_cil(
        spect_data["acquisition_data"],
        spect_data["additive"],
        get_am_fast,
        spect_data["initial_image"],
        num_subsets,
        coordinator=coordinator,
        eta_floor=eta_floor,
        count_floor=count_floor,
    )

    # Loop over beta values
    results = []
    for beta in config["rdp"]["beta_values"]:
        logging.info(f"--- Mode: 4, Beta: {beta} ---")
        output_dir_for_run = os.path.join(output_dir, f"mode4/b{beta}")
        os.makedirs(output_dir_for_run, exist_ok=True)
        output_prefix = ""

        # Reset coordinator for this beta value
        coordinator.initialize_with_additive(spect_data["additive"])
        coordinator.reset_iteration_counter()
        logging.info(f"Coordinator reset for beta={beta}")

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
            subset_sensitivity_max=subset_sensitivity_max,
            subset_eta_min=subset_eta_min,
        )
        results.append(recon)

    return results


def run_mode_5(spect_data, config, output_dir):
    """Mode 5: Fast + SIMIND Geometric residual correction - all beta values."""
    return _run_mode_core(
        spect_data,
        config,
        output_dir,
        mode_no=5,
        mode_title="Fast + SIMIND Geometric residual",
        use_psf=False,
        use_gaussian=False,
        residual_correction=True,
        update_additive=False,
        simind_dir_suffix="mode5",
    )


def run_mode_6(spect_data, config, output_dir):
    """Mode 6: Accurate + SIMIND Geometric residual correction - all beta values."""
    return _run_mode_core(
        spect_data,
        config,
        output_dir,
        mode_no=6,
        mode_title="Accurate + SIMIND Geometric residual",
        use_psf=True,
        use_gaussian=True,
        residual_correction=True,
        update_additive=False,
        simind_dir_suffix="mode6",
    )


def run_mode_7(spect_data, config, output_dir):
    """Mode 7: Fast + SIMIND Full residual correction - all beta values."""
    return _run_mode_core(
        spect_data,
        config,
        output_dir,
        mode_no=7,
        mode_title="Fast + SIMIND Full residual",
        use_psf=False,
        use_gaussian=False,
        residual_correction=True,
        update_additive=True,
        simind_dir_suffix="mode7",
    )


def run_mode_8(spect_data, config, output_dir):
    """Mode 8: Accurate + SIMIND Full residual correction - all beta values."""
    return _run_mode_core(
        spect_data,
        config,
        output_dir,
        mode_no=8,
        mode_title="Accurate + SIMIND Full residual",
        use_psf=True,
        use_gaussian=True,
        residual_correction=True,
        update_additive=True,
        simind_dir_suffix="mode8",
    )


# ---------------------------
# Main
# ---------------------------


def main():
    """Main entry point."""
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.override)
    solver_cfg = _get_solver_config(config)

    configure_logging(config["output"]["verbose"])

    logging.info("=" * 80)
    logging.info("SPECT PSF Model Comparison")
    logging.info("=" * 80)
    logging.info(f"Config: {args.config}")
    logging.info(f"Data: {config['data_path']}")
    logging.info(f"Output: {args.output_path}")
    logging.info(f"Execution mode: {args.execution}")
    logging.info(f"Modes: {config['reconstruction']['modes']}")
    logging.info(f"Beta values: {config['rdp']['beta_values']}")
    logging.info(
        "Stochastic solver: %s - %d epochs, %d subsets",
        solver_cfg.get("algorithm", "SVRG").upper(),
        solver_cfg["num_epochs"],
        solver_cfg["num_subsets"],
    )
    logging.info(
        f"Energy window: "
        f"[{config['simind']['energy_lower']}, {config['simind']['energy_upper']}] keV"
    )
    logging.info("=" * 80)

    # Ensure output path exists for both modes
    os.makedirs(args.output_path, exist_ok=True)

    if args.execution == "cluster":
        run_cluster_sweep(args, config)
        return

    AcquisitionData.set_storage_scheme("memory")
    msg = MessageRedirector()

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
        8: run_mode_8,
    }

    # Run reconstructions - each mode handles all beta values internally
    total_start = time.time()

    for mode in config["reconstruction"]["modes"]:
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


def positivity_preserving_ista_update(self) -> None:
    """ISTA update with per-iteration positivity and magnitude caps."""

    gradient = self.f.gradient(self.x_old, out=self.gradient_update)

    try:
        raw_step = self.step_size_rule.get_step_size(self)
    except NameError as exc:  # pragma: no cover - defensive
        raise NameError(
            "`step_size` must be None, a real float or a StepSizeRule instance"
        ) from exc

    if self.preconditioner is not None:
        descent = self.preconditioner.apply(self, gradient)
    else:
        descent = gradient

    descent_norm = float(np.max(np.abs(get_array(descent))))
    step_size = raw_step

    clip_hit = False
    positivity_hit = False
    subset_idx = getattr(getattr(self, "f", None), "function_num", None)

    # Optional amplitude-based cap |Δx| <= scale * max(x)
    max_scale = getattr(self, "max_update_scale", None)
    if max_scale and max_scale > 0 and descent_norm > 0:
        reference = float(self.x.max())
        if reference > 0:
            limit = max_scale * reference
            scale_cap = limit / descent_norm
            if scale_cap > 0 and scale_cap < step_size:
                step_size = scale_cap
                clip_hit = True
                self._last_clip_limit = limit

    # Positivity-preserving cap α <= θ μ_min / (||d||∞ s_max)
    if (
        getattr(self, "positivity_cap_enabled", False)
        and descent_norm > 0
        and getattr(self, "positivity_theta", None) is not None
    ):
        theta = float(self.positivity_theta)
        s_max_list = getattr(self, "positivity_s_max_per_subset", None)
        eta_min_list = getattr(self, "positivity_eta_min_per_subset", None)
        mu_min = getattr(self, "positivity_eta_min_global", None)
        s_max = getattr(self, "positivity_s_max_global", None)

        if eta_min_list and s_max_list and subset_idx is not None:
            if 0 <= subset_idx < len(eta_min_list):
                mu_min = eta_min_list[subset_idx]
            if 0 <= subset_idx < len(s_max_list):
                s_max = s_max_list[subset_idx]

        if mu_min is not None and s_max is not None and mu_min > 0 and s_max > 0:
            positivity_cap = theta * mu_min / (descent_norm * s_max)
            if positivity_cap > 0 and positivity_cap < step_size:
                step_size = positivity_cap
                positivity_hit = True
                self._last_positivity_cap = positivity_cap
                self._last_positivity_subset = subset_idx

    if step_size < raw_step:
        if hasattr(self, "step_size_rule") and hasattr(
            self.step_size_rule, "current_step_size"
        ):
            self.step_size_rule.current_step_size = step_size
            if hasattr(self.step_size_rule, "min_step_size_seen"):
                self.step_size_rule.min_step_size_seen = min(
                    self.step_size_rule.min_step_size_seen, step_size
                )

    if clip_hit:
        self._clip_iterations = getattr(self, "_clip_iterations", 0) + 1
        self._last_clip_step = step_size

    if positivity_hit:
        self._positivity_cap_hits = getattr(self, "_positivity_cap_hits", 0) + 1
    else:
        self._last_positivity_cap = getattr(self, "_last_positivity_cap", step_size)

    self._last_step_size = step_size

    self.x_old.sapyb(1.0, descent, -step_size, out=self.x_old)

    M = self.x.max()
    self.x_old = self.x_old.maximum(-M)
    self.x_old = self.x_old.minimum(M)
    self.g.proximal(self.x_old, step_size, out=self.x)


ISTA.update = positivity_preserving_ista_update

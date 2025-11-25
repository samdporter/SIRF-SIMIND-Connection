"""
SVRG/ISTA solver wiring for PSF comparison.
"""

import logging
import os
from typing import Optional

import numpy as np
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import IndicatorBox, ScaledFunction
from cil.optimisation.utilities import Sampler
from RDP import RDP
from setr.cil_extensions.algorithms.algorithms import ista_update_step
from setr.cil_extensions.callbacks import (
    PrintObjectiveCallback,
    SavePreconditionerCallback,
)
from sirf.STIR import CudaRelativeDifferencePrior
from step_size_rules import (
    ArmijoAfterCorrectionStepSize,
    ArmijoTriggerCallback,
    LinearDecayStepSizeRule,
    SaveStepSizeHistoryCallback,
)

from sirf_simind_connection.utils.cil_partitioner import create_svrg_objective_with_rdp

from .callbacks import SaveImageCallback, SaveObjectiveCallback, UpdateEtaCallback
from .config import get_solver_config
from .preconditioning import build_preconditioner


ISTA.update = ista_update_step


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
    """
    solver_cfg = get_solver_config(config)
    algorithm = solver_cfg.get("algorithm", "SVRG").upper()
    data_fidelity_cfg = config.get("data_fidelity", {})
    eta_floor = data_fidelity_cfg.get("eta_floor", 1e-5)
    count_floor = data_fidelity_cfg.get("counts_floor", 1e-8)
    projector_cfg = config.get("projector", {})
    prefetch_initial_correction = bool(
        projector_cfg.get("prefetch_initial_correction", True)
    )

    logging.info("%s (CIL): %s, beta=%s", algorithm, output_prefix, beta)

    num_subsets = len(kl_objectives)

    rdp_prior: Optional[CudaRelativeDifferencePrior] = None
    if beta > 0:
        rdp_prior = CudaRelativeDifferencePrior()
        rdp_prior.set_penalisation_factor(beta)
        rdp_prior.set_gamma(config["rdp"].get("gamma", 1.0))
        rdp_prior.set_epsilon(config["rdp"].get("epsilon", 1e-6))
        rdp_prior.set_up(initial_image)
        prior = ScaledFunction(rdp_prior, 1)
    else:
        prior = False

    sampler = Sampler.random_without_replacement(num_subsets)
    update_interval = num_subsets
    if algorithm == "SVRG":
        snapshot_update_interval = solver_cfg.get(
            "snapshot_update_interval", update_interval
        )
    else:
        snapshot_update_interval = None

    total_objective = create_svrg_objective_with_rdp(
        kl_objectives,
        prior,
        sampler,
        snapshot_update_interval,
        initial_image,
        algorithm=algorithm,
    )

    g = IndicatorBox(lower=0, upper=np.inf)

    use_armijo_after_correction = coordinator is not None and solver_cfg.get(
        "use_armijo_after_correction", True
    )
    use_armijo = use_armijo_after_correction

    if use_armijo:
        step_size_rule = ArmijoAfterCorrectionStepSize(
            initial_step_size=solver_cfg["initial_step_size"],
            beta=solver_cfg.get("armijo_beta", 0.5),
            decay_rate=solver_cfg["relaxation_eta"],
            max_iter=solver_cfg.get("armijo_max_iter", 20),
            tol=solver_cfg.get("armijo_tol", 1e-4),
        )
        logging.info(
            "Using ArmijoAfterCorrectionStepSize (triggered only after residual updates)"
        )
    else:
        step_size_rule = LinearDecayStepSizeRule(
            solver_cfg["initial_step_size"],
            solver_cfg["relaxation_eta"],
        )
        logging.info("Using LinearDecayStepSizeRule")

    rdp_prior_for_hessian = RDP(
        initial_image.shape,
        gamma=config["rdp"].get("gamma", 1.0),
        eps=config["rdp"].get("epsilon", 1e-6),
    )
    rdp_prior_for_hessian.scale = beta

    preconditioner = build_preconditioner(
        config=config,
        projectors=projectors,
        kl_objectives=kl_objectives,
        initial_image=initial_image,
        num_subsets=num_subsets,
        beta=beta,
        rdp_prior=rdp_prior,
        rdp_prior_for_hessian=rdp_prior_for_hessian,
    )

    objective_csv_path = os.path.join(output_dir, f"{output_prefix}objective.csv")
    objective_logger = SaveObjectiveCallback(
        objective_csv_path,
        interval=update_interval,
        start_iteration=0,
        log_on_armijo=True,
    )

    image_prefix = os.path.join(output_dir, f"{output_prefix}image")
    image_logger = SaveImageCallback(image_prefix, interval=update_interval)

    step_size_logger = None
    if use_armijo:
        step_csv_path = os.path.join(output_dir, f"{output_prefix}step_sizes.csv")
        step_size_logger = SaveStepSizeHistoryCallback(
            step_csv_path, coordinator=coordinator
        )
        logging.info(
            "Configured SaveStepSizeHistoryCallback (logging to %s)", step_csv_path
        )

    eta_callback = None
    if coordinator is not None and kl_data_functions is not None:
        debug_eta_path = os.path.join(
            output_dir, f"{output_prefix}objective_eta_checks.csv"
        )
        df_cfg = config.get("data_fidelity", {})
        image_smoothing_fwhm = df_cfg.get("image_smoothing_fwhm", None)
        eta_callback = UpdateEtaCallback(
            coordinator,
            kl_data_functions,
            partition_indices,
            eta_floor=eta_floor,
            debug_objective_path=debug_eta_path,
            image_smoothing_fwhm=image_smoothing_fwhm,
        )
        logging.info("Configured UpdateEtaCallback for eta updates after corrections")

    armijo_trigger = None
    if use_armijo and coordinator is not None:
        armijo_trigger = ArmijoTriggerCallback(coordinator)
        logging.info(
            "Configured ArmijoTriggerCallback to request Armijo after corrections"
        )

    def _prefetch_corrections(stage_label, algorithm_instance, image):
        """Force a correction update before the next iteration and refresh eta/Armijo."""
        if not (
            prefetch_initial_correction
            and coordinator is not None
            and image is not None
            and algorithm_instance is not None
        ):
            return False

        try:
            coordinator.run_accurate_projection(image, force=True)
        except TypeError:
            coordinator.run_accurate_projection(image)
        except Exception as exc:
            logging.warning(
                "Skipping %s correction prefetch due to error: %s", stage_label, exc
            )
            return False

        if eta_callback is not None:
            eta_callback(algorithm_instance)
        if armijo_trigger is not None:
            armijo_trigger(algorithm_instance)

        logging.info(
            "Prefetched coordinator corrections before %s (iteration=%d)",
            stage_label,
            algorithm_instance.iteration,
        )
        return True

    callbacks = []

    if eta_callback is not None:
        callbacks.append(eta_callback)

    callbacks.extend(
        [
            PrintObjectiveCallback(interval=update_interval),
            objective_logger,
            image_logger,
        ]
    )

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

    if step_size_logger is not None:
        callbacks.append(step_size_logger)

    if armijo_trigger is not None:
        callbacks.append(armijo_trigger)

    algo = ISTA(
        initial=initial_image.clone(),
        f=total_objective,
        g=g,
        step_size=step_size_rule,
        update_objective_interval=update_interval,
        preconditioner=preconditioner,
    )

    if coordinator is not None:
        coordinator.algorithm = algo

    _prefetch_corrections("initial", algo, initial_image)

    num_iterations = solver_cfg["num_epochs"] * num_subsets
    logging.info(
        "Running %s for %d iterations (%d epochs)",
        algorithm,
        num_iterations,
        solver_cfg["num_epochs"],
    )

    algo.run(num_iterations, verbose=True, callbacks=callbacks)

    output_fname = os.path.join(output_dir, f"{output_prefix}_final.hv")
    algo.solution.write(output_fname)
    logging.info("Saved: %s", output_fname)

    return algo.solution

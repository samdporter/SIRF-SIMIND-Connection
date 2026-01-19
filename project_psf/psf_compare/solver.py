"""
SVRG/ISTA solver wiring for PSF comparison.
"""

import logging
import os
from typing import Optional

import numpy as np
from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import IndicatorBox
from cil.optimisation.utilities import Sampler
from RDP import RDP
from recon_core.cil_extensions.algorithms.algorithms import ista_update_step
from recon_core.cil_extensions.callbacks import (
    PrintObjectiveCallback,
    SavePreconditionerCallback,
)
from sirf.STIR import CudaRelativeDifferencePrior as RelativeDifferencePrior
from step_size_rules import (
    ArmijoAfterCorrectionStepSize,
    ArmijoPeriodicCallback,
    ArmijoTriggerCallback,
    LinearDecayStepSizeRule,
    SaveStepSizeHistoryCallback,
)

from sirf_simind_connection.utils.cil_partitioner import create_svrg_objective_with_rdp

from .callbacks import (
    SaveEffectiveObjectiveCallback,
    SaveImageCallback,
    SaveObjectiveCallback,
    UpdateEtaCallback,
)
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
    num_correction_cycles=1,
    restart_on_correction_reset=False,
    mask_image=None,
    restart_state=None,
    full_acquisition_data=None,
):
    """
    Run SVRG reconstruction with CIL objectives and RDP prior.
    """
    solver_cfg = get_solver_config(config)
    algorithm = solver_cfg.get("algorithm", "SVRG").upper()
    data_fidelity_cfg = config.get("data_fidelity", {})
    eta_floor = data_fidelity_cfg.get("eta_floor", 1e-5)
    data_fidelity_cfg.get("counts_floor", 1e-8)
    projector_cfg = config.get("projector", {})
    prefetch_initial_correction = bool(
        projector_cfg.get("prefetch_initial_correction", True)
    )

    logging.info("%s (CIL): %s, beta=%s", algorithm, output_prefix, beta)

    num_subsets = len(kl_objectives)
    num_cycles = max(1, int(num_correction_cycles))
    restart_iteration = 0
    if restart_state is not None:
        restart_iteration = max(int(getattr(restart_state, "iteration", 0)), 0)
        logging.info("Resuming reconstruction from iteration %d", restart_iteration)
        if getattr(restart_state, "image_path", None):
            logging.info("Restart image source: %s", restart_state.image_path)

    rdp_prior: Optional[RelativeDifferencePrior] = None
    if beta > 0:
        rdp_prior = RelativeDifferencePrior()
        rdp_prior.set_penalisation_factor(beta)
        rdp_prior.set_gamma(config["rdp"].get("gamma", 1.0))
        rdp_prior.set_epsilon(config["rdp"].get("epsilon", 1e-6))
        rdp_prior.set_up(initial_image)

        # Apply spatial penalty if enabled in config
        spatial_beta_cfg = config.get("rdp", {}).get("spatial_beta", {})
        if spatial_beta_cfg.get("enabled", False):
            if coordinator is None:
                logging.warning(
                    "spatial_beta enabled but no coordinator provided; "
                    "using scalar beta instead"
                )
            else:
                from .preconditioning import compute_spatial_penalty_map

                kappa_map = compute_spatial_penalty_map(
                    coordinator=coordinator,
                    initial_image=initial_image,
                    floor=spatial_beta_cfg.get("floor", 1e-6),
                    normalize=spatial_beta_cfg.get("normalize", False),
                )

                if kappa_map is not None:
                    # For SIRF RDP: set kappa directly (NOT squared!)
                    # The RDP formula includes κ_r * κ_{r+dr}, which gives ~kappa^2 for smooth maps
                    rdp_prior.set_kappa(kappa_map)
                    logging.info(
                        "Applied spatial penalty map (kappa) to SIRF RDP prior"
                    )

    sampler = Sampler.random_without_replacement(num_subsets)
    update_interval = num_subsets
    if algorithm == "SVRG":
        snapshot_update_interval = solver_cfg.get(
            "snapshot_update_interval", update_interval * 2
        )
    else:
        snapshot_update_interval = None

    if coordinator is not None and getattr(coordinator, "algorithm", None) is None:

        class _WarmStartAlgorithm:
            iteration = 0

        coordinator.algorithm = _WarmStartAlgorithm()

    total_objective = create_svrg_objective_with_rdp(
        kl_objectives,
        rdp_prior,
        sampler,
        snapshot_update_interval,
        initial_image,
        algorithm=algorithm,
    )

    g = IndicatorBox(lower=0, upper=np.inf)

    periodic_epoch_interval = solver_cfg.get("armijo_trigger_every_n_epochs")
    periodic_iteration_interval = None
    if periodic_epoch_interval is not None:
        try:
            periodic_epoch_interval = float(periodic_epoch_interval)
        except (TypeError, ValueError):
            logging.warning(
                "Invalid armijo_trigger_every_n_epochs=%s; ignoring periodic Armijo",
                periodic_epoch_interval,
            )
            periodic_epoch_interval = None
        else:
            if periodic_epoch_interval <= 0:
                logging.info(
                    "armijo_trigger_every_n_epochs=%s <= 0; periodic Armijo disabled",
                    periodic_epoch_interval,
                )
                periodic_epoch_interval = None
            else:
                periodic_iteration_interval = max(
                    int(round(periodic_epoch_interval * num_subsets)), 1
                )

    periodic_requests_armijo = periodic_iteration_interval is not None

    use_armijo_after_correction = coordinator is not None and solver_cfg.get(
        "use_armijo_after_correction", True
    )
    use_armijo = use_armijo_after_correction or periodic_requests_armijo

    if use_armijo:
        step_size_rule = ArmijoAfterCorrectionStepSize(
            initial_step_size=solver_cfg["initial_step_size"],
            beta=solver_cfg.get("armijo_beta", 0.5),
            decay_rate=solver_cfg["relaxation_eta"],
            max_iter=solver_cfg.get("armijo_max_iter", 20),
            tol=solver_cfg.get("armijo_tol", 1e-4),
        )
        if use_armijo_after_correction and periodic_requests_armijo:
            logging.info(
                "Using ArmijoAfterCorrectionStepSize (correction-triggered + periodic)"
            )
        elif use_armijo_after_correction:
            logging.info(
                "Using ArmijoAfterCorrectionStepSize (triggered only after corrections)"
            )
        else:
            logging.info("Using ArmijoAfterCorrectionStepSize (periodic trigger only)")
        if periodic_requests_armijo:
            # Ensure the very first iteration also uses an Armijo search.
            step_size_rule.trigger_armijo = True
            logging.info(
                "Periodic Armijo enabled: interval=%d iterations (~%.2f epochs)",
                periodic_iteration_interval,
                periodic_iteration_interval / num_subsets,
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

    # Apply spatial beta to torch RDP as well
    spatial_beta_cfg = config.get("rdp", {}).get("spatial_beta", {})
    if spatial_beta_cfg.get("enabled", False) and coordinator is not None:
        from .preconditioning import compute_spatial_penalty_map

        kappa_map = compute_spatial_penalty_map(
            coordinator=coordinator,
            initial_image=initial_image,
            floor=spatial_beta_cfg.get("floor", 1e-6),
            normalize=spatial_beta_cfg.get("normalize", False),
        )

        if kappa_map is not None:
            import torch

            kappa_arr = get_array(kappa_map)
            # For torch RDP: set spatial_beta = kappa^2 (squared)
            # This directly scales the penalty, unlike SIRF where kappa appears in neighbor products
            kappa_tensor = torch.from_numpy(kappa_arr).to(dtype=torch.float32)
            rdp_prior_for_hessian.spatial_beta = kappa_tensor**2
            logging.info(
                "Applied spatial penalty map (kappa^2) to torch RDP for Hessian"
            )

    preconditioner = build_preconditioner(
        config=config,
        projectors=projectors,
        kl_objectives=kl_objectives,
        initial_image=initial_image,
        num_subsets=num_subsets,
        beta=beta,
        rdp_prior=rdp_prior,
        rdp_prior_for_hessian=rdp_prior_for_hessian,
        mask_image=mask_image,
    )

    objective_csv_path = os.path.join(output_dir, f"{output_prefix}objective.csv")
    objective_logger = SaveObjectiveCallback(
        objective_csv_path,
        interval=update_interval,
        start_iteration=0,
        log_on_armijo=True,
    )
    if restart_state is not None and restart_state.objective_history:
        objective_logger.load_history(
            restart_state.objective_history, interval=update_interval
        )

    image_prefix = os.path.join(output_dir, f"{output_prefix}image")
    image_logger = SaveImageCallback(image_prefix, interval=update_interval)

    step_csv_path = os.path.join(output_dir, f"{output_prefix}step_sizes.csv")
    step_size_logger = SaveStepSizeHistoryCallback(
        step_csv_path, coordinator=coordinator
    )
    if restart_state is not None and restart_state.step_history:
        step_size_logger.load_history(restart_state.step_history)
    logging.info(
        "Configured SaveStepSizeHistoryCallback (logging to %s)", step_csv_path
    )

    restart_image = initial_image.clone() if restart_on_correction_reset else None

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
            restart_on_update=restart_on_correction_reset,
            restart_image=restart_image,
        )
        logging.info("Configured UpdateEtaCallback for eta updates after corrections")

    effective_obj_callback = None
    if coordinator is not None and full_acquisition_data is not None:
        effective_obj_csv_path = os.path.join(
            output_dir, f"{output_prefix}effective_objective.csv"
        )
        effective_obj_callback = SaveEffectiveObjectiveCallback(
            coordinator=coordinator,
            measured_data=full_acquisition_data,
            csv_path=effective_obj_csv_path,
        )
        # Support restart/resume
        if restart_state is not None:
            effective_csv = os.path.join(
                output_dir, f"{output_prefix}effective_objective.csv"
            )
            if os.path.exists(effective_csv):
                import csv as csv_module

                try:
                    with open(effective_csv, "r") as f:
                        reader = csv_module.DictReader(f)
                        effective_records = list(reader)
                        effective_obj_callback.load_history(effective_records)
                except Exception as exc:
                    logging.warning(
                        "Failed to load effective objective history: %s", exc
                    )

        logging.info(
            "Configured SaveEffectiveObjectiveCallback (logging to %s)",
            effective_obj_csv_path,
        )

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

    if effective_obj_callback is not None:
        callbacks.append(effective_obj_callback)

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

    periodic_callback = None
    if use_armijo and periodic_requests_armijo:
        periodic_callback = ArmijoPeriodicCallback(periodic_iteration_interval)

    callbacks.append(step_size_logger)

    if armijo_trigger is not None:
        callbacks.append(armijo_trigger)
    if periodic_callback is not None:
        callbacks.append(periodic_callback)

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
        if restart_iteration > 0 and hasattr(coordinator, "set_iteration_counter"):
            coordinator.set_iteration_counter(restart_iteration)

    if restart_iteration > 0:
        try:
            algo.iteration = restart_iteration
        except AttributeError:
            logging.warning(
                "Algorithm does not expose iteration attribute; restart iteration ignored"
            )

    base_epochs = solver_cfg["num_epochs"]
    total_epochs = base_epochs * num_cycles
    iterations_per_run = total_epochs * num_subsets
    start_iteration = restart_iteration
    target_iteration = start_iteration + iterations_per_run
    remaining_iterations = max(target_iteration - start_iteration, 0)
    if num_cycles > 1:
        logging.info(
            "Running %s for %d iterations (%d epochs x %d cycles; start=%d, stop=%d)",
            algorithm,
            remaining_iterations,
            base_epochs,
            num_cycles,
            start_iteration,
            target_iteration,
        )
    else:
        logging.info(
            "Running %s for %d iterations (%d epochs; start=%d, stop=%d)",
            algorithm,
            remaining_iterations,
            base_epochs,
            start_iteration,
            target_iteration,
        )
    if remaining_iterations > 0:
        _prefetch_corrections("initial", algo, initial_image)
    if remaining_iterations > 0:
        algo.run(remaining_iterations, verbose=True, callbacks=callbacks)
    else:
        logging.info(
            "All %d planned iterations already completed; skipping solver run",
            iterations_per_run,
        )

    output_fname = os.path.join(output_dir, f"{output_prefix}_final.hv")
    algo.solution.write(output_fname)
    logging.info("Saved: %s", output_fname)

    return algo.solution

"""
Mode dispatch and orchestration for PSF comparison.
"""

import gc
import logging
import os

from cil.optimisation.operators import CompositionOperator
from setr.cil_extensions.operators.blurring import GaussianBlurringOperator

from sirf_simind_connection.core.coordinator import (
    SimindCoordinator,
    StirPsfCoordinator,
)

from .config import get_solver_config
from .partitioning import partition_data_once_cil
from .preconditioning import _create_mask_from_attenuation
from .restart import load_restart_state
from .restart import restart_enabled as resume_runs_enabled
from .simind import create_simind_simulator
from .solver import run_svrg_with_prior_cil
from .spect_utils import get_spect_am


def _log_mode_banner(title: str):
    logging.info("=" * 80)
    logging.info(title)
    logging.info("=" * 80)


def _make_stir_acq_model_factory(
    spect_data, config, *, use_psf: bool, use_gaussian: bool
):
    """
    Return a factory function that builds STIR acquisition models.

    Returns:
        tuple: (get_am_func, blur_operator_or_none)
            - get_am_func: Factory that creates SIRF acquisition models (no Gaussian)
            - blur_operator_or_none: GaussianBlurringOperator if use_gaussian=True, else None

    The blur operator should be applied AFTER partitioning by wrapping each subset model
    in a CompositionOperator.
    """

    def get_am():
        res = None
        if use_psf:
            res = (
                config["resolution"]["stir_psf_params"][0],
                config["resolution"]["stir_psf_params"][1],
                False,
            )

        # Always create acquisition model WITHOUT image processor
        stir_am = get_spect_am(
            spect_data,
            res=res,
            keep_all_views_in_cache=True,
            gauss_fwhm=None,  # Don't use SIRF's image processor
            attenuation=True,
        )

        # Always set up the SIRF acquisition model
        stir_am.set_up(spect_data["acquisition_data"], spect_data["initial_image"])

        return stir_am

    # Create blur operator if needed (will be applied after partitioning)
    blur_op = None
    if use_gaussian:
        gauss_fwhm = config["resolution"]["psf_fwhm"]
        blur_op = GaussianBlurringOperator(
            gauss_fwhm, spect_data["initial_image"], backend="auto"
        )

    return get_am, blur_op


def _get_restart_strategy(config):
    """Return restart flag and number of reconstruction cycles."""

    projector_cfg = config.get("projector", {})
    restart_enabled = bool(
        projector_cfg.get("restart_reconstruction_on_correction_reset", False)
    )

    raw_cycles = projector_cfg.get("n_corrections", 1)
    try:
        configured_cycles = int(raw_cycles)
    except (TypeError, ValueError):
        logging.warning(
            "Invalid projector.n_corrections=%s. Defaulting to 1 cycle", raw_cycles
        )
        configured_cycles = 1

    configured_cycles = max(1, configured_cycles)

    if restart_enabled:
        logging.info(
            "Restart strategy: enabled=%s, cycles=%d",
            restart_enabled,
            configured_cycles,
        )

    return restart_enabled, configured_cycles


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

    solver_cfg = get_solver_config(config)
    num_subsets = solver_cfg["num_subsets"]
    mask_image = _create_mask_from_attenuation(spect_data.get("attenuation"), 0.05)
    restart_enabled, configured_cycles = _get_restart_strategy(config)
    resume_reconstruction = resume_runs_enabled(config)
    # All modes run for num_epochs * n_corrections (even without residual correction)
    num_correction_cycles = configured_cycles
    base_iterations = solver_cfg["num_epochs"] * num_subsets
    get_am, blur_op = _make_stir_acq_model_factory(
        spect_data,
        config,
        use_psf=use_psf,
        use_gaussian=use_gaussian,
    )

    data_fidelity_cfg = config.get("data_fidelity", {})
    eta_floor = data_fidelity_cfg.get("eta_floor", 1e-5)
    count_floor = data_fidelity_cfg.get("count_floor", 1e-5)

    full_acq_data = spect_data["acquisition_data"]
    initial_image = spect_data["initial_image"]

    # Prepare SIMIND simulator if needed (reused across betas)
    simind_sim = None
    if residual_correction or update_additive:
        simind_dir = os.path.join(output_dir, f"simind_{simind_dir_suffix}")
        os.makedirs(simind_dir, exist_ok=True)
        simind_sim = create_simind_simulator(config, spect_data, simind_dir)

    results = []
    for beta in config["rdp"]["beta_values"]:
        logging.info("--- Mode: %s, Beta: %s ---", mode_no, beta)
        output_dir_for_run = os.path.join(output_dir, f"mode{mode_no}/b{beta}")
        os.makedirs(output_dir_for_run, exist_ok=True)
        output_prefix = ""
        restart_state = None
        if resume_reconstruction:
            restart_state = load_restart_state(
                config,
                output_dir_for_run,
                f"{output_prefix}image",
            )
        if restart_state is not None:
            run_initial_image = restart_state.image.clone()
        else:
            run_initial_image = spect_data["initial_image"]

        # Create fresh coordinator and acquisition models for each beta
        coordinator = None
        linear_am = None
        stir_am = None

        if residual_correction or update_additive:
            beta_simind_dir = os.path.join(output_dir_for_run, "simind_corrections")
            os.makedirs(beta_simind_dir, exist_ok=True)

            correction_update_interval = base_iterations
            total_iterations = base_iterations * num_correction_cycles

            linear_am = get_am()

            if residual_correction and update_additive:
                stir_am = get_am()

            coordinator = SimindCoordinator(
                simind_simulator=simind_sim,
                num_subsets=num_subsets,
                correction_update_interval=correction_update_interval,
                residual_correction=residual_correction,
                update_additive=update_additive,
                linear_acquisition_model=linear_am,
                stir_acquisition_model=stir_am,
                output_dir=beta_simind_dir,
                total_iterations=total_iterations,
                mask_image=mask_image,
            )

            coordinator.initialize_with_additive(spect_data["additive"])
            coordinator.reset_iteration_counter()
            logging.info(
                "Created fresh coordinator for beta=%s, output_dir=%s",
                beta,
                beta_simind_dir,
            )

        # Create fresh partitioning for each beta
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
            attenuation_map=spect_data["attenuation"],
            blur_operator=blur_op,
        )

        recon = run_svrg_with_prior_cil(
            kl_objectives,
            projectors,
            run_initial_image,
            beta,
            config,
            output_prefix,
            output_dir_for_run,
            coordinator=coordinator,
            kl_data_functions=kl_data_functions,
            partition_indices=partition_indices,
            subset_sensitivity_max=subset_sensitivity_max,
            subset_eta_min=subset_eta_min,
            mask_image=mask_image,
            num_correction_cycles=num_correction_cycles,
            restart_on_correction_reset=restart_enabled and coordinator is not None,
            restart_state=restart_state,
        )
        results.append(recon)

        # Explicit cleanup after each beta
        logging.info("Cleaning up memory after beta=%s", beta)
        del kl_objectives
        del projectors
        del kl_data_functions
        del coordinator
        del linear_am
        del stir_am
        gc.collect()

    return results


def _run_stir_psf_residual_mode(
    spect_data,
    config,
    output_dir,
    *,
    mode_no: int,
    mode_title: str,
    baseline_use_psf: bool,
    baseline_use_gaussian: bool,
    accurate_use_psf: bool,
    accurate_use_gaussian: bool,
):
    """
    Shared implementation for STIR PSF residual correction modes.
    """
    _log_mode_banner(f"MODE {mode_no}: {mode_title}")

    solver_cfg = get_solver_config(config)
    num_subsets = solver_cfg["num_subsets"]
    mask_image = _create_mask_from_attenuation(spect_data.get("attenuation"), 0.05)
    restart_enabled, configured_cycles = _get_restart_strategy(config)
    # All modes run for num_epochs * n_corrections
    num_correction_cycles = configured_cycles
    base_iterations = solver_cfg["num_epochs"] * num_subsets

    get_am_baseline, blur_op_baseline = _make_stir_acq_model_factory(
        spect_data,
        config,
        use_psf=baseline_use_psf,
        use_gaussian=baseline_use_gaussian,
    )
    get_am_accurate, blur_op_accurate = _make_stir_acq_model_factory(
        spect_data,
        config,
        use_psf=accurate_use_psf,
        use_gaussian=accurate_use_gaussian,
    )

    full_acq_data = spect_data["acquisition_data"]
    initial_image = spect_data["initial_image"]

    data_fidelity_cfg = config.get("data_fidelity", {})
    eta_floor = data_fidelity_cfg.get("eta_floor", 1e-5)
    count_floor = data_fidelity_cfg.get("count_floor", 1e-5)

    results = []
    for beta in config["rdp"]["beta_values"]:
        logging.info("--- Mode: %s, Beta: %s ---", mode_no, beta)
        output_dir_for_run = os.path.join(output_dir, f"mode{mode_no}/b{beta}")
        os.makedirs(output_dir_for_run, exist_ok=True)
        output_prefix = ""
        restart_state = None
        if resume_runs_enabled(config):
            restart_state = load_restart_state(
                config,
                output_dir_for_run,
                f"{output_prefix}image",
            )
        if restart_state is not None:
            run_initial_image = restart_state.image.clone()
        else:
            run_initial_image = spect_data["initial_image"]

        # Create fresh acquisition models and coordinator for each beta
        stir_baseline_am = get_am_baseline()
        stir_accurate_am = get_am_accurate()

        # Compose with blur operators if needed (for coordinator's full-data models)
        if blur_op_baseline is not None:
            stir_baseline_am = CompositionOperator(stir_baseline_am, blur_op_baseline)
        if blur_op_accurate is not None:
            stir_accurate_am = CompositionOperator(stir_accurate_am, blur_op_accurate)

        beta_stir_dir = os.path.join(output_dir_for_run, "stir_psf_corrections")
        os.makedirs(beta_stir_dir, exist_ok=True)

        correction_update_interval = base_iterations
        total_iterations = base_iterations * num_correction_cycles

        coordinator = StirPsfCoordinator(
            stir_psf_projector=stir_accurate_am,
            stir_fast_projector=stir_baseline_am,
            correction_update_interval=correction_update_interval,
            initial_additive=spect_data["additive"],
            output_dir=beta_stir_dir,
            total_iterations=total_iterations,
            mask_image=mask_image,
        )

        coordinator.initialize_with_additive(spect_data["additive"])
        coordinator.reset_iteration_counter()
        logging.info(
            "Created fresh coordinator for beta=%s, output_dir=%s", beta, beta_stir_dir
        )

        # Create fresh partitioning for each beta
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
            get_am_baseline,
            spect_data["initial_image"],
            num_subsets,
            coordinator=coordinator,
            eta_floor=eta_floor,
            count_floor=count_floor,
            attenuation_map=spect_data["attenuation"],
            blur_operator=blur_op_baseline,
        )

        recon = run_svrg_with_prior_cil(
            kl_objectives,
            projectors,
            run_initial_image,
            beta,
            config,
            output_prefix,
            output_dir_for_run,
            coordinator=coordinator,
            kl_data_functions=kl_data_functions,
            partition_indices=partition_indices,
            subset_sensitivity_max=subset_sensitivity_max,
            subset_eta_min=subset_eta_min,
            mask_image=mask_image,
            num_correction_cycles=num_correction_cycles,
            restart_on_correction_reset=restart_enabled,
            restart_state=restart_state,
        )
        results.append(recon)

        # Explicit cleanup after each beta
        logging.info("Cleaning up memory after beta=%s", beta)
        del kl_objectives
        del projectors
        del kl_data_functions
        del coordinator
        del stir_baseline_am
        del stir_accurate_am
        gc.collect()

    return results


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
    return _run_stir_psf_residual_mode(
        spect_data,
        config,
        output_dir,
        mode_no=4,
        mode_title="STIR PSF residual correction",
        baseline_use_psf=False,
        baseline_use_gaussian=False,
        accurate_use_psf=True,
        accurate_use_gaussian=True,
    )


def run_mode_5(spect_data, config, output_dir):
    """Mode 5: PSF (no brems) with STIR PSF+brems residual - all beta values."""
    return _run_stir_psf_residual_mode(
        spect_data,
        config,
        output_dir,
        mode_no=5,
        mode_title="PSF (no brems) with STIR PSF+brems residual",
        baseline_use_psf=True,
        baseline_use_gaussian=False,
        accurate_use_psf=True,
        accurate_use_gaussian=True,
    )


def run_mode_6(spect_data, config, output_dir):
    """Mode 6: Fast + SIMIND Geometric residual correction - all beta values."""
    return _run_mode_core(
        spect_data,
        config,
        output_dir,
        mode_no=6,
        mode_title="Fast + SIMIND Geometric residual",
        use_psf=False,
        use_gaussian=False,
        residual_correction=True,
        update_additive=False,
        simind_dir_suffix="mode6",
    )


def run_mode_7(spect_data, config, output_dir):
    """Mode 7: Accurate (no brems) + SIMIND Geometric residual correction - all beta values."""
    return _run_mode_core(
        spect_data,
        config,
        output_dir,
        mode_no=7,
        mode_title="Accurate (no brems) + SIMIND Geometric residual",
        use_psf=True,
        use_gaussian=False,
        residual_correction=True,
        update_additive=False,
        simind_dir_suffix="mode7",
    )


def run_mode_8(spect_data, config, output_dir):
    """Mode 8: Accurate (with brems) + SIMIND Geometric residual - all beta values."""
    return _run_mode_core(
        spect_data,
        config,
        output_dir,
        mode_no=8,
        mode_title="Accurate (with brems) + SIMIND Geometric residual",
        use_psf=True,
        use_gaussian=True,
        residual_correction=True,
        update_additive=False,
        simind_dir_suffix="mode8",
    )


def run_mode_9(spect_data, config, output_dir):
    """Mode 9: Fast + SIMIND Full residual correction - all beta values."""
    return _run_mode_core(
        spect_data,
        config,
        output_dir,
        mode_no=9,
        mode_title="Fast + SIMIND Full residual",
        use_psf=False,
        use_gaussian=False,
        residual_correction=True,
        update_additive=True,
        simind_dir_suffix="mode9",
    )


def run_mode_10(spect_data, config, output_dir):
    """Mode 10: Accurate (no brems) + SIMIND Full residual - all beta values."""
    return _run_mode_core(
        spect_data,
        config,
        output_dir,
        mode_no=10,
        mode_title="Accurate (no brems) + SIMIND Full residual",
        use_psf=True,
        use_gaussian=False,
        residual_correction=True,
        update_additive=True,
        simind_dir_suffix="mode10",
    )


def run_mode_11(spect_data, config, output_dir):
    """Mode 11: Accurate (with brems) + SIMIND Full residual - all beta values."""
    return _run_mode_core(
        spect_data,
        config,
        output_dir,
        mode_no=11,
        mode_title="Accurate (with brems) + SIMIND Full residual",
        use_psf=True,
        use_gaussian=True,
        residual_correction=True,
        update_additive=True,
        simind_dir_suffix="mode11",
    )

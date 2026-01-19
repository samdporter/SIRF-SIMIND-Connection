"""
Preconditioner construction utilities.
"""

import logging
from typing import Any, Dict

import numpy as np
from recon_core.cil_extensions.preconditioners import (
    BSREMPreconditioner,
    ImageFunctionPreconditioner,
    LehmerMeanPreconditioner,
)

from sirf_simind_connection.utils import get_array


def compute_sensitivity_inverse(projectors):
    """
    Compute the inverse of the sensitivity image for BSREM preconditioner.
    The sensitivity image is the sum of back-projections of ones from each subset.
    """
    sens = projectors[0].domain_geometry().get_uniform_copy(0)

    for proj in projectors:
        ones = proj.range_geometry().get_uniform_copy(1)
        sens += proj.adjoint(ones)

    sens = sens.maximum(0)
    sens_arr = get_array(sens)
    s_inv = np.reciprocal(sens_arr, where=sens_arr > 0)
    sens.fill(s_inv)
    sens = sens.minimum(1e3)
    return sens


def compute_spatial_penalty_map(
    coordinator,
    initial_image,
    floor=1e-6,
    normalize=False,
):
    """
    Compute spatially varying penalty strength kappa(x) for RDP.

    Formula: kappa(x) = sqrt(A^T D[1/A(x_init)] A 1)

    This accounts for local sensitivity variations in the forward model,
    applying stronger regularization where the projector is more sensitive.

    Args:
        coordinator (Coordinator): Coordinator with accurate projector.
                                   Must have linear_acquisition_model set.
        initial_image (ImageData): Initial activity estimate x_init.
        floor (float): Minimum value for 1/A(x_init) to avoid division by zero.
        normalize (bool): If True, normalize kappa to have mean=1 (preserves beta scale).

    Returns:
        ImageData: Spatial penalty map kappa(x), same geometry as initial_image.
                   Returns None if coordinator is None or doesn't support accurate projection.

    Raises:
        ValueError: If coordinator.linear_acquisition_model is not set.
    """
    if coordinator is None:
        logging.warning(
            "No coordinator provided; cannot compute spatial penalty map. "
            "Using scalar beta instead."
        )
        return None

    # Get accurate projector from coordinator
    if not hasattr(coordinator, "linear_acquisition_model"):
        logging.warning(
            "Coordinator does not have linear_acquisition_model; "
            "cannot compute spatial penalty map. Using scalar beta instead."
        )
        return None

    accurate_am = coordinator.linear_acquisition_model
    if accurate_am is None:
        raise ValueError(
            "Coordinator.linear_acquisition_model is None. "
            "Must be set before computing spatial penalty map."
        )

    logging.info("Computing spatial penalty map kappa(x) using accurate projector...")

    # Step 1: Forward projection A(x_init)
    logging.info("  Step 1/4: Computing A(x_init)...")
    # Ensure full-data projection (num_subsets=1, subset_num=0)
    if hasattr(accurate_am, "num_subsets"):
        accurate_am.num_subsets = 1
        accurate_am.subset_num = 0

    fwd_proj = accurate_am.direct(initial_image)

    # Step 2: Compute D[1/A(x_init)] = element-wise reciprocal with floor
    logging.info("  Step 2/4: Computing D[1/A(x_init)] with floor=%g...", floor)
    fwd_arr = get_array(fwd_proj)

    # Apply floor to avoid division by zero
    fwd_arr_safe = np.maximum(fwd_arr, floor)
    inv_fwd_arr = 1.0 / fwd_arr_safe

    inv_fwd = fwd_proj.clone()
    inv_fwd.fill(inv_fwd_arr)

    # Step 3: Compute A 1 (forward projection of uniform image)
    logging.info("  Step 3/4: Computing A(1)...")
    ones_image = initial_image.get_uniform_copy(1.0)
    fwd_ones = accurate_am.direct(ones_image)

    # Step 4: Apply diagonal D and backproject: A^T (D * A(1))
    logging.info("  Step 4/4: Computing A^T D[1/A(x_init)] A(1)...")
    weighted_fwd = fwd_ones * inv_fwd  # Element-wise multiplication
    kappa_squared = accurate_am.adjoint(weighted_fwd)

    # Take square root to get kappa
    kappa_squared_arr = get_array(kappa_squared)
    kappa_arr = np.sqrt(np.maximum(kappa_squared_arr, 0))  # Ensure non-negative

    kappa = initial_image.clone()
    kappa.fill(kappa_arr)

    # Optional normalization
    if normalize:
        mean_kappa = kappa_arr.mean()
        if mean_kappa > 0:
            kappa_arr_norm = kappa_arr / mean_kappa
            kappa.fill(kappa_arr_norm)
            logging.info("  Normalized kappa to mean=1 (original mean=%g)", mean_kappa)

    # Log statistics
    logging.info(
        "Spatial penalty map computed: min=%g, max=%g, mean=%g",
        kappa_arr.min(),
        kappa_arr.max(),
        kappa_arr.mean(),
    )

    return kappa


def _normalise_preconditioner_cfg(precond_cfg: Any) -> Dict[str, Any]:
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


def _create_mask_from_attenuation(attenuation_map, threshold=0.05):
    """Create a binary mask image from attenuation map using given threshold."""
    if attenuation_map is None:
        return None
    mask = attenuation_map.clone()
    mask_arr = (get_array(attenuation_map) > threshold).astype(np.float32)
    mask.fill(mask_arr)
    return mask


def build_preconditioner(
    config,
    projectors,
    kl_objectives,
    initial_image,
    num_subsets,
    beta,
    rdp_prior,
    rdp_prior_for_hessian,
    mask_image=None,
):
    """
    Build the preconditioner:
    - If beta == 0: return BSREM preconditioner
    - If beta > 0: return Lehmer(BSREM, prior inv-hessian)
    """
    precond_cfg = _normalise_preconditioner_cfg(config.get("preconditioner"))
    mask_precond = precond_cfg.get("mask_with_body_mask", True)
    bsrem_cfg = precond_cfg.get("bsrem", {})
    bsrem_update_interval = bsrem_cfg.get(
        "update_interval", precond_cfg.get("update_interval", num_subsets)
    )
    bsrem_freeze_iter = bsrem_cfg.get(
        "freeze_iter", precond_cfg.get("freeze_iter", np.inf)
    )
    bsrem_epsilon = bsrem_cfg.get("epsilon", precond_cfg.get("epsilon", 0.0))
    bsrem_smooth = bsrem_cfg.get("smooth", False)
    bsrem_smoothing_fwhm = tuple(bsrem_cfg.get("smoothing_fwhm", (10, 10, 10)))
    bsrem_max_value = _ensure_numeric(
        bsrem_cfg.get("max_value", precond_cfg.get("max_value", initial_image.max())),
        initial_image.max(),
    )

    s_inv = compute_sensitivity_inverse(projectors)
    if mask_precond and mask_image is not None:
        s_inv *= mask_image
    if mask_image is not None:
        s_inv *= mask_image
    bsrem_precond = BSREMPreconditioner(
        s_inv,
        update_interval=bsrem_update_interval,
        freeze_iter=bsrem_freeze_iter,
        epsilon=bsrem_epsilon,
        smooth=bsrem_smooth,
        max_val=bsrem_max_value,
        smoothing_fwhm=bsrem_smoothing_fwhm,
    )

    fwhm = bsrem_cfg.get("gaussian_fwhm", precond_cfg.get("gaussian_fwhm"))
    if bsrem_smooth and fwhm is not None and hasattr(bsrem_precond, "gaussian"):
        try:
            bsrem_precond.gaussian.set_fwhms(tuple(fwhm))
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.warning("Unable to set BSREM Gaussian FWHM to %s: %s", fwhm, exc)

    if rdp_prior is None or beta <= 0:
        return bsrem_precond

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
        arr = rdp_prior_for_hessian.inv_diag_hessian(get_array(image))
        result = image.clone()
        result.fill(arr)
        if mask_precond and mask_image is not None:
            result *= mask_image
        return result

    prior_precond = ImageFunctionPreconditioner(
        rdp_inv_hessian,
        update_interval=prior_update_interval,
        freeze_iter=prior_freeze_iter,
        epsilon=prior_epsilon,
        max_value=prior_max_value,
    )

    lehmer_cfg = precond_cfg.get("lehmer", {})
    p_value = lehmer_cfg.get("p", 1e-1)
    lehmer_epsilon = lehmer_cfg.get("epsilon", rdp_cfg.get("lehmer_epsilon", 1e-8))
    combined_update_interval = max(
        val for val in (bsrem_update_interval, prior_update_interval) if val is not None
    )
    combined_freeze_iter = min(
        val for val in (bsrem_freeze_iter, prior_freeze_iter) if val is not None
    )

    return LehmerMeanPreconditioner(
        [bsrem_precond, prior_precond],
        p=p_value,
        epsilon=lehmer_epsilon,
        update_interval=combined_update_interval,
        freeze_iter=combined_freeze_iter,
        scales=[1.0, prior_cfg.get("scale", 1.0)],
    )

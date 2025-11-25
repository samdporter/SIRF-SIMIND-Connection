"""
Preconditioner construction utilities.
"""

import logging
from typing import Any, Dict

import numpy as np
from setr.cil_extensions.preconditioners import (
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
):
    """
    Build the preconditioner:
    - If beta == 0: return BSREM preconditioner
    - If beta > 0: return Lehmer(BSREM, prior inv-hessian)
    """
    precond_cfg = _normalise_preconditioner_cfg(config.get("preconditioner"))
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

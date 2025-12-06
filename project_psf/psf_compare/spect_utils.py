"""
Local helpers for loading SPECT data and constructing STIR acquisition models.

These functions replace the subset we previously pulled from ``setr.utils`` so
that the PSF comparison workflow no longer depends on the ``setr`` package (and
its heavy import graph) at runtime.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sirf.Reg import NiftiImageData3DDisplacement
from sirf.STIR import (
    AcquisitionData,
    AcquisitionModelUsingMatrix,
    ImageData,
    SeparableGaussianImageFilter,
    SPECTUBMatrix,
)


logger = logging.getLogger(__name__)


def load_zoom_factors(spect_dir: str) -> Tuple[float, float, float]:
    """
    Load zoom factors saved by the pre-processing pipeline.

    Args:
        spect_dir: Directory containing ``spect_to_pet_zoom_factors.txt``.
    """
    zoom_file_path = os.path.join(spect_dir, "spect_to_pet_zoom_factors.txt")
    if not os.path.exists(zoom_file_path):
        raise FileNotFoundError(f"Zoom factors file not found: {zoom_file_path}")

    with open(zoom_file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid zoom factor line '{line.strip()}' in {zoom_file_path}"
                )
            return (float(parts[0]), float(parts[1]), float(parts[2]))

    raise ValueError(f"No zoom factors found in file {zoom_file_path}")


def get_spect_data(path: str, load_sinos: bool = True) -> Dict[str, object]:
    """
    Load SPECT sinograms, images, and registration artefacts from ``path``.
    """
    spect_data: Dict[str, object] = {}

    if load_sinos:
        prompts_path = os.path.join(path, "peak.hs")
        spect_data["acquisition_data"] = AcquisitionData(prompts_path)
        try:
            scatter_path = os.path.join(path, "scatter_dl.hs")
            spect_data["additive"] = AcquisitionData(scatter_path)
        except Exception as exc:
            logger.warning(
                "No scatter data found (%s). Filling additive term with zeros.", exc
            )
            additive = spect_data["acquisition_data"].clone()
            additive.fill(0)
            spect_data["additive"] = additive

    attenuation = ImageData(os.path.join(path, "umap_zoomed.hv"))
    attn_arr = attenuation.as_array()
    attenuation.fill(np.flip(attn_arr, axis=-1))
    spect_data["attenuation"] = attenuation

    template_path = os.path.join(path, "template_image.hv")
    try:
        template_img = ImageData(template_path)
    except Exception as exc:
        raise RuntimeError(f"Unable to load SPECT template image: {exc}") from exc
    spect_data["template_image"] = template_img

    initial_path = os.path.join(path, "initial_image.hv")
    try:
        spect_data["initial_image"] = ImageData(initial_path).maximum(0)
    except Exception as exc:
        logger.warning(
            "No SPECT initial image found (%s). Using uniform template copy.", exc
        )
        spect_data["initial_image"] = template_img.get_uniform_copy(1)

    displacement_files = [
        "spect2pet_zoom_nonrigid.nii",
        "spect2pet_zoom_rigid.nii",
        "spect2pet.nii",
    ]
    spect_data["displacement"] = _load_first_displacement(path, displacement_files)

    no_zoom_files = [
        "spect2pet_nozoom_nonrigid.nii",
        "spect2pet_nozoom_rigid.nii",
        "spect2pet_nozoom.nii",
    ]
    spect_data["no_zoom_displacement"] = _load_first_displacement(path, no_zoom_files)

    try:
        spect_data["zoom_factors"] = load_zoom_factors(path)
    except Exception as exc:
        logger.warning("No SPECT zoom factors found (%s).", exc)
        spect_data["zoom_factors"] = None

    return spect_data


def _load_first_displacement(
    base_dir: str, filenames: Sequence[str]
) -> Optional[NiftiImageData3DDisplacement]:
    """Helper to load the first available displacement file."""
    for name in filenames:
        full_path = os.path.join(base_dir, name)
        if os.path.exists(full_path):
            try:
                return NiftiImageData3DDisplacement(full_path)
            except Exception as exc:  # pragma: no cover - diagnostic path
                logger.warning(
                    "Failed to load displacement field from %s (%s).", full_path, exc
                )
                return None
    logger.warning(
        "No displacement field found among %s (directory=%s)", filenames, base_dir
    )
    return None


def get_spect_am(
    spect_data: Dict[str, object],
    res: Optional[Sequence[float]] = None,
    keep_all_views_in_cache: bool = True,
    gauss_fwhm: Optional[Sequence[float]] = None,
    attenuation: bool = True,
):
    """
    Build a STIR SPECT acquisition model equivalent to ``setr.utils.get_spect_am``.
    """
    matrix = SPECTUBMatrix()
    matrix.set_keep_all_views_in_cache(keep_all_views_in_cache)
    if attenuation:
        try:
            matrix.set_attenuation_image(spect_data["attenuation"])
        except KeyError:
            logger.warning(
                "Spect data missing attenuation image; proceeding without it"
            )
    if res:
        matrix.set_resolution_model(*res)

    spect_am = AcquisitionModelUsingMatrix(matrix)
    if gauss_fwhm:
        psf = SeparableGaussianImageFilter()
        psf.set_fwhms(gauss_fwhm)
        spect_am.set_image_data_processor(psf)
    return spect_am

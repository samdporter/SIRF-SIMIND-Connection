"""
Centralized import helpers for SIRF/STIR backends.

This module provides reusable functions for importing backend types with
proper fallback handling. This eliminates the need for duplicate try/except
import blocks scattered throughout the codebase.
"""

from typing import Tuple, Type


def get_sirf_types() -> Tuple[Type, Type, bool]:
    """Import SIRF types with fallback.

    Returns:
        Tuple of (ImageData, AcquisitionData, SIRF_AVAILABLE)
        If SIRF is not available, returns (type(None), type(None), False)
    """
    try:
        from sirf.STIR import AcquisitionData, ImageData

        return ImageData, AcquisitionData, True
    except ImportError:
        return type(None), type(None), False


def get_stir_types() -> Tuple[Type, Type, bool]:
    """Import STIR Python types with fallback.

    Returns:
        Tuple of (FloatVoxelsOnCartesianGrid, ProjData, STIR_AVAILABLE)
        If STIR is not available, returns (type(None), type(None), False)
    """
    try:
        import stir

        return stir.FloatVoxelsOnCartesianGrid, stir.ProjData, True
    except ImportError:
        return type(None), type(None), False


def get_backend_types() -> Tuple[Type, Type, Type, Type, str]:
    """Import backend-agnostic types (tries SIRF first, then STIR).

    Returns:
        Tuple of (ImageData, AcquisitionData, ImageType, AcqType, backend_name)
        where ImageType/AcqType are the native backend types

    Raises:
        ImportError: If neither SIRF nor STIR is available
    """
    # Try SIRF first
    ImageData, AcquisitionData, sirf_available = get_sirf_types()
    if sirf_available:
        return ImageData, AcquisitionData, ImageData, AcquisitionData, "sirf"

    # Fall back to STIR
    StirImage, StirProj, stir_available = get_stir_types()
    if stir_available:
        return StirImage, StirProj, StirImage, StirProj, "stir"

    raise ImportError(
        "Neither SIRF nor STIR Python is available. "
        "Please install one of:\n"
        "  - SIRF (sirf.STIR)\n"
        "  - STIR Python (stir + stirextra)"
    )


__all__ = [
    "get_sirf_types",
    "get_stir_types",
    "get_backend_types",
]

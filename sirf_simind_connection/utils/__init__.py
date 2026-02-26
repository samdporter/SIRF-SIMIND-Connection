"""
Grab-bag of general utilities shared across modules.
They are re-exported here for a single, easy import path.
"""

import contextlib
import importlib


# Lazy imports to avoid SIRF dependencies in CI
def __getattr__(name):
    if name in (
        "interfile_numpy",
        "io_utils",
        "simind_utils",
        "stir_utils",
        "sirf_stir_utils",
    ):
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def get_array(obj):
    """Return NumPy array from SIRF or STIR object.

    This function works with both SIRF and STIR Python backends:
    - SIRF: Uses .asarray() or .as_array()
    - STIR: Uses stirextra.to_numpy()
    - Backend wrappers: Uses .as_array() method

    Args:
        obj: Image or acquisition data object (SIRF, STIR, or wrapped)

    Returns:
        np.ndarray: NumPy array view of the data

    Raises:
        AttributeError: If object cannot be converted to array
    """
    # Check if it's a wrapped backend object
    if hasattr(obj, "as_array") and callable(obj.as_array):
        return obj.as_array()

    # Try SIRF native methods
    if hasattr(obj, "asarray"):
        try:
            return obj.asarray()
        except Exception:
            return obj.as_array()  # Fallback to as_array if asarray fails
    if hasattr(obj, "as_array"):
        return obj.as_array()

    # Try STIR native conversion
    with contextlib.suppress(ImportError, TypeError, AttributeError):
        import stirextra

        return stirextra.to_numpy(obj)
    raise AttributeError(
        f"Cannot convert {type(obj)} to numpy array. "
        f"Object must have asarray(), as_array() method, or be a STIR object."
    )


def to_projdata_in_memory(proj_data):
    """Convert ProjData to ProjDataInMemory for arithmetic operations.

    STIR's ProjData objects don't support arithmetic operations directly.
    This function converts them to ProjDataInMemory which does.

    Args:
        proj_data: STIR ProjData object or SIRF AcquisitionData

    Returns:
        ProjDataInMemory object (STIR) or cloned AcquisitionData (SIRF)

    Note:
        For SIRF AcquisitionData, this returns a clone since SIRF already
        supports arithmetic operations.
    """
    # Check if it's SIRF (has clone method and supports arithmetic)
    if hasattr(proj_data, "clone") and not hasattr(proj_data, "get_proj_data_info"):
        return proj_data.clone()

    # STIR backend: convert to ProjDataInMemory
    try:
        import stir

        # Check if already ProjDataInMemory
        if isinstance(proj_data, stir.ProjDataInMemory):
            return proj_data

        # Convert ProjData to ProjDataInMemory
        proj_data_in_mem = stir.ProjDataInMemory(
            proj_data.get_exam_info(), proj_data.get_proj_data_info()
        )

        # Copy data segment by segment
        for seg_num in range(
            proj_data.get_min_segment_num(), proj_data.get_max_segment_num() + 1
        ):
            seg = proj_data.get_segment_by_sinogram(seg_num)
            proj_data_in_mem.set_segment(seg)

        return proj_data_in_mem
    except ImportError:
        # Neither SIRF nor STIR available - return original
        return proj_data


__all__ = [
    "get_array",
    "interfile_numpy",
    "to_projdata_in_memory",
    "io_utils",
    "simind_utils",
    "stir_utils",
    "sirf_stir_utils",
    "step_size_rules",
    "cil_partitioner",
]

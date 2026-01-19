"""
Grab-bag of general utilities shared across modules.
They are re-exported here for a single, easy import path.
"""

import importlib


# Lazy imports to avoid SIRF dependencies in CI
def __getattr__(name):
    if name in ("io_utils", "simind_utils", "stir_utils"):
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def get_array(obj):
    """Return the fastest available NumPy view of a SIRF-backed object."""

    if hasattr(obj, "asarray"):
        try:
            return obj.asarray()
        except:
            return obj.as_array()
    if hasattr(obj, "as_array"):
        return obj.as_array()
    raise AttributeError(
        f"Object {type(obj)} has neither asarray() nor as_array() method"
    )


__all__ = ["get_array", "io_utils", "simind_utils", "stir_utils"]

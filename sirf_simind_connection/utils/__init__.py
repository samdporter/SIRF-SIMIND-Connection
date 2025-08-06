"""
Grab-bag of general utilities shared across modules.
They are re-exported here for a single, easy import path.
"""


# Lazy imports to avoid SIRF dependencies in CI
def __getattr__(name):
    if name == "io_utils":
        from . import io_utils

        return io_utils
    elif name == "simind_utils":
        from . import simind_utils

        return simind_utils
    elif name == "stir_utils":
        from . import stir_utils

        return stir_utils
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["io_utils", "simind_utils", "stir_utils"]

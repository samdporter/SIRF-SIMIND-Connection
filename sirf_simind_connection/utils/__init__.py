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


__all__ = ["io_utils", "simind_utils", "stir_utils"]

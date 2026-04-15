"""
Converters between SIMIND/DICOM/STIR files and in-memory
representations used elsewhere in the package.
"""

import importlib


# Lazy imports to avoid SIRF dependencies in CI
def __getattr__(name):
    if name in ("attenuation", "dicom_to_stir", "simind_to_stir"):
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["attenuation", "dicom_to_stir", "simind_to_stir"]

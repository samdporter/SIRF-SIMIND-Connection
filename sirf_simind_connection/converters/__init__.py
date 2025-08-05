"""
Converters between SIMIND/DICOM/STIR files and in-memory
representations used elsewhere in the package.
"""


# Lazy imports to avoid SIRF dependencies in CI
def __getattr__(name):
    if name == "attenuation":
        from . import attenuation

        return attenuation
    elif name == "dicom_to_stir":
        from . import dicom_to_stir

        return dicom_to_stir
    elif name == "simind_to_stir":
        from . import simind_to_stir

        return simind_to_stir
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["attenuation", "dicom_to_stir", "simind_to_stir"]

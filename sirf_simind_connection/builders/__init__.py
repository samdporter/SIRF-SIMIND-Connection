"""
Numerical core: configuration acquisition data and image builders for SPECT.
"""

import importlib


# Lazy imports to avoid SIRF dependencies in CI
def __getattr__(name):
    if name == "STIRSPECTAcquisitionDataBuilder":
        mod = importlib.import_module(".acquisition_builder", __name__)
        return getattr(mod, "STIRSPECTAcquisitionDataBuilder")
    elif name == "STIRSPECTImageDataBuilder":
        mod = importlib.import_module(".image_builder", __name__)
        return getattr(mod, "STIRSPECTImageDataBuilder")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["STIRSPECTAcquisitionDataBuilder", "STIRSPECTImageDataBuilder"]

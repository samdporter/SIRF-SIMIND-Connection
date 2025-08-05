"""
Numerical core: configuration acquisition data and image builders for SPECT.
"""


# Lazy imports to avoid SIRF dependencies in CI
def __getattr__(name):
    if name == "STIRSPECTAcquisitionDataBuilder":
        from .acquisition_builder import STIRSPECTAcquisitionDataBuilder

        return STIRSPECTAcquisitionDataBuilder
    elif name == "STIRSPECTImageDataBuilder":
        from .image_builder import STIRSPECTImageDataBuilder

        return STIRSPECTImageDataBuilder
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["STIRSPECTAcquisitionDataBuilder", "STIRSPECTImageDataBuilder"]

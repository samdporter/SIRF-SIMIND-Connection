"""
Converters between SIMIND/DICOM/STIR files and in-memory
representations used elsewhere in the package.
"""

from importlib import import_module as _imp

for _name in ("attenuation", "dicom_to_stir", "simind_to_stir"):
    _imp(f".{_name}", package=__name__)

del _imp, _name

__all__ = ["attenuation", "dicom_to_stir", "simind_to_stir"]

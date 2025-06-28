"""
Access to the built-in attenuation tables bundled with the package.
"""

from importlib import resources as _res
from pathlib import Path

import numpy as _np

__all__ = ["data_path", "load_table"]


def data_path(filename: str) -> Path:
    """Return a path to a packaged data file."""
    return _res.files(__package__).joinpath(filename)


def load_table(filename: str) -> _np.ndarray:
    """
    Load an attenuation table (energy [keV], μ [cm⁻¹]).
    Example filenames: 'bone.atn', 'h2o.atn'.
    """
    with _res.as_file(data_path(filename)) as fp:
        return _np.loadtxt(fp, dtype=float)

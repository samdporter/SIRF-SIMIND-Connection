"""
Packaged scanner presets (YAML or SMC).

Example
-------
>>> from sirf_simind_connection.configs import get
>>> cfg = get("AnyScan.yaml")          # Path object

>>> from sirf_simind_connection.core import SimulationConfig
>>> sim_cfg = SimulationConfig(cfg)    # load file
"""

from importlib import resources as _res
from pathlib import Path
from typing import Union

__all__ = ["get", "list"]


def get(name: str) -> Path:
    """
    Return a **Path** to a bundled config file.

    Parameters
    ----------
    name : str
        Filename, e.g. ``"AnyScan.yaml"`` or ``"input.smc"``.
    """
    return _res.files(__package__).joinpath(name)


def list(ext: Union[str, tuple[str, ...]] = (".yaml", ".smc")) -> list[str]:
    """
    List files in this package (default: *.yaml / *.smc).
    """
    return [
        p.name
        for p in _res.files(__package__).iterdir()
        if p.suffix in (ext if isinstance(ext, tuple) else (ext,))
    ]

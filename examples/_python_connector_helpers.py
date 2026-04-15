from __future__ import annotations

import shutil

import numpy as np

from simind_python_connector import SimindPythonConnector


def require_simind() -> None:
    if shutil.which("simind") is None:
        raise RuntimeError(
            "SIMIND executable not found in PATH. "
            "Install SIMIND before running this example."
        )


def build_small_phantom_zyx() -> tuple[np.ndarray, np.ndarray]:
    dims = (32, 32, 32)  # z, y, x
    z, y, x = np.indices(dims)

    body = ((x - 16) ** 2 + (y - 16) ** 2 <= 10**2) & (np.abs(z - 16) <= 8)
    hot = (x - 22) ** 2 + (y - 16) ** 2 + (z - 16) ** 2 <= 4**2

    source = np.zeros(dims, dtype=np.float32)
    source[body] = 0.8
    source[hot] = 2.0

    mu_map = np.zeros_like(source)
    mu_map[body] = 0.15
    return source, mu_map


def add_standard_runtime(
    connector: SimindPythonConnector,
    photon_multiplier: int = 1,
    seed: int = 12345,
    nuclide: str = "y90_tissue",
    collimator: str | None = None,
) -> None:
    if collimator is None:
        collimator = "ma-lehr" if str(nuclide).lower() == "tc99m" else "ma-megp"
    connector.add_runtime_switch("FI", nuclide)
    connector.add_runtime_switch("CC", collimator)
    connector.add_runtime_switch("NN", int(photon_multiplier))
    connector.add_runtime_switch("RR", int(seed))


def projection_view0(projection: np.ndarray) -> np.ndarray:
    if projection.ndim == 4:
        return projection[0, :, 0, :]
    if projection.ndim == 3:
        return projection[0, :, :]
    raise ValueError(
        f"Expected projection with 3 or 4 dimensions, got shape {projection.shape}"
    )

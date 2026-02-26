from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from sirf_simind_connection import SimindPythonConnector
from sirf_simind_connection.converters.attenuation import attenuation_to_density
from sirf_simind_connection.core.types import MAX_SOURCE, SIMIND_VOXEL_UNIT_CONVERSION
from sirf_simind_connection.utils.simind_utils import create_window_file


def require_simind() -> None:
    if shutil.which("simind") is None:
        raise RuntimeError(
            "SIMIND executable not found in PATH. Install SIMIND before running this example."
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


def configure_voxel_input(
    connector: SimindPythonConnector,
    source: np.ndarray,
    mu_map: np.ndarray,
    voxel_size_mm: float = 4.0,
    scoring_routine: int = 1,
) -> tuple[Path, Path]:
    cfg = connector.get_config()
    dim_z, dim_y, dim_x = (int(v) for v in source.shape)
    vox_cm = voxel_size_mm / SIMIND_VOXEL_UNIT_CONVERSION

    cfg.set_flag(5, True)
    cfg.set_value(15, -1)
    cfg.set_value(14, -1)
    cfg.set_flag(14, True)
    cfg.set_value(84, int(scoring_routine))

    cfg.set_value(2, dim_z * vox_cm / 2.0)
    cfg.set_value(3, dim_x * vox_cm / 2.0)
    cfg.set_value(4, dim_y * vox_cm / 2.0)
    cfg.set_value(28, vox_cm)
    cfg.set_value(76, dim_x)
    cfg.set_value(77, dim_y)

    cfg.set_value(5, dim_z * vox_cm / 2.0)
    cfg.set_value(6, dim_x * vox_cm / 2.0)
    cfg.set_value(7, dim_y * vox_cm / 2.0)
    cfg.set_value(31, vox_cm)
    cfg.set_value(33, 1)
    cfg.set_value(34, dim_z)
    cfg.set_value(78, dim_x)
    cfg.set_value(79, dim_y)

    connector.add_runtime_switch("PX", vox_cm)

    source_max = float(source.max())
    if source_max > 0:
        source_scaled = source / source_max * (MAX_SOURCE * float(connector.quantization_scale))
    else:
        source_scaled = np.zeros_like(source)
    source_u16 = np.clip(np.round(source_scaled), 0, MAX_SOURCE).astype(np.uint16)

    src_prefix = f"{connector.output_prefix}_src"
    source_path = connector.output_dir / f"{src_prefix}.smi"
    source_u16.tofile(source_path)
    cfg.set_data_file(6, src_prefix)

    photon_energy = float(cfg.get_value("photon_energy"))
    if cfg.get_flag(11):
        density = attenuation_to_density(mu_map, photon_energy) * 1000.0
    else:
        density = np.zeros_like(mu_map)
    density_u16 = np.clip(np.round(density), 0, np.iinfo(np.uint16).max).astype(np.uint16)

    dns_prefix = f"{connector.output_prefix}_dns"
    density_path = connector.output_dir / f"{dns_prefix}.dmi"
    density_u16.tofile(density_path)
    cfg.set_data_file(5, dns_prefix)

    connector.add_config_value(1, 140.0)
    connector.add_config_value(19, 2)
    connector.add_config_value(53, 0)
    return source_path, density_path


def write_windows(
    connector: SimindPythonConnector,
    lowers: list[float],
    uppers: list[float],
    orders: list[int],
) -> None:
    create_window_file(
        lowers,
        uppers,
        orders,
        output_filename=str(connector.output_dir / connector.output_prefix),
    )


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

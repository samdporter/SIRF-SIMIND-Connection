from __future__ import annotations

import ast
import math
import struct
from array import array
from pathlib import Path
from typing import Callable

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.requires_simind


def _parse_interfile_header(path: Path) -> dict[str, str]:
    header: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        if ":=" not in raw_line:
            continue
        key, value = raw_line.split(":=", 1)
        header[key.strip().lower()] = value.strip()
    return header


def _header_value(header: dict[str, str], *keys: str) -> str:
    for key in keys:
        if key in header:
            return header[key]
    raise KeyError(f"Missing Interfile key. Tried: {keys}")


def _image_geometry_from_hv(hv_path: Path) -> dict[str, float]:
    header = _parse_interfile_header(hv_path)
    return {
        "nx": int(float(_header_value(header, "!matrix size [1]", "matrix size [1]"))),
        "ny": int(float(_header_value(header, "!matrix size [2]", "matrix size [2]"))),
        "nz": int(float(_header_value(header, "!matrix size [3]", "matrix size [3]"))),
        "vx": float(
            _header_value(
                header,
                "scaling factor (mm/pixel) [1]",
                "!scaling factor (mm/pixel) [1]",
            )
        ),
        "vy": float(
            _header_value(
                header,
                "scaling factor (mm/pixel) [2]",
                "!scaling factor (mm/pixel) [2]",
            )
        ),
        "vz": float(
            _header_value(
                header,
                "scaling factor (mm/pixel) [3]",
                "!scaling factor (mm/pixel) [3]",
            )
        ),
    }


def _load_hv_volume(hv_path: Path) -> list[list[list[float]]]:
    header = _parse_interfile_header(hv_path)
    nx = int(float(_header_value(header, "!matrix size [1]", "matrix size [1]")))
    ny = int(float(_header_value(header, "!matrix size [2]", "matrix size [2]")))
    nz = int(float(_header_value(header, "!matrix size [3]", "matrix size [3]")))
    data_filename = _header_value(header, "!name of data file", "name of data file")
    data_path = (hv_path.parent / data_filename).resolve()

    raw = array("f")
    with open(data_path, "rb") as handle:
        raw.frombytes(handle.read())

    expected = nx * ny * nz
    if len(raw) != expected:
        raise AssertionError(
            f"Unexpected voxel count for {hv_path}: expected {expected}, got {len(raw)}"
        )

    values = list(raw)
    volume: list[list[list[float]]] = []
    cursor = 0
    for _z in range(nz):
        plane: list[list[float]] = []
        for _y in range(ny):
            row = values[cursor : cursor + nx]
            cursor += nx
            plane.append(row)
        volume.append(plane)
    return volume


def _mid_slice(volume: list[list[list[float]]]) -> list[list[float]]:
    return volume[len(volume) // 2]


def _flatten_2d(image: list[list[float]]) -> list[float]:
    return [value for row in image for value in row]


def _normalize_2d(image: list[list[float]]) -> list[list[float]]:
    flat = _flatten_2d(image)
    mean = sum(flat) / len(flat)
    var = sum((value - mean) * (value - mean) for value in flat) / len(flat)
    std = math.sqrt(var + 1e-12)
    return [[(value - mean) / std for value in row] for row in image]


def _corr_2d(a: list[list[float]], b: list[list[float]]) -> float:
    flat_a = _flatten_2d(a)
    flat_b = _flatten_2d(b)
    if len(flat_a) != len(flat_b):
        raise AssertionError(
            f"Correlation shape mismatch: {len(flat_a)} vs {len(flat_b)} elements"
        )
    return sum(x * y for x, y in zip(flat_a, flat_b)) / len(flat_a)


def _flip_x(image: list[list[float]]) -> list[list[float]]:
    return [list(reversed(row)) for row in image]


def _flip_y(image: list[list[float]]) -> list[list[float]]:
    return list(reversed(image))


def _rot90(image: list[list[float]]) -> list[list[float]]:
    height = len(image)
    width = len(image[0])
    return [[image[height - 1 - r][c] for r in range(height)] for c in range(width)]


def _rot180(image: list[list[float]]) -> list[list[float]]:
    return _flip_y(_flip_x(image))


def _rot270(image: list[list[float]]) -> list[list[float]]:
    return _rot90(_rot180(image))


def _weighted_com_top_percentile(
    image: list[list[float]], percentile: float = 90.0
) -> tuple[float, float]:
    flat = sorted(_flatten_2d(image))
    cutoff_index = int((percentile / 100.0) * len(flat))
    cutoff_index = min(max(cutoff_index, 0), len(flat) - 1)
    threshold = flat[cutoff_index]

    mass = 0.0
    sum_x = 0.0
    sum_y = 0.0
    for y, row in enumerate(image):
        for x, value in enumerate(row):
            if value < threshold:
                continue
            mass += value
            sum_x += x * value
            sum_y += y * value

    if mass <= 0:
        raise AssertionError("Failed to compute hotspot center of mass (zero mass)")
    return sum_x / mass, sum_y / mass


def _variance(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return sum((value - mean) * (value - mean) for value in values) / len(values)


def _load_npy_float(path: Path) -> tuple[list[float], tuple[int, ...]]:
    with open(path, "rb") as handle:
        magic = handle.read(6)
        if magic != b"\x93NUMPY":
            raise AssertionError(f"{path} is not a .npy file")

        major, _minor = struct.unpack("BB", handle.read(2))
        if major == 1:
            header_len = struct.unpack("<H", handle.read(2))[0]
        elif major in (2, 3):
            header_len = struct.unpack("<I", handle.read(4))[0]
        else:
            raise AssertionError(f"Unsupported .npy version in {path}: {major}")

        header_raw = handle.read(header_len).decode("latin1")
        header = ast.literal_eval(header_raw.strip())

        dtype = header["descr"]
        fortran_order = bool(header["fortran_order"])
        shape = tuple(int(dim) for dim in header["shape"])

        if fortran_order:
            raise AssertionError(f"Fortran-order .npy is not supported in {path}")
        if dtype not in ("<f4", "|f4", "<f8"):
            raise AssertionError(f"Unsupported dtype {dtype} in {path}")

        payload = handle.read()

    count = math.prod(shape)
    item_size = 4 if dtype in ("<f4", "|f4") else 8
    expected_bytes = count * item_size
    if len(payload) != expected_bytes:
        raise AssertionError(
            f"Unexpected payload size in {path}: expected {expected_bytes}, got {len(payload)}"
        )

    fmt = "<f" if item_size == 4 else "<d"
    values = [item[0] for item in struct.iter_unpack(fmt, payload)]
    return values, shape


def _require_output_files(paths: list[Path], hint: str) -> None:
    missing = [path for path in paths if not path.exists()]
    if not missing:
        return
    missing_list = ", ".join(str(path) for path in missing)
    pytest.skip(f"{hint}. Missing output file(s): {missing_list}")


def _osem_case_paths(backend: str) -> tuple[Path, Path, Path]:
    case_dir = ROOT / "output" / f"{backend}_adaptor_osem"
    source_hv = case_dir / f"{backend}_source.hv"
    recon_hv = case_dir / f"{backend}_osem_recon.hv"
    total_hs = case_dir / f"{backend}_total.hs"
    return source_hv, recon_hv, total_hs


@pytest.mark.integration
@pytest.mark.ci_skip
@pytest.mark.parametrize("backend", ["stir", "sirf"])
def test_osem_reconstruction_orientation_matches_input(backend: str) -> None:
    source_hv, recon_hv, _total_hs = _osem_case_paths(backend)
    _require_output_files(
        [source_hv, recon_hv],
        hint=(
            f"Run examples/07{'A' if backend == 'stir' else 'B'}_"
            f"{backend}_adaptor_osem.py to generate geometry diagnostics data"
        ),
    )

    source_slice = _mid_slice(_load_hv_volume(source_hv))
    recon_slice = _mid_slice(_load_hv_volume(recon_hv))

    source_norm = _normalize_2d(source_slice)
    recon_norm = _normalize_2d(recon_slice)

    transforms: dict[str, Callable[[list[list[float]]], list[list[float]]]] = {
        "identity": lambda image: image,
        "flip_x": _flip_x,
        "flip_y": _flip_y,
        "flip_xy": lambda image: _flip_x(_flip_y(image)),
        "rot90": _rot90,
        "rot180": _rot180,
        "rot270": _rot270,
    }
    correlations = {
        name: _corr_2d(source_norm, transform(recon_norm))
        for name, transform in transforms.items()
    }

    identity = correlations["identity"]
    best_name, best_corr = max(correlations.items(), key=lambda item: item[1])
    assert identity + 0.05 >= best_corr, (
        f"{backend.upper()} reconstruction appears orientation-misaligned. "
        f"Identity correlation={identity:.4f}, best transform={best_name} "
        f"({best_corr:.4f}). Correlations={correlations}"
    )

    src_hotspot = _weighted_com_top_percentile(source_slice)
    rec_hotspot = _weighted_com_top_percentile(recon_slice)
    dx = rec_hotspot[0] - src_hotspot[0]
    dy = rec_hotspot[1] - src_hotspot[1]
    assert abs(dx) <= 1.75 and abs(dy) <= 1.75, (
        f"{backend.upper()} hotspot displacement is too large: "
        f"dx={dx:.3f}, dy={dy:.3f}. src={src_hotspot}, rec={rec_hotspot}"
    )


@pytest.mark.integration
@pytest.mark.ci_skip
@pytest.mark.parametrize("backend", ["stir", "sirf"])
def test_osem_projection_header_matches_source_sampling(backend: str) -> None:
    source_hv, _recon_hv, total_hs = _osem_case_paths(backend)
    _require_output_files(
        [source_hv, total_hs],
        hint=(
            f"Run examples/07{'A' if backend == 'stir' else 'B'}_"
            f"{backend}_adaptor_osem.py to generate geometry diagnostics data"
        ),
    )

    source_geom = _image_geometry_from_hv(source_hv)
    projection_header = _parse_interfile_header(total_hs)

    proj_nx = int(float(_header_value(projection_header, "!matrix size [1]")))
    proj_ny = int(float(_header_value(projection_header, "!matrix size [2]")))
    proj_vx = float(
        _header_value(
            projection_header,
            "!scaling factor (mm/pixel) [1]",
            "scaling factor (mm/pixel) [1]",
        )
    )
    proj_vy = float(
        _header_value(
            projection_header,
            "!scaling factor (mm/pixel) [2]",
            "scaling factor (mm/pixel) [2]",
        )
    )

    assert proj_nx == int(source_geom["nx"])
    assert proj_ny == int(source_geom["ny"])
    assert math.isclose(proj_vx, source_geom["vx"], rel_tol=0.0, abs_tol=1e-3)
    assert math.isclose(proj_vy, source_geom["vy"], rel_tol=0.0, abs_tol=1e-3)

    # Rotation metadata should always be present in projection headers used for OSEM.
    _header_value(projection_header, "!direction of rotation", "direction of rotation")
    _header_value(projection_header, "start angle", "!start angle")
    projections = int(float(_header_value(projection_header, "!number of projections")))
    assert projections > 0


@pytest.mark.integration
@pytest.mark.ci_skip
@pytest.mark.requires_pytomography
def test_pytomography_osem_reconstruction_not_axis_collapsed() -> None:
    recon_path = ROOT / "output" / "pytomography_adaptor_osem" / "pytomo_osem_recon.npy"
    _require_output_files(
        [recon_path],
        hint=(
            "Run examples/07C_pytomography_adaptor_osem.py to generate "
            "PyTomography geometry diagnostics data"
        ),
    )

    values, shape = _load_npy_float(recon_path)
    if len(shape) != 3:
        pytest.skip(
            f"Expected 3D reconstruction tensor for diagnostics, got shape={shape}"
        )

    nz, ny, nx = shape
    z_mid = nz // 2
    offset = z_mid * ny * nx
    slice_flat = values[offset : offset + ny * nx]
    rows = [slice_flat[row * nx : (row + 1) * nx] for row in range(ny)]

    row_means = [sum(row) / nx for row in rows]
    col_means = [sum(rows[row][col] for row in range(ny)) / ny for col in range(nx)]

    row_var = _variance(row_means)
    col_var = _variance(col_means)
    min_var = min(row_var, col_var)
    max_var = max(row_var, col_var)
    anisotropy_ratio = min_var / (max_var + 1e-12)

    assert anisotropy_ratio > 0.25, (
        "PyTomography OSEM reconstruction appears axis-collapsed "
        f"(row_var={row_var:.6e}, col_var={col_var:.6e}, ratio={anisotropy_ratio:.6f})"
    )

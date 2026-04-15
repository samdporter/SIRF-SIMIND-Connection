"""
Utilities for loading Interfile projection data directly into NumPy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from .interfile_parser import parse_interfile_header


MatrixShape = Sequence[int]
HeaderInput = Union[str, Path]

_MATRIX_SIZE_PATTERN = re.compile(r"!?matrix size\s*\[(\d+)\]", re.IGNORECASE)


@dataclass(frozen=True)
class InterfileArray:
    """NumPy payload and file references parsed from an Interfile header."""

    array: np.ndarray
    header_path: Path
    data_path: Path
    metadata: dict[str, str]


def _lookup_header_value(header: Mapping[str, str], *keys: str) -> Optional[str]:
    """Return the first matching value for a header key (case-insensitive)."""
    lower_map = {key.lower(): value for key, value in header.items()}
    for key in keys:
        value = lower_map.get(key.lower())
        if value is not None:
            return value
    return None


def _extract_matrix_shape(header: Mapping[str, str]) -> tuple[int, ...]:
    """Extract matrix sizes and return NumPy shape in memory order."""
    sizes_by_index: dict[int, int] = {}

    for key, raw_value in header.items():
        match = _MATRIX_SIZE_PATTERN.fullmatch(key.strip())
        if not match:
            continue
        axis = int(match.group(1))
        sizes_by_index[axis] = int(raw_value)

    if not sizes_by_index:
        raise ValueError("No 'matrix size [i]' entries found in Interfile header")

    ordered_axes = [sizes_by_index[idx] for idx in sorted(sizes_by_index)]
    if any(size <= 0 for size in ordered_axes):
        raise ValueError(f"Invalid matrix sizes in header: {ordered_axes}")

    # Interfile stores axis [1] as the fastest changing index.
    return tuple(reversed(ordered_axes))


def _extract_numpy_dtype(header: Mapping[str, str]) -> np.dtype:
    """Infer the binary payload dtype from Interfile metadata."""
    number_format = (
        _lookup_header_value(header, "!number format", "number format") or "float"
    ).strip()
    bytes_per_pixel_raw = _lookup_header_value(
        header, "!number of bytes per pixel", "number of bytes per pixel"
    )
    bytes_per_pixel = int(float(bytes_per_pixel_raw)) if bytes_per_pixel_raw else 4

    number_format_lc = number_format.lower()
    if "float" in number_format_lc:
        kind = "f"
    elif "unsigned" in number_format_lc:
        kind = "u"
    elif "signed" in number_format_lc or "integer" in number_format_lc:
        kind = "i"
    else:
        raise ValueError(f"Unsupported Interfile number format: {number_format!r}")

    dtype = np.dtype(f"{kind}{bytes_per_pixel}")
    byte_order = (
        _lookup_header_value(header, "imagedata byte order", "!imagedata byte order")
        or ""
    ).lower()

    if "big" in byte_order:
        return dtype.newbyteorder(">")
    if "little" in byte_order:
        return dtype.newbyteorder("<")
    return dtype.newbyteorder("=")


def _extract_projection_count(header: Mapping[str, str]) -> Optional[int]:
    """Extract projection/image count used for 3D SPECT sinograms."""
    raw = _lookup_header_value(
        header,
        "!number of projections",
        "number of projections",
        "!total number of images",
        "total number of images",
        "!number of images/energy window",
        "number of images/energy window",
    )
    if raw is None:
        return None
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _infer_leading_axis_count(
    metadata: Mapping[str, str], plane_elements: int, payload_elements: int
) -> Optional[int]:
    """Infer leading stack axis for headers that only declare 2D planes."""
    if plane_elements <= 0:
        return None

    # Prefer explicit projection/image-count fields from the header.
    projection_count = _extract_projection_count(metadata)
    if projection_count is not None:
        return projection_count

    # Otherwise infer directly from payload length when possible.
    if payload_elements % plane_elements == 0:
        inferred = payload_elements // plane_elements
        if inferred > 0:
            return inferred

    # Fall back to a single-frame stack rather than treating projections as 2D.
    return 1


def load_interfile_array(header_path: HeaderInput) -> InterfileArray:
    """Load projection data referenced by an Interfile header into NumPy."""
    header_path = Path(header_path).expanduser().resolve()
    metadata = parse_interfile_header(str(header_path))

    data_filename = _lookup_header_value(
        metadata, "!name of data file", "name of data file"
    )
    if not data_filename:
        raise ValueError(
            f"Interfile header {header_path} does not define 'name of data file'"
        )

    data_path = (header_path.parent / data_filename).resolve()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Interfile data file referenced by header does not exist: {data_path}"
        )

    dtype = _extract_numpy_dtype(metadata)
    shape = _extract_matrix_shape(metadata)
    expected_elements = int(np.prod(shape))

    flat = np.fromfile(data_path, dtype=dtype)

    # Projection and tomographic payloads are treated as at least 3D:
    # [leading_axis, axis2, axis1]. If headers only declare matrix sizes [1],[2],
    # recover the leading axis from projection count fields or payload length.
    if len(shape) == 2:
        leading_count = _infer_leading_axis_count(
            metadata, expected_elements, flat.size
        )
        if leading_count is not None:
            shape = (leading_count, *shape)
            expected_elements = int(np.prod(shape))

    if flat.size < expected_elements:
        raise ValueError(
            f"Data size mismatch for {data_path}: expected {expected_elements} "
            f"elements for shape {shape}, found {flat.size}"
        )

    # SIMIND files can contain trailing payload not represented by Interfile keys.
    # Use the declared geometry and discard trailing elements.
    if flat.size > expected_elements:
        flat = flat[:expected_elements]

    return InterfileArray(
        array=flat.reshape(shape),
        header_path=header_path,
        data_path=data_path,
        metadata=dict(metadata),
    )


__all__ = ["InterfileArray", "load_interfile_array"]

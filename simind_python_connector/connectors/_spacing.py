"""Helpers for extracting voxel spacing from SIRF/STIR image objects."""

from __future__ import annotations

from typing import Any


def _call_or_value(value: Any) -> Any:
    return value() if callable(value) else value


def _voxel_size_from_spacing(spacing: Any) -> float | None:
    try:
        values = tuple(float(v) for v in spacing)
    except Exception:
        values = ()

    if values:
        if len(values) >= 4:
            return float(values[3])
        if len(values) >= 3:
            return float(values[2])

    if hasattr(spacing, "z"):
        try:
            return float(_call_or_value(getattr(spacing, "z")))
        except Exception:
            pass

    if hasattr(spacing, "at"):
        for index in (3, 2):
            try:
                return float(spacing.at(index))
            except Exception:
                continue

    for index in (3, 2):
        try:
            return float(spacing[index])
        except Exception:
            continue

    return None


def extract_voxel_size_mm(image: Any, backend_name: str) -> float:
    """Extract z voxel spacing in mm from backend image metadata."""
    if hasattr(image, "voxel_sizes"):
        voxel_sizes = image.voxel_sizes()
        try:
            if len(voxel_sizes) >= 3:
                return float(voxel_sizes[2])
        except Exception:
            pass

    if hasattr(image, "get_grid_spacing"):
        voxel_size_mm = _voxel_size_from_spacing(image.get_grid_spacing())
        if voxel_size_mm is not None:
            return voxel_size_mm
        raise ValueError(
            f"Unable to read voxel spacing from {backend_name} get_grid_spacing()."
        )

    raise ValueError(
        f"{backend_name} source object must expose voxel_sizes() or get_grid_spacing()."
    )


__all__ = ["extract_voxel_size_mm"]

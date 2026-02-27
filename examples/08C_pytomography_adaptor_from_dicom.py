#!/usr/bin/env python
"""
Create a PyTomography projection tensor from DICOM data.

This example follows PyTomography's DICOM IO API and requires a DICOM NM
projection file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pytomography.io.SPECT import dicom as pytomo_dicom


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _reduce_to_theta_r_z(array: np.ndarray) -> np.ndarray:
    result = array
    while result.ndim > 3:
        result = result[0]
    if result.ndim != 3:
        raise ValueError(
            f"Expected a 3D projection tensor after squeezing, got {result.shape}"
        )
    return result


def main(dicom_file: str) -> None:
    output_dir = Path("output/dicom_projection_objects/pytomography")
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = pytomo_dicom.get_projections(dicom_file, index_peak=0)
    if isinstance(loaded, tuple) and len(loaded) == 3:
        object_meta, proj_meta, projections = loaded
    else:
        object_meta = None
        proj_meta = None
        projections = loaded

    projection = _reduce_to_theta_r_z(
        _as_numpy(projections).astype(np.float32, copy=False)
    )
    np.save(output_dir / "pytomography_projection.npy", projection)

    view0 = projection[0, :, :]

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.imshow(view0, cmap="magma")
    ax.set_title("PyTomography projection (view 0)")
    ax.axis("off")
    plt.tight_layout()
    plot_path = output_dir / "pytomography_from_dicom_summary.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    print(f"DICOM input: {dicom_file}")
    print(f"Projection shape (theta, r, z): {projection.shape}")
    if object_meta is not None:
        print(f"Object metadata type: {type(object_meta)}")
    if proj_meta is not None:
        print(f"Projection metadata type: {type(proj_meta)}")
    print(f"Summary plot: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a PyTomography projection tensor from DICOM"
    )
    parser.add_argument(
        "dicom_file", type=str, help="Path to SPECT projection DICOM (.dcm)"
    )
    args = parser.parse_args()
    main(args.dicom_file)

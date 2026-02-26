#!/usr/bin/env python
"""
Create a SIRF projection object from DICOM data.

This example requires a DICOM NM projection file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from sirf_simind_connection.builders import STIRSPECTAcquisitionDataBuilder
from sirf_simind_connection.utils import get_array


def _view0(projection):
    if projection.ndim == 4:
        return projection[0, :, 0, :]
    if projection.ndim == 3:
        return projection[0, :, :]
    raise ValueError(f"Unexpected projection dimensions: {projection.shape}")


def main(dicom_file: str) -> None:
    output_dir = Path("output/dicom_projection_objects/sirf")
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = STIRSPECTAcquisitionDataBuilder(backend="sirf")
    builder.update_header_from_dicom(dicom_file)

    output_base = output_dir / "sirf_from_dicom"
    acq = builder.build(output_path=output_base)

    projection = get_array(acq)
    view0 = _view0(projection)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.imshow(view0, cmap="magma")
    ax.set_title("SIRF projection (view 0)")
    ax.axis("off")
    plt.tight_layout()
    plot_path = output_dir / "sirf_from_dicom_summary.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    print(f"DICOM input: {dicom_file}")
    print(f"Projection object type: {type(acq)}")
    print(f"Projection shape: {projection.shape}")
    print(f"Interfile output base: {output_base}")
    print(f"Summary plot: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a SIRF projection object from DICOM"
    )
    parser.add_argument("dicom_file", type=str, help="Path to SPECT projection DICOM (.dcm)")
    args = parser.parse_args()
    main(args.dicom_file)

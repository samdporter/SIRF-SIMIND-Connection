#!/usr/bin/env python
"""
Basic pure Python connector simulation.

This example runs SIMIND with voxelized NumPy inputs and reads projections
back as NumPy arrays via ``SimindPythonConnector``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _python_connector_helpers import (
    add_standard_runtime,
    build_small_phantom_zyx,
    configure_voxel_input,
    projection_view0,
    require_simind,
    write_windows,
)

from sirf_simind_connection import SimindPythonConnector, configs


def main() -> None:
    require_simind()

    output_dir = Path("output/basic_simulation")
    output_dir.mkdir(parents=True, exist_ok=True)

    connector = SimindPythonConnector(
        config_source=configs.get("Example.yaml"),
        output_dir=output_dir,
        output_prefix="basic_case01",
        quantization_scale=0.05,
    )

    source, mu_map = build_small_phantom_zyx()
    source_path, density_path = configure_voxel_input(
        connector,
        source,
        mu_map,
        voxel_size_mm=4.0,
        scoring_routine=1,
    )
    write_windows(connector, [126.0], [154.0], [0])
    add_standard_runtime(connector, photon_multiplier=1, seed=12345, nuclide="tc99m")

    outputs = connector.run()
    total = outputs["tot_w1"].projection
    scatter = outputs.get("sca_w1", outputs["tot_w1"]).projection
    primary = np.clip(total - scatter, a_min=0.0, a_max=None)

    np.save(output_dir / "total.npy", total)
    np.save(output_dir / "scatter.npy", scatter)
    np.save(output_dir / "primary.npy", primary)

    source_slice = source[source.shape[0] // 2, :, :]
    total_view0 = projection_view0(total)
    scatter_view0 = projection_view0(scatter)
    primary_view0 = projection_view0(primary)
    proj_vmin = float(min(total_view0.min(), scatter_view0.min(), primary_view0.min()))
    proj_vmax = float(max(total_view0.max(), scatter_view0.max(), primary_view0.max()))

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(source_slice, cmap="viridis")
    axes[0].set_title("Input source")
    axes[1].imshow(total_view0, cmap="magma", vmin=proj_vmin, vmax=proj_vmax)
    axes[1].set_title("Total (view 0)")
    axes[2].imshow(scatter_view0, cmap="magma", vmin=proj_vmin, vmax=proj_vmax)
    axes[2].set_title("Scatter (view 0)")
    axes[3].imshow(primary_view0, cmap="magma", vmin=proj_vmin, vmax=proj_vmax)
    axes[3].set_title("Primary (view 0)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plot_path = output_dir / "basic_simulation_summary.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    print(f"Output directory: {output_dir}")
    print(f"Source file: {source_path}")
    print(f"Density file: {density_path}")
    print(f"Total counts: {float(total.sum()):.3f}")
    print(f"Scatter counts: {float(scatter.sum()):.3f}")
    print(f"Primary counts: {float(primary.sum()):.3f}")
    print(f"Summary plot: {plot_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Multi-window simulation (TEW-style) using the pure Python connector.
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

    output_dir = Path("output/multi_window")
    output_dir.mkdir(parents=True, exist_ok=True)

    connector = SimindPythonConnector(
        config_source=configs.get("Example.yaml"),
        output_dir=output_dir,
        output_prefix="tew_case01",
        quantization_scale=0.05,
    )

    source, mu_map = build_small_phantom_zyx()
    configure_voxel_input(
        connector,
        source,
        mu_map,
        voxel_size_mm=4.0,
        scoring_routine=1,
    )

    # Lu-177 TEW around the 208 keV peak:
    # lower scatter 166-187, photopeak 187-229, upper scatter 229-250 keV.
    lowers = [166.0, 187.0, 229.0]
    uppers = [187.0, 229.0, 250.0]
    orders = [0, 0, 0]
    write_windows(connector, lowers, uppers, orders)
    connector.add_config_value(1, 208.0)
    add_standard_runtime(
        connector, photon_multiplier=1, seed=12345, nuclide="lu177"
    )

    outputs = connector.run()

    lower_scatter = outputs["tot_w1"].projection
    photopeak = outputs["tot_w2"].projection
    upper_scatter = outputs["tot_w3"].projection

    width1 = uppers[0] - lowers[0]
    width2 = uppers[1] - lowers[1]
    width3 = uppers[2] - lowers[2]
    scatter_estimate = lower_scatter * (width2 / (2.0 * width1)) + upper_scatter * (
        width2 / (2.0 * width3)
    )
    corrected = np.clip(photopeak - scatter_estimate, a_min=0.0, a_max=None)

    np.save(output_dir / "tew_lower_window.npy", lower_scatter)
    np.save(output_dir / "tew_photopeak.npy", photopeak)
    np.save(output_dir / "tew_upper_window.npy", upper_scatter)
    np.save(output_dir / "tew_scatter_estimate.npy", scatter_estimate)
    np.save(output_dir / "tew_corrected.npy", corrected)

    source_slice = source[source.shape[0] // 2, :, :]
    photopeak_view0 = projection_view0(photopeak)
    scatter_view0 = projection_view0(scatter_estimate)
    corrected_view0 = projection_view0(corrected)
    lower_view0 = projection_view0(lower_scatter)
    proj_vmin = float(
        min(
            photopeak_view0.min(),
            scatter_view0.min(),
            corrected_view0.min(),
            lower_view0.min(),
        )
    )
    proj_vmax = float(
        max(
            photopeak_view0.max(),
            scatter_view0.max(),
            corrected_view0.max(),
            lower_view0.max(),
        )
    )

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(source_slice, cmap="viridis")
    axes[0].set_title("Input source")
    axes[1].imshow(photopeak_view0, cmap="magma", vmin=proj_vmin, vmax=proj_vmax)
    axes[1].set_title("Photopeak (view 0)")
    axes[2].imshow(scatter_view0, cmap="magma", vmin=proj_vmin, vmax=proj_vmax)
    axes[2].set_title("TEW scatter est.")
    axes[3].imshow(corrected_view0, cmap="magma", vmin=proj_vmin, vmax=proj_vmax)
    axes[3].set_title("TEW corrected")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plot_path = output_dir / "multi_window_summary.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    print(f"Output directory: {output_dir}")
    print(f"Photopeak counts: {float(photopeak.sum()):.3f}")
    print(f"TEW scatter estimate counts: {float(scatter_estimate.sum()):.3f}")
    print(f"TEW corrected counts: {float(corrected.sum()):.3f}")
    print(f"Summary plot: {plot_path}")


if __name__ == "__main__":
    main()

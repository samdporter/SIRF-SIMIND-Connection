#!/usr/bin/env python
"""
Runtime-switch comparison using the pure Python connector.

This example runs two SIMIND cases with different runtime switch settings and
compares the resulting projection view.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sirf_simind_connection import SimindPythonConnector, configs

from _python_connector_helpers import (
    add_standard_runtime,
    build_small_phantom_zyx,
    configure_voxel_input,
    projection_view0,
    require_simind,
    write_windows,
)


def _run_case(
    output_dir: Path,
    prefix: str,
    source: np.ndarray,
    mu_map: np.ndarray,
    seed: int,
) -> np.ndarray:
    connector = SimindPythonConnector(
        config_source=configs.get("Example.yaml"),
        output_dir=output_dir,
        output_prefix=prefix,
        quantization_scale=0.05,
    )
    configure_voxel_input(
        connector,
        source,
        mu_map,
        voxel_size_mm=4.0,
        scoring_routine=1,
    )
    write_windows(connector, [75.0], [225.0], [0])
    add_standard_runtime(connector, photon_multiplier=1, seed=seed)
    outputs = connector.run()
    return outputs["tot_w1"].projection


def main() -> None:
    require_simind()

    output_dir = Path("output/runtime_switch_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    source, mu_map = build_small_phantom_zyx()

    base_projection = _run_case(
        output_dir=output_dir / "seed_12345",
        prefix="runtime_seed_12345",
        source=source,
        mu_map=mu_map,
        seed=12345,
    )
    alt_projection = _run_case(
        output_dir=output_dir / "seed_54321",
        prefix="runtime_seed_54321",
        source=source,
        mu_map=mu_map,
        seed=54321,
    )

    np.save(output_dir / "projection_seed_12345.npy", base_projection)
    np.save(output_dir / "projection_seed_54321.npy", alt_projection)

    source_slice = source[source.shape[0] // 2, :, :]
    base_view = projection_view0(base_projection)
    alt_view = projection_view0(alt_projection)
    diff_view = alt_view - base_view

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(source_slice, cmap="viridis")
    axes[0].set_title("Input source")
    axes[1].imshow(base_view, cmap="magma")
    axes[1].set_title("Seed 12345")
    axes[2].imshow(alt_view, cmap="magma")
    axes[2].set_title("Seed 54321")
    axes[3].imshow(diff_view, cmap="coolwarm")
    axes[3].set_title("Difference")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plot_path = output_dir / "runtime_switch_comparison.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    print(f"Output directory: {output_dir}")
    print(f"Seed 12345 counts: {float(base_projection.sum()):.3f}")
    print(f"Seed 54321 counts: {float(alt_projection.sum()):.3f}")
    print(f"Difference sum: {float(diff_view.sum()):.3f}")
    print(f"Summary plot: {plot_path}")


if __name__ == "__main__":
    main()

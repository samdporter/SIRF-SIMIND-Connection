#!/usr/bin/env python
"""
SCATTWIN vs PENETRATE comparison using the pure Python connector.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
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


def _run_case(
    output_dir: Path,
    prefix: str,
    source: np.ndarray,
    mu_map: np.ndarray,
    scoring_routine: int,
) -> dict[str, np.ndarray]:
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
        scoring_routine=scoring_routine,
    )
    # Use full detector-hit acceptance for both SCATTWIN and PENETRATE in this
    # comparison example.
    connector.add_config_value(19, -90)
    connector.add_config_value(53, 1)
    write_windows(connector, [75.0], [225.0], [0])
    add_standard_runtime(connector, photon_multiplier=1, seed=12345)

    results = connector.run()
    return {key: value.projection for key, value in results.items()}


def _pick_penetrate_preview(outputs: dict[str, np.ndarray]) -> tuple[str, np.ndarray]:
    preferred = (
        "all_interactions_w1",
        "all_interactions",
        "tot_w1",
        "b01",
    )
    for key in preferred:
        if key in outputs:
            return key, outputs[key]
    first_key = sorted(outputs.keys())[0]
    return first_key, outputs[first_key]


def _save_total_bar_comparison(
    scatt_primary_sum: float,
    scatt_scatter_sum: float,
    pen_outputs: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    scatt_primary_color = cm.Greens(0.6)
    scatt_scatter_color = cm.Reds(0.6)

    # SCATTWIN total bar: primary + scatter.
    ax.bar(
        0,
        scatt_primary_sum,
        color=scatt_primary_color,
        alpha=0.9,
        width=0.7,
        label="SCATTWIN Primary",
    )
    ax.bar(
        0,
        scatt_scatter_sum,
        bottom=scatt_primary_sum,
        color=scatt_scatter_color,
        alpha=0.9,
        width=0.7,
        label="SCATTWIN Scatter",
    )

    # PENETRATE total bar: stacked detailed components.
    primary_components = [
        ("geom_coll_primary", scatt_primary_color, "Primary: Geom Coll"),
        ("septal_pen_primary", cm.Greens(0.85), "Primary: Septal Pen"),
        ("coll_scatter_primary", cm.Greens(0.45), "Primary: Coll Scatter"),
        ("coll_xray_primary", cm.Greens(0.30), "Primary: Coll X-ray"),
    ]
    scatter_components = [
        ("geom_coll_scattered", scatt_scatter_color, "Scatter: Geom Coll"),
        ("septal_pen_scattered", cm.Reds(0.85), "Scatter: Septal Pen"),
        ("coll_scatter_scattered", cm.Reds(0.45), "Scatter: Coll Scatter"),
        ("coll_xray_scattered", cm.Reds(0.30), "Scatter: Coll X-ray"),
    ]
    primary_backscatter_components = [
        "geom_coll_primary_back",
        "septal_pen_primary_back",
        "coll_scatter_primary_back",
        "coll_xray_primary_back",
    ]
    scatter_backscatter_components = [
        "geom_coll_scattered_back",
        "septal_pen_scattered_back",
        "coll_scatter_scattered_back",
        "coll_xray_scattered_back",
    ]
    primary_backscatter_total = sum(
        float(pen_outputs.get(comp, np.array(0.0)).sum())
        if comp in pen_outputs
        else 0.0
        for comp in primary_backscatter_components
    )
    scatter_backscatter_total = sum(
        float(pen_outputs.get(comp, np.array(0.0)).sum())
        if comp in pen_outputs
        else 0.0
        for comp in scatter_backscatter_components
    )

    pen_x = 1
    pen_bottom = 0.0
    for comp_slug, color, label in primary_components:
        if comp_slug in pen_outputs:
            val = float(pen_outputs[comp_slug].sum())
            if val > 0:
                ax.bar(
                    pen_x,
                    val,
                    bottom=pen_bottom,
                    color=color,
                    alpha=0.9,
                    width=0.7,
                    label=label,
                )
                pen_bottom += val
    if primary_backscatter_total > 0:
        ax.bar(
            pen_x,
            primary_backscatter_total,
            bottom=pen_bottom,
            color=cm.Greens(0.2),
            alpha=0.9,
            width=0.7,
            label="Primary Backscatter",
        )
        pen_bottom += primary_backscatter_total

    for comp_slug, color, label in scatter_components:
        if comp_slug in pen_outputs:
            val = float(pen_outputs[comp_slug].sum())
            if val > 0:
                ax.bar(
                    pen_x,
                    val,
                    bottom=pen_bottom,
                    color=color,
                    alpha=0.9,
                    width=0.7,
                    label=label,
                )
                pen_bottom += val
    if scatter_backscatter_total > 0:
        ax.bar(
            pen_x,
            scatter_backscatter_total,
            bottom=pen_bottom,
            color=cm.Reds(0.2),
            alpha=0.9,
            width=0.7,
            label="Scatter Backscatter",
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["SCATTWIN total", "PENETRATE total"])
    ax.set_ylabel("Counts")
    ax.set_title("SCATTWIN vs PENETRATE Total Components")
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _save_penetrate_count_bars(
    scatt_total: np.ndarray,
    scatt_scatter: np.ndarray,
    scatt_primary: np.ndarray,
    pen_outputs: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    scatt_primary_sum = float(scatt_primary.sum())
    scatt_scatter_sum = float(scatt_scatter.sum())

    scatt_primary_color = cm.Greens(0.6)
    scatt_scatter_color = cm.Reds(0.6)

    ax1.bar(
        "Total",
        scatt_primary_sum,
        color=scatt_primary_color,
        alpha=0.9,
        label="Primary",
    )
    ax1.bar(
        "Total",
        scatt_scatter_sum,
        bottom=scatt_primary_sum,
        color=scatt_scatter_color,
        alpha=0.9,
        label="Scatter",
    )
    ax1.bar("Primary", scatt_primary_sum, color=scatt_primary_color, alpha=0.9)
    ax1.bar("Scatter", scatt_scatter_sum, color=scatt_scatter_color, alpha=0.9)
    ax1.set_title("SCATTWIN Count Components")
    ax1.set_ylabel("Counts")
    ax1.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax1.legend()

    # Keep Geom Coll colors aligned with SCATTWIN primary/scatter bars so the
    # two routine panels are directly comparable at a glance.
    primary_components = [
        ("geom_coll_primary", scatt_primary_color, "Geom Coll"),
        ("septal_pen_primary", cm.Greens(0.85), "Septal Pen"),
        ("coll_scatter_primary", cm.Greens(0.45), "Coll Scatter"),
        ("coll_xray_primary", cm.Greens(0.30), "Coll X-ray"),
    ]
    scatter_components = [
        ("geom_coll_scattered", scatt_scatter_color, "Geom Coll"),
        ("septal_pen_scattered", cm.Reds(0.85), "Septal Pen"),
        ("coll_scatter_scattered", cm.Reds(0.45), "Coll Scatter"),
        ("coll_xray_scattered", cm.Reds(0.30), "Coll X-ray"),
    ]

    primary_backscatter_components = [
        "geom_coll_primary_back",
        "septal_pen_primary_back",
        "coll_scatter_primary_back",
        "coll_xray_primary_back",
    ]
    scatter_backscatter_components = [
        "geom_coll_scattered_back",
        "septal_pen_scattered_back",
        "coll_scatter_scattered_back",
        "coll_xray_scattered_back",
    ]
    primary_backscatter_total = sum(
        float(pen_outputs.get(comp, np.array(0.0)).sum())
        if comp in pen_outputs
        else 0.0
        for comp in primary_backscatter_components
    )
    scatter_backscatter_total = sum(
        float(pen_outputs.get(comp, np.array(0.0)).sum())
        if comp in pen_outputs
        else 0.0
        for comp in scatter_backscatter_components
    )

    primary_backscatter_color = cm.Greens(0.3)
    scatter_backscatter_color = cm.Reds(0.3)

    x_pos = 0
    bottom = 0.0
    for comp_slug, color, label in primary_components:
        if comp_slug in pen_outputs:
            val = float(pen_outputs[comp_slug].sum())
            if val > 0:
                ax2.bar(
                    x_pos,
                    val,
                    bottom=bottom,
                    color=color,
                    alpha=0.9,
                    width=0.8,
                    label=f"Primary: {label}",
                )
                bottom += val
    if primary_backscatter_total > 0:
        ax2.bar(
            x_pos,
            primary_backscatter_total,
            bottom=bottom,
            color=primary_backscatter_color,
            alpha=0.9,
            width=0.8,
            label="Primary Backscatter",
        )
        bottom += primary_backscatter_total

    for comp_slug, color, label in scatter_components:
        if comp_slug in pen_outputs:
            val = float(pen_outputs[comp_slug].sum())
            if val > 0:
                ax2.bar(
                    x_pos,
                    val,
                    bottom=bottom,
                    color=color,
                    alpha=0.9,
                    width=0.8,
                    label=f"Scatter: {label}",
                )
                bottom += val
    if scatter_backscatter_total > 0:
        ax2.bar(
            x_pos,
            scatter_backscatter_total,
            bottom=bottom,
            color=scatter_backscatter_color,
            alpha=0.9,
            width=0.8,
            label="Scatter Backscatter",
        )
        bottom += scatter_backscatter_total

    x_pos = 1
    bottom = 0.0
    for comp_slug, color, _ in primary_components:
        if comp_slug in pen_outputs:
            val = float(pen_outputs[comp_slug].sum())
            if val > 0:
                ax2.bar(x_pos, val, bottom=bottom, color=color, alpha=0.9, width=0.8)
                bottom += val
    if primary_backscatter_total > 0:
        ax2.bar(
            x_pos,
            primary_backscatter_total,
            bottom=bottom,
            color=primary_backscatter_color,
            alpha=0.9,
            width=0.8,
        )
        bottom += primary_backscatter_total

    x_pos = 2
    bottom = 0.0
    for comp_slug, color, _ in scatter_components:
        if comp_slug in pen_outputs:
            val = float(pen_outputs[comp_slug].sum())
            if val > 0:
                ax2.bar(x_pos, val, bottom=bottom, color=color, alpha=0.9, width=0.8)
                bottom += val
    if scatter_backscatter_total > 0:
        ax2.bar(
            x_pos,
            scatter_backscatter_total,
            bottom=bottom,
            color=scatter_backscatter_color,
            alpha=0.9,
            width=0.8,
        )
        bottom += scatter_backscatter_total

    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["Total", "Primary", "Scatter"])
    ax2.set_title("PENETRATE Count Components (Stacked)")
    ax2.set_ylabel("Counts")
    ax2.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> None:
    require_simind()

    output_dir = Path("output/routine_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    source, mu_map = build_small_phantom_zyx()

    scatt_dir = output_dir / "scattwin"
    pen_dir = output_dir / "penetrate"
    scatt_dir.mkdir(parents=True, exist_ok=True)
    pen_dir.mkdir(parents=True, exist_ok=True)

    scatt_outputs = _run_case(scatt_dir, "scatt_case01", source, mu_map, scoring_routine=1)
    pen_outputs = _run_case(pen_dir, "pen_case01", source, mu_map, scoring_routine=4)

    scatt_total = scatt_outputs.get("tot_w1", scatt_outputs[sorted(scatt_outputs)[0]])
    scatt_scatter = scatt_outputs.get("sca_w1", np.zeros_like(scatt_total))
    scatt_primary = np.clip(scatt_total - scatt_scatter, a_min=0.0, a_max=None)
    scatt_primary_sum = float(scatt_primary.sum())
    scatt_scatter_sum = float(scatt_scatter.sum())

    pen_key, pen_preview = _pick_penetrate_preview(pen_outputs)

    np.save(output_dir / "scatt_total.npy", scatt_total)
    np.save(output_dir / "scatt_scatter.npy", scatt_scatter)
    np.save(output_dir / "scatt_primary.npy", scatt_primary)
    np.save(output_dir / f"penetrate_{pen_key}.npy", pen_preview)

    with open(output_dir / "penetrate_components.txt", "w", encoding="utf-8") as handle:
        for key in sorted(pen_outputs):
            handle.write(f"{key}\t{float(pen_outputs[key].sum()):.6f}\n")

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(source[source.shape[0] // 2, :, :], cmap="viridis")
    axes[0].set_title("Input source")
    scatt_total_view0 = projection_view0(scatt_total)
    scatt_primary_view0 = projection_view0(scatt_primary)
    pen_preview_view0 = projection_view0(pen_preview)
    proj_vmin = float(
        min(scatt_total_view0.min(), scatt_primary_view0.min(), pen_preview_view0.min())
    )
    proj_vmax = float(
        max(scatt_total_view0.max(), scatt_primary_view0.max(), pen_preview_view0.max())
    )
    axes[1].imshow(scatt_total_view0, cmap="magma", vmin=proj_vmin, vmax=proj_vmax)
    axes[1].set_title("SCATTWIN total")
    axes[2].imshow(scatt_primary_view0, cmap="magma", vmin=proj_vmin, vmax=proj_vmax)
    axes[2].set_title("SCATTWIN primary")
    axes[3].imshow(pen_preview_view0, cmap="magma", vmin=proj_vmin, vmax=proj_vmax)
    axes[3].set_title(f"PENETRATE {pen_key}")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plot_path = output_dir / "routine_comparison.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    count_plot_path = output_dir / "routine_component_counts.png"
    _save_penetrate_count_bars(
        scatt_total,
        scatt_scatter,
        scatt_primary,
        pen_outputs,
        count_plot_path,
    )
    total_compare_path = output_dir / "routine_total_comparison.png"
    _save_total_bar_comparison(
        scatt_primary_sum,
        scatt_scatter_sum,
        pen_outputs,
        total_compare_path,
    )

    print(f"Output directory: {output_dir}")
    print(f"SCATTWIN total counts: {float(scatt_total.sum()):.3f}")
    print(f"SCATTWIN scatter counts: {float(scatt_scatter.sum()):.3f}")
    print(f"SCATTWIN primary counts: {float(scatt_primary.sum()):.3f}")
    print(f"PENETRATE preview key: {pen_key}")
    print(f"PENETRATE preview counts: {float(pen_preview.sum()):.3f}")
    print(f"Summary plot: {plot_path}")
    print(f"Count-bar plot: {count_plot_path}")
    print(f"Total-compare plot: {total_compare_path}")


if __name__ == "__main__":
    main()

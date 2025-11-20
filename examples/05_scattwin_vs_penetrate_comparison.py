#!/usr/bin/env python
"""
SCATTWIN vs PENETRATE Routine Comparison (Updated)

Simple comparison of SIMIND's two scoring routines using the new API.

Compatible with both SIRF and STIR Python backends.
"""

import argparse
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from sirf_simind_connection import SimindSimulator, SimulationConfig, configs, utils
from sirf_simind_connection.backends import get_backend, set_backend
from sirf_simind_connection.core.components import ScoringRoutine
from sirf_simind_connection.utils import get_array


# Create output directory
output_dir = Path("output/routine_comparison")
output_dir.mkdir(parents=True, exist_ok=True)


def main(collimator="megp", source_type="y90_tissue"):
    """Run comparison between SCATTWIN and PENETRATE routines."""
    print("Creating phantom and attenuation map...")
    phantom = utils.stir_utils.create_simple_phantom()
    mu_map = utils.stir_utils.create_attenuation_map(phantom)

    # Save phantom for reference
    phantom.write(str(output_dir / "phantom.hv"))
    mu_map.write(str(output_dir / "mu_map.hv"))

    # Load common configuration
    config = SimulationConfig(configs.get("input.smc"))
    config.import_yaml(configs.get("AnyScan.yaml"))

    print(f"Using collimator: {collimator}")
    print(f"Using source type: {source_type}")

    print("\n" + "=" * 50)
    print("Running SCATTWIN simulation...")
    print("=" * 50)

    # SCATTWIN simulation using new API
    simulator_scatt = SimindSimulator(
        config_source=config,
        output_dir=output_dir / "scattwin",
        output_prefix="scatt_sim",
        photon_multiplier=1,
        scoring_routine=ScoringRoutine.SCATTWIN,
    )

    # Set inputs
    simulator_scatt.set_source(phantom)
    simulator_scatt.set_mu_map(mu_map)

    # Set energy window to match PENETRATE (75-225 keV from AnyScan.yaml)
    simulator_scatt.set_energy_windows(
        lower_bounds=[75], upper_bounds=[225], scatter_orders=[0]
    )

    simulator_scatt.add_runtime_switch("CC", collimator)
    simulator_scatt.add_runtime_switch("FI", source_type)

    # Set photon energy
    simulator_scatt.add_config_value(1, 150.0)

    simulator_scatt.run_simulation()

    # Get SCATTWIN results using new API
    scatt_total = simulator_scatt.get_total_output(window=1)
    scatt_scatter = simulator_scatt.get_scatter_output(window=1)
    scatt_primary = scatt_total - scatt_scatter

    print("SCATTWIN Results:")
    print(f"  Total counts: {scatt_total.sum():.0f}")
    print(f"  Scatter counts: {scatt_scatter.sum():.0f}")
    print(f"  Primary counts: {scatt_primary.sum():.0f}")
    print(f"  Scatter fraction: {scatt_scatter.sum() / scatt_total.sum():.2%}")

    print("\n" + "=" * 50)
    print("Running PENETRATE simulation...")
    print("=" * 50)

    # PENETRATE simulation using new API
    simulator_pen = SimindSimulator(
        config_source=config,
        output_dir=output_dir / "penetrate",
        output_prefix="pen_sim",
        photon_multiplier=1,
        scoring_routine=ScoringRoutine.PENETRATE,
    )

    # Set inputs (same as SCATTWIN)
    simulator_pen.set_source(phantom)
    simulator_pen.set_mu_map(mu_map)

    simulator_pen.add_runtime_switch("CC", collimator)
    simulator_pen.add_runtime_switch("FI", source_type)

    simulator_pen.add_config_value(1, 150.0)

    simulator_pen.run_simulation()

    # Get PENETRATE results
    pen_outputs = simulator_pen.get_outputs()

    print("PENETRATE Results:")
    print(f"  Available outputs: {len(pen_outputs)}")
    for key, data in pen_outputs.items():
        print(f"  {key}: {data.sum():.0f} counts")

    print("\n" + "=" * 50)
    print("Creating comparison plots...")
    print("=" * 50)

    # Create comparison plots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # SCATTWIN plots (top row) - only use first 3 columns
    axes[0, 0].imshow(get_array(scatt_total)[0, :, 32, :], cmap="hot")
    axes[0, 0].set_title("SCATTWIN: Total Counts")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(get_array(scatt_primary)[0, :, 32, :], cmap="hot")
    axes[0, 1].set_title("SCATTWIN: Primary Counts")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(get_array(scatt_scatter)[0, :, 32, :], cmap="hot")
    axes[0, 2].set_title("SCATTWIN: Scatter Counts")
    axes[0, 2].axis("off")

    # Hide the 4th column in top row
    axes[0, 3].axis("off")

    # PENETRATE plots (bottom row) - show key components
    pen_keys = [
        "all_interactions",
        "geom_coll_primary",
        "geom_coll_scattered",
        "septal_pen_primary",
    ]
    for i, key in enumerate(pen_keys):
        if key in pen_outputs:
            axes[1, i].imshow(get_array(pen_outputs[key])[0, :, 32, :], cmap="plasma")
            axes[1, i].set_title(f"PENETRATE: {key.replace('_', ' ').title()}")
            axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "routine_comparison.png", dpi=150, bbox_inches="tight")

    # Create count comparison bar plot with stacked bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # SCATTWIN bar plot - stacked to show composition
    scatt_primary_sum = scatt_primary.sum()
    scatt_scatter_sum = scatt_scatter.sum()

    # Use mid-tone colors from the colormaps (not the dark ones like PENETRATE geom_coll)
    scatt_primary_color = cm.Greens(0.6)  # Mid green
    scatt_scatter_color = cm.Reds(0.6)  # Mid red

    # Total bar is stacked (primary + scatter)
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

    # Individual bars for primary and scatter
    ax1.bar("Primary", scatt_primary_sum, color=scatt_primary_color, alpha=0.9)
    ax1.bar("Scatter", scatt_scatter_sum, color=scatt_scatter_color, alpha=0.9)

    ax1.set_title("SCATTWIN Count Components")
    ax1.set_ylabel("Counts")
    ax1.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax1.legend()

    # PENETRATE stacked bar plot
    # b02-b09 are without backscatter (individual colors)
    # b10-b17 are with backscatter (shown as two total layers: primary and scatter)

    # Use colormaps - Greens for primary (dark to light), Reds for scatter (dark to light)
    greens = cm.Greens(np.linspace(0.5, 0.9, 4))  # Dark to light green
    reds = cm.Reds(np.linspace(0.5, 0.9, 4))  # Dark to light red

    # Define primary components b02-b05 (using colormap)
    primary_components = [
        ("geom_coll_primary", greens[0], "Geom Coll"),
        ("septal_pen_primary", greens[1], "Septal Pen"),
        ("coll_scatter_primary", greens[2], "Coll Scatter"),
        ("coll_xray_primary", greens[3], "Coll X-ray"),
    ]

    # Define scatter components b06-b09 (using colormap)
    scatter_components = [
        ("geom_coll_scattered", reds[0], "Geom Coll"),
        ("septal_pen_scattered", reds[1], "Septal Pen"),
        ("coll_scatter_scattered", reds[2], "Coll Scatter"),
        ("coll_xray_scattered", reds[3], "Coll X-ray"),
    ]

    # Calculate backscatter totals (b10-b13 for primary, b14-b17 for scatter)
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
        pen_outputs.get(comp, 0).sum() if comp in pen_outputs else 0
        for comp in primary_backscatter_components
    )
    scatter_backscatter_total = sum(
        pen_outputs.get(comp, 0).sum() if comp in pen_outputs else 0
        for comp in scatter_backscatter_components
    )

    # Build three bars: Total, Primary, Scatter
    # Colors for backscatter - use very dark from the colormap
    primary_backscatter_color = cm.Greens(0.3)  # Very dark green for backscatter
    scatter_backscatter_color = cm.Reds(0.3)  # Very dark red for backscatter

    # Bar 0: Total (all primary components grouped together, then scatter)
    x_pos = 0
    bottom = 0

    # Stack individual primary components (b02-b05) first
    for comp_slug, color, label in primary_components:
        if comp_slug in pen_outputs:
            val = pen_outputs[comp_slug].sum()
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

    # Add total primary backscatter layer (b10-b13) on top of primary
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

    # Now stack scatter components (b06-b09)
    for comp_slug, color, label in scatter_components:
        if comp_slug in pen_outputs:
            val = pen_outputs[comp_slug].sum()
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

    # Add total scatter backscatter layer (b14-b17) on top
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

    # Bar 1: Primary only
    x_pos = 1
    bottom = 0

    # Stack individual primary components (b02-b05)
    for comp_slug, color, _ in primary_components:
        if comp_slug in pen_outputs:
            val = pen_outputs[comp_slug].sum()
            if val > 0:
                ax2.bar(x_pos, val, bottom=bottom, color=color, alpha=0.9, width=0.8)
                bottom += val

    # Add total primary backscatter layer (b10-b13)
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

    # Bar 2: Scatter only
    x_pos = 2
    bottom = 0

    # Stack individual scatter components (b06-b09)
    for comp_slug, color, _ in scatter_components:
        if comp_slug in pen_outputs:
            val = pen_outputs[comp_slug].sum()
            if val > 0:
                ax2.bar(x_pos, val, bottom=bottom, color=color, alpha=0.9, width=0.8)
                bottom += val

    # Add total scatter backscatter layer (b14-b17)
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
    plt.savefig(output_dir / "count_comparison.png", dpi=150, bbox_inches="tight")

    print("\nComparison complete!")
    print(f"Results saved to: {output_dir}")
    print("- routine_comparison.png: Projection images")
    print("- count_comparison.png: Count statistics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare SCATTWIN and PENETRATE scoring routines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["sirf", "stir"],
        help="Force a specific backend (sirf or stir). If not specified, auto-detection is used.",
    )
    parser.add_argument(
        "--collimator",
        type=str,
        default="megp",
        help="Collimator type to use (default: megp)",
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default="y90_tissue",
        help="Source type to use (default: y90_tissue)",
    )
    args = parser.parse_args()

    # Set backend if specified
    if args.backend:
        set_backend(args.backend)

    # Print which backend is being used
    print(f"\n{'=' * 60}")
    print(f"Using backend: {get_backend().upper()}")
    print(f"{'=' * 60}\n")

    main(collimator=args.collimator, source_type=args.source_type)

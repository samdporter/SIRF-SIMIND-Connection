#!/usr/bin/env python
"""
SCATTWIN vs PENETRATE Routine Comparison (Updated)

Simple comparison of SIMIND's two scoring routines using the new API.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from sirf_simind_connection import SimindSimulator, SimulationConfig, configs, utils
from sirf_simind_connection.core.components import ScoringRoutine

# Create output directory
output_dir = Path("output/routine_comparison")
output_dir.mkdir(parents=True, exist_ok=True)


def main():
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

    # Set energy window for Tc-99m (140 keV ± 10%)
    simulator_scatt.set_energy_windows(
        lower_bounds=[126], upper_bounds=[154], scatter_orders=[0]
    )

    # Set photon energy
    simulator_scatt.add_config_value(1, 140.0)

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

    # Set photon energy (no energy windows needed for PENETRATE)
    simulator_pen.add_config_value(1, 140.0)

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
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # SCATTWIN plots (top row)
    axes[0, 0].imshow(scatt_total.as_array()[0, :, 32, :], cmap="hot")
    axes[0, 0].set_title("SCATTWIN: Total Counts")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(scatt_primary.as_array()[0, :, 32, :], cmap="hot")
    axes[0, 1].set_title("SCATTWIN: Primary Counts")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(scatt_scatter.as_array()[0, :, 32, :], cmap="hot")
    axes[0, 2].set_title("SCATTWIN: Scatter Counts")
    axes[0, 2].axis("off")

    # PENETRATE plots (bottom row) - show first 3 available outputs
    pen_keys = list(pen_outputs.keys())[:3]
    for i, key in enumerate(pen_keys):
        axes[1, i].imshow(pen_outputs[key].as_array()[0, :, 32, :], cmap="plasma")
        axes[1, i].set_title(f"PENETRATE: {key.replace('_', ' ').title()}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "routine_comparison.png", dpi=150, bbox_inches="tight")

    # Create count comparison bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # SCATTWIN bar plot
    ax1.bar(
        ["Total", "Primary", "Scatter"],
        [scatt_total.sum(), scatt_primary.sum(), scatt_scatter.sum()],
        color=["blue", "green", "red"],
        alpha=0.7,
    )
    ax1.set_title("SCATTWIN Count Components")
    ax1.set_ylabel("Counts")
    ax1.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # PENETRATE bar plot (top 5 components)
    pen_counts = [(k, v.sum()) for k, v in pen_outputs.items()]
    pen_counts.sort(key=lambda x: x[1], reverse=True)
    top_5 = pen_counts[:5]

    names = [name.replace("_", "\n") for name, _ in top_5]
    values = [count for _, count in top_5]

    ax2.bar(names, values, color="purple", alpha=0.7)
    ax2.set_title("PENETRATE Top 5 Components")
    ax2.set_ylabel("Counts")
    ax2.tick_params(axis="x", rotation=45)
    ax2.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(output_dir / "count_comparison.png", dpi=150, bbox_inches="tight")

    print("\nComparison complete!")
    print(f"Results saved to: {output_dir}")
    print("- routine_comparison.png: Projection images")
    print("- count_comparison.png: Count statistics")


if __name__ == "__main__":
    main()

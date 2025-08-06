#!/usr/bin/env python
"""
Basic SIMIND Simulation Example (Updated)

This example demonstrates how to run a basic SPECT Monte Carlo simulation
using SIRF-SIMIND-Connection with the new API.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from sirf_simind_connection import SimindSimulator, SimulationConfig, configs, utils
from sirf_simind_connection.core.components import ScoringRoutine


# Create output directory
output_dir = Path("output/basic_simulation")
output_dir.mkdir(parents=True, exist_ok=True)


def main():
    """Run the basic simulation."""
    print("Creating phantom and attenuation map...")
    phantom = utils.stir_utils.create_simple_phantom()
    mu_map = utils.stir_utils.create_attenuation_map(phantom)

    # Save phantom for visualization
    phantom.write(str(output_dir / "phantom.hv"))
    mu_map.write(str(output_dir / "mu_map.hv"))

    print("Setting up SIMIND simulator using new API...")

    # Load and configure the simulation config
    config = SimulationConfig(configs.get("input.smc"))
    config.import_yaml(configs.get("AnyScan.yaml"))

    # Method 1: Using the new constructor directly
    simulator = SimindSimulator(
        config_source=config,
        output_dir=output_dir,
        output_prefix="basic_sim",
        photon_multiplier=1,
        scoring_routine=ScoringRoutine.SCATTWIN,
    )

    # Set the inputs using new methods
    simulator.set_source(phantom)
    simulator.set_mu_map(mu_map)

    # Set energy windows for Tc-99m (140 keV ± 10%)
    simulator.set_energy_windows(
        lower_bounds=[126],  # 140 - 14 keV
        upper_bounds=[154],  # 140 + 14 keV
        scatter_orders=[0],  # Include scatter
    )

    # Set photon energy for Tc-99m
    simulator.add_config_value(1, 140.0)  # 140 keV

    print("Running simulation (this may take a few minutes)...")
    simulator.run_simulation()

    print("Retrieving results...")
    # Get the output sinograms using new API
    total_counts = simulator.get_total_output(window=1)
    scatter_counts = simulator.get_scatter_output(window=1)
    primary_counts = total_counts - scatter_counts

    # Save results
    total_counts.write(str(output_dir / "total_sinogram.hs"))
    scatter_counts.write(str(output_dir / "scatter_sinogram.hs"))
    primary_counts.write(str(output_dir / "primary_sinogram.hs"))

    # Print statistics
    print("\nSimulation Results:")
    print(f"Total counts: {total_counts.sum():.0f}")
    print(f"Scatter counts: {scatter_counts.sum():.0f}")
    print(f"Primary counts: {primary_counts.sum():.0f}")
    print(f"Scatter fraction: {scatter_counts.sum() / total_counts.sum():.2%}")

    print(f"\nResults saved to: {output_dir}")

    # Create visualizations
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    axim0 = ax[0].imshow(phantom.as_array()[32, :, :], cmap="viridis")
    ax[0].set_title("Phantom (Axial Slice)")
    axim1 = ax[1].imshow(mu_map.as_array()[32, :, :], cmap="gray")
    ax[1].set_title("Attenuation Map (Axial Slice)")
    plt.colorbar(axim0, ax=ax[0])
    plt.colorbar(axim1, ax=ax[1])
    plt.tight_layout()
    plt.savefig(output_dir / "phantom_and_attenuation.png")

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    axim0 = ax[0].imshow(total_counts.as_array()[0, :, 32, :], cmap="hot")
    ax[0].set_title("Total Counts (Projection)")
    axim1 = ax[1].imshow(scatter_counts.as_array()[0, :, 32, :], cmap="hot")
    ax[1].set_title("Scatter Counts (Projection)")
    axim2 = ax[2].imshow(primary_counts.as_array()[0, :, 32, :], cmap="hot")
    ax[2].set_title("Primary Counts (Projection)")
    plt.colorbar(axim0, ax=ax[0])
    plt.colorbar(axim1, ax=ax[1])
    plt.colorbar(axim2, ax=ax[2])
    plt.tight_layout()
    plt.savefig(output_dir / "projection_results.png")

    print("Visualization plots saved!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Basic SIMIND Simulation Example

This example demonstrates how to run a basic SPECT Monte Carlo simulation
using SIRF-SIMIND-SPECT with a simple phantom.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from sirf_simind_connection import (SimindSimulator, SimulationConfig, configs,
                                    utils)

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

    print("Setting up SIMIND simulator...")
    # Use the provided scanner configuration
    config = SimulationConfig(configs.get("input.smc"))
    config.import_yaml(configs.get("AnyScan.yaml"))  # Load a scanner configuration
    config_file = config.save_file(str(output_dir / "sim_config.smc"))

    simulator = SimindSimulator(
        template_smc_file_path=config_file,
        output_dir=output_dir,
        output_prefix="basic_sim",
        source=phantom,
        mu_map=mu_map,
        photon_multiplier=10,
    )

    # Set energy window for Tc-99m (140 keV ± 10%)
    # due to a bug in SIMIND, we need an extra throwaway window
    print("Setting energy window...")
    simulator.set_windows(
        lower_bounds=[126],  # 140 - 14
        upper_bounds=[154],  # 140 + 14
        scatter_orders=[0],  # Up to 5th order scatter
    )

    print("Running simulation (this may take a few minutes)...")
    simulator.run_simulation()

    print("Retrieving results...")
    # Get the output sinograms
    total_counts = simulator.get_output_total(window=1)
    scatter_counts = simulator.get_output_scatter(window=1)
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

    # view input images and resultant projections
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    axim0 = ax[0].imshow(phantom.as_array()[32, :, :], cmap="viridis")
    ax[0].set_title("Phantom (Axial Slice)")
    axim1 = ax[1].imshow(mu_map.as_array()[32, :, :], cmap="gray")
    ax[1].set_title("Total Counts (Axial Slice)")
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


if __name__ == "__main__":
    main()

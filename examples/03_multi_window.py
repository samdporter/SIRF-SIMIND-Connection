#!/usr/bin/env python
"""
Multi-Energy Window Simulation Example (Updated)

This example demonstrates how to simulate SPECT with multiple energy windows,
useful for scatter correction techniques like TEW (Triple Energy Window).
"""

from pathlib import Path

import matplotlib.pyplot as plt

from sirf_simind_connection import SimindSimulator, SimulationConfig, configs, utils
from sirf_simind_connection.core.components import ScoringRoutine


def setup_tew_windows():
    """
    Set up Triple Energy Window (TEW) for Tc-99m.

    Returns:
        tuple: (lower_bounds, upper_bounds, scatter_orders)
    """
    # Main photopeak window: 140 keV ± 10% (126-154 keV)
    # Lower scatter window: 120-126 keV
    # Upper scatter window: 154-160 keV

    lower_bounds = [120, 126, 154]  # keV
    upper_bounds = [126, 154, 160]  # keV
    scatter_orders = [0, 0, 0]  # Include high-order scatter

    return lower_bounds, upper_bounds, scatter_orders


def setup_dew_windows():
    """
    Set up Dual Energy Window (DEW) for I-123.

    Returns:
        tuple: (lower_bounds, upper_bounds, scatter_orders)
    """
    # I-123 has primary emission at 159 keV
    # Main window: 159 keV ± 10% (143-175 keV)
    # Scatter window: 120-143 keV

    lower_bounds = [120, 143]  # keV
    upper_bounds = [143, 175]  # keV
    scatter_orders = [10, 10]

    return lower_bounds, upper_bounds, scatter_orders


def calculate_tew_correction(outputs, window_widths):
    """
    Calculate TEW scatter correction.

    Args:
        outputs: Dictionary of simulation outputs
        window_widths: List of energy window widths in keV

    Returns:
        Corrected photopeak data
    """
    # TEW assumes linear interpolation of scatter under photopeak
    lower_scatter = outputs["tot_w1"]
    photopeak_total = outputs["tot_w2"]
    upper_scatter = outputs["tot_w3"]

    # Scatter estimate under photopeak
    scatter_estimate = lower_scatter * (
        window_widths[1] / (2 * window_widths[0])
    ) + upper_scatter * (window_widths[1] / (2 * window_widths[2]))

    # Corrected photopeak
    corrected = (photopeak_total - scatter_estimate).maximum(0)  # Ensure non-negative

    return corrected, scatter_estimate


def main():
    """Run multi-window simulation."""
    output_dir = Path("output/multi_window")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating phantom...")
    phantom = utils.stir_utils.create_simple_phantom()
    mu_map = utils.stir_utils.create_attenuation_map(phantom)

    # Save phantom
    phantom.write(str(output_dir / "phantom.hv"))

    # Choose window configuration
    print("\nSetting up Triple Energy Window (TEW) for Tc-99m...")
    lower_bounds, upper_bounds, scatter_orders = setup_tew_windows()
    window_widths = [upper - lower for lower, upper in zip(lower_bounds, upper_bounds)]

    print("Energy windows:")
    for i, (lower, upper) in enumerate(zip(lower_bounds, upper_bounds)):
        print(f"  Window {i + 1}: {lower}-{upper} keV (width: {upper - lower} keV)")

    # Configure simulator using new API
    print("\nConfiguring SIMIND simulator...")
    config = SimulationConfig(configs.get("input.smc"))
    config.import_yaml(configs.get("AnyScan.yaml"))

    simulator = SimindSimulator(
        config_source=config,
        output_dir=output_dir,
        output_prefix="tew_sim",
        photon_multiplier=1,
        scoring_routine=ScoringRoutine.SCATTWIN,
    )

    # Set inputs using new methods
    simulator.set_source(phantom)
    simulator.set_mu_map(mu_map)

    # Set multiple energy windows
    simulator.set_energy_windows(lower_bounds, upper_bounds, scatter_orders)

    # Set Tc-99m parameters
    simulator.add_config_value(1, 140.0)  # Photon energy

    print("Running simulation...")
    simulator.run_simulation()

    print("Retrieving results...")
    outputs = simulator.get_outputs()

    # Print results for each window
    print("\nRaw window counts:")
    for key, data in outputs.items():
        print(f"  {key}: {data.sum():.0f} counts")

    # Apply TEW correction
    print("\nApplying TEW scatter correction...")
    corrected, scatter_estimate = calculate_tew_correction(outputs, window_widths)

    # Save results
    total_counts = outputs["tot_w2"]
    total_counts.write(str(output_dir / "photopeak_total.hs"))
    scatter_estimate.write(str(output_dir / "scatter_estimate.hs"))
    corrected.write(str(output_dir / "photopeak_corrected.hs"))
    scatter_counts = outputs["sca_w2"]
    scatter_counts.write(str(output_dir / "scatter_counts.hs"))
    true_counts = total_counts - scatter_counts
    true_counts.write(str(output_dir / "true_counts.hs"))

    # Calculate scatter fractions
    print("\nScatter analysis:")
    print(
        f"  True scatter fraction: {scatter_counts.sum() / outputs['tot_w2'].sum():.2%}"
    )
    print(f"  TEW estimated scatter: {scatter_estimate.sum():.0f}")
    print(f"  TEW corrected counts: {corrected.sum():.0f}")

    # view input images and resultant projections
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    axim0 = ax[0].imshow(phantom.as_array()[32, :, :], cmap="viridis")
    ax[0].set_title("Phantom (Axial Slice)")
    axim1 = ax[1].imshow(mu_map.as_array()[32, :, :], cmap="gray")
    ax[1].set_title("Attenuation Map (Axial Slice)")
    plt.colorbar(axim0, ax=ax[0])
    plt.colorbar(axim1, ax=ax[1])
    plt.tight_layout()
    plt.savefig(output_dir / "phantom_and_attenuation.png")

    fig, ax = plt.subplots(1, 5, figsize=(18, 6))
    axim0 = ax[0].imshow(scatter_counts.as_array()[0, :, 60, :], cmap="hot")
    ax[0].set_title("Scatter Counts (Projection)")
    axim1 = ax[1].imshow(scatter_estimate.as_array()[0, :, 60, :], cmap="hot")
    ax[1].set_title("TEW Scatter Estimate (Projection)")
    axim2 = ax[2].imshow(true_counts.as_array()[0, :, 60, :], cmap="hot")
    ax[2].set_title("True Counts (Projection)")
    axim3 = ax[3].imshow(corrected.as_array()[0, :, 60, :], cmap="hot")
    ax[3].set_title("TEW Corrected Counts (Projection)")
    axim4 = ax[4].imshow(total_counts.as_array()[0, :, 60, :], cmap="hot")
    ax[4].set_title("Total Counts (Projection)")
    plt.colorbar(axim0, ax=ax[0])
    plt.colorbar(axim1, ax=ax[1])
    plt.colorbar(axim2, ax=ax[2])
    plt.colorbar(axim3, ax=ax[3])
    plt.colorbar(axim4, ax=ax[4])
    plt.tight_layout()
    plt.savefig(output_dir / "projection_results.png")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

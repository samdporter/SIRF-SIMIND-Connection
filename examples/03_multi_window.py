#!/usr/bin/env python
"""
Multi-Energy Window Simulation Example

This example demonstrates how to simulate SPECT with multiple energy windows,
useful for scatter correction techniques like TEW (Triple Energy Window).
"""

from pathlib import Path

import numpy as np

from sirf_simind_connection import SimindSimulator, create_stir_image


def create_test_phantom():
    """Create a simple test phantom."""
    # Create 64x64x64 image for faster simulation
    matrix_dim = [64, 64, 64]
    voxel_size = [6.0, 6.0, 6.0]  # mm

    phantom = create_stir_image(matrix_dim, voxel_size)
    phantom_array = np.zeros(matrix_dim)

    # Add uniform cylinder
    center = [32, 32, 32]
    radius = 20
    for z in range(10, 54):
        for y in range(64):
            for x in range(64):
                if (x - center[1]) ** 2 + (y - center[2]) ** 2 <= radius**2:
                    phantom_array[z, y, x] = 100

    phantom.fill(phantom_array)
    return phantom


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
    scatter_orders = [10, 10, 10]  # Include high-order scatter

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
    corrected = photopeak_total - scatter_estimate
    corrected.maximum(0)  # Ensure non-negative

    return corrected, scatter_estimate


def main():
    """Run multi-window simulation."""
    output_dir = Path("output/multi_window")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating phantom...")
    phantom = create_test_phantom()

    # Simple uniform attenuation
    mu_map = phantom.clone()
    mu_map.fill(0.15 * (phantom.as_array() > 0))

    # Save phantom
    phantom.write(str(output_dir / "phantom.hv"))

    # Choose window configuration
    print("\nSetting up Triple Energy Window (TEW) for Tc-99m...")
    lower_bounds, upper_bounds, scatter_orders = setup_tew_windows()
    window_widths = [u - l for l, u in zip(lower_bounds, upper_bounds)]

    print("Energy windows:")
    for i, (l, u) in enumerate(zip(lower_bounds, upper_bounds)):
        print(f"  Window {i + 1}: {l}-{u} keV (width: {u - l} keV)")

    # Configure simulator
    print("\nConfiguring SIMIND simulator...")
    config_file = Path(__file__).parent.parent / "configs" / "Discovery670.yaml"

    simulator = SimindSimulator(
        template_smc_file_path=config_file,
        output_dir=output_dir,
        output_prefix="tew_sim",
        source=phantom,
        mu_map=mu_map,
        photon_multiplier=5e5,  # Fewer histories for example
    )

    # Set Tc-99m parameters
    simulator.add_index(1, 140.0)  # Photon energy

    # Set multiple windows
    simulator.set_windows(lower_bounds, upper_bounds, scatter_orders)

    print("Running simulation...")
    simulator.run_simulation()

    print("Retrieving results...")
    outputs = simulator.get_output()

    # Print results for each window
    print("\nRaw window counts:")
    for key, data in outputs.items():
        print(f"  {key}: {data.sum():.0f} counts")

    # Apply TEW correction
    print("\nApplying TEW scatter correction...")
    corrected, scatter_estimate = calculate_tew_correction(outputs, window_widths)

    # Save results
    outputs["tot_w2"].write(str(output_dir / "photopeak_total.hs"))
    scatter_estimate.write(str(output_dir / "scatter_estimate.hs"))
    corrected.write(str(output_dir / "photopeak_corrected.hs"))

    # Calculate scatter fractions
    true_scatter = outputs["sca_w2"]
    print("\nScatter analysis:")
    print(
        f"  True scatter fraction: {true_scatter.sum() / outputs['tot_w2'].sum():.2%}"
    )
    print(f"  TEW estimated scatter: {scatter_estimate.sum():.0f}")
    print(f"  TEW corrected counts: {corrected.sum():.0f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

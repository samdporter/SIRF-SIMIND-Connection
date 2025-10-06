#!/usr/bin/env python3
"""
Example 08: SimindCoordinator for Multi-Subset Reconstruction

This example demonstrates:
1. Setting up SimindCoordinator to manage SIMIND across multiple subsets
2. Using SimindSubsetProjector for efficient subset-based reconstruction
3. Comparing coordinator-based vs standard OSEM reconstruction

The coordinator runs ONE full SIMIND simulation (all projections) and distributes
results to multiple subset projectors. This is more efficient than running separate
simulations for each subset.

Uses SCATTWIN scoring with collimator_routine=0 for fast execution.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sirf.STIR import (
    AcquisitionModelUsingRayTracingMatrix,
    OSMAPOSLReconstructor,
)

from sirf_simind_connection import SimindSimulator, SimulationConfig
from sirf_simind_connection.configs import get
from sirf_simind_connection.core.components import ScoringRoutine
from sirf_simind_connection.core.coordinator import SimindCoordinator
from sirf_simind_connection.core.projector import SimindSubsetProjector
from sirf_simind_connection.utils.stir_utils import (
    create_attenuation_map,
    create_simple_phantom,
)


def create_fast_config():
    """Create SIMIND config for fast simulation."""
    config = SimulationConfig(get("AnyScan.yaml"))

    config.set_value(26, 0.1)  # number_photon_histories (10^5 photons)
    config.set_value(29, 60)  # spect_no_projections (60 views)
    config.set_value(76, 64)  # matrix_size_image_i
    config.set_value(77, 64)  # matrix_size_image_j
    config.set_value(28, 0.442)  # pixel_size_simulated_image (cm)
    config.set_value(53, 0)  # collimator_routine = 0 (geometric only)
    config.set_value(84, 1)  # scoring_routine = 1 (SCATTWIN)

    return config


def run_standard_osem(measured_data, initial_estimate, num_subsets=6, num_iters=24):
    """Run standard OSEM for comparison."""
    print("\n=== Standard OSEM (Baseline) ===")

    am = AcquisitionModelUsingRayTracingMatrix()
    am.set_up(measured_data, initial_estimate)

    recon = OSMAPOSLReconstructor()
    recon.set_acquisition_model(am)
    recon.set_input(measured_data)
    recon.set_num_subsets(num_subsets)
    recon.set_num_subiterations(num_iters)

    recon.set_up(initial_estimate)
    print(f"  Running {num_iters} subiterations...")
    recon.process()

    return recon.get_output()


def run_coordinator_osem(
    coordinator, measured_data, initial_estimate, num_subsets=6, num_iters=24
):
    """Run OSEM-style reconstruction using SimindCoordinator."""
    print("\n=== OSEM with SimindCoordinator ===")

    # Create subset projectors
    subset_projectors = []
    for subset_idx in range(num_subsets):
        # Create STIR projector for this subset
        stir_am = AcquisitionModelUsingRayTracingMatrix()
        stir_am.set_up(measured_data, initial_estimate)
        stir_am.set_num_subsets(num_subsets)
        stir_am.set_subset_num(subset_idx)

        # Calculate subset view indices
        total_views = 60  # from config
        views_per_subset = total_views // num_subsets
        subset_indices = list(
            range(subset_idx * views_per_subset, (subset_idx + 1) * views_per_subset)
        )

        # Create SimindSubsetProjector
        subset_proj = SimindSubsetProjector(
            stir_projector=stir_am,
            coordinator=coordinator,
            subset_indices=subset_indices,
        )

        subset_projectors.append(subset_proj)

    print(f"  Created {len(subset_projectors)} subset projectors")
    print(f"  Each subset has {60 // num_subsets} views")

    # Manual OSEM loop
    current_image = initial_estimate.clone()
    num_epochs = num_iters // num_subsets

    print(f"  Running {num_epochs} epochs × {num_subsets} subsets...")

    for epoch in range(num_epochs):
        for subset_idx in range(num_subsets):
            # Forward project
            proj = subset_projectors[subset_idx]
            fwd = proj.forward(current_image)

            # Compute ratio (measured / estimated)
            fwd_arr = fwd.as_array()
            measured_arr = measured_data.as_array()

            ratio = measured_data.clone()
            ratio.fill(np.divide(measured_arr, fwd_arr + 1e-10))

            # Backproject ratio
            correction = proj.backward(ratio)

            # Multiplicative update
            current_arr = current_image.as_array()
            corr_arr = correction.as_array()
            current_arr *= corr_arr / num_subsets
            current_image.fill(np.maximum(current_arr, 0))  # Enforce non-negativity

        if (epoch + 1) % max(1, num_epochs // 4) == 0:
            print(f"    Epoch {epoch + 1}/{num_epochs} complete")

    return current_image


def plot_comparison(phantom, measured_data, recon_std, recon_coord, output_dir):
    """Create comparison plots."""
    phantom_arr = phantom.as_array()
    measured_arr = measured_data.as_array()
    std_arr = recon_std.as_array()
    coord_arr = recon_coord.as_array()

    mid_z = phantom_arr.shape[0] // 2
    mid_proj = measured_arr.shape[0] // 2

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Phantom
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(phantom_arr[mid_z], cmap="hot")
    ax0.set_title("Ground Truth Phantom")
    ax0.axis("off")
    plt.colorbar(im0, ax=ax0, fraction=0.046)

    # Projection
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(measured_arr[mid_proj], cmap="gray", aspect="auto")
    ax1.set_title(f"SIMIND Projection (View {mid_proj})")
    ax1.set_xlabel("Detector X")
    ax1.set_ylabel("Detector Z")
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # Standard OSEM
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(std_arr[mid_z], cmap="hot")
    ax2.set_title("Standard OSEM")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # Coordinator OSEM
    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(coord_arr[mid_z], cmap="hot")
    ax3.set_title("Coordinator OSEM")
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # Line profiles
    ax4 = fig.add_subplot(gs[0:, 2])
    mid_x = phantom_arr.shape[2] // 2
    phantom_profile = phantom_arr[mid_z, :, mid_x]
    std_profile = std_arr[mid_z, :, mid_x]
    coord_profile = coord_arr[mid_z, :, mid_x]

    ax4.plot(phantom_profile, "k-", linewidth=2.5, label="Ground Truth", alpha=0.7)
    ax4.plot(std_profile, "b--", linewidth=2, label="Standard OSEM")
    ax4.plot(coord_profile, "r--", linewidth=2, label="Coordinator OSEM")
    ax4.set_title("Central Line Profile")
    ax4.set_xlabel("Position (pixels)")
    ax4.set_ylabel("Activity")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    output_path = Path(output_dir) / "coordinator_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to: {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("Example 08: SimindCoordinator for Multi-Subset Reconstruction")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nWorking directory: {temp_dir}\n")

        # ============================================================
        # STEP 1: Create Phantom and Setup
        # ============================================================
        print("[1/5] Creating phantom...")
        phantom = create_simple_phantom()
        mu_map = create_attenuation_map(phantom)
        print("  Phantom: 64×64×64 voxels, 4.42mm spacing")

        # ============================================================
        # STEP 2: Run SIMIND Simulation
        # ============================================================
        print("\n[2/5] Running SIMIND simulation for measured data...")
        config = create_fast_config()

        simulator = SimindSimulator(
            config,
            output_dir=temp_dir,
            scoring_routine=ScoringRoutine.SCATTWIN,
        )

        simulator.set_source(phantom)
        simulator.set_mu_map(mu_map)
        simulator.set_energy_windows([126], [154], [0])

        print("  SIMIND: 10^5 photons/projection, 60 projections, SCATTWIN")

        measured_data = simulator.run_simulation()

        # Add noise
        measured_arr = measured_data.as_array()
        scale = 1e6 / measured_arr.sum()
        measured_arr_noisy = np.random.poisson(measured_arr * scale) / scale
        measured_data.fill(measured_arr_noisy)

        print(f"  Total counts: {measured_data.as_array().sum():.2e}")

        # ============================================================
        # STEP 3: Create SimindCoordinator
        # ============================================================
        print("\n[3/5] Creating SimindCoordinator...")
        num_subsets = 6
        update_interval = 6  # Update every epoch

        # Create full-data linear acquisition model for coordinator
        linear_am = AcquisitionModelUsingRayTracingMatrix()
        linear_am.set_up(measured_data, phantom)
        linear_am.set_num_subsets(1)
        linear_am.set_subset_num(0)

        coordinator = SimindCoordinator(
            simind_simulator=simulator,
            num_subsets=num_subsets,
            correction_update_interval=update_interval,
            residual_correction=True,  # Mode A: residual only
            update_additive=False,
            linear_acquisition_model=linear_am,
            output_dir=temp_dir,
        )

        print("  Coordinator configured:")
        print(f"    - Subsets: {num_subsets}")
        print(f"    - Update interval: {update_interval} subiterations (every epoch)")
        print("    - Mode: Residual correction (fast)")

        # ============================================================
        # STEP 4: Run Reconstructions
        # ============================================================
        print("\n[4/5] Running reconstructions...")

        initial_estimate = phantom.get_uniform_copy(1.0)

        # Standard OSEM
        recon_std = run_standard_osem(
            measured_data,
            initial_estimate.clone(),
            num_subsets=num_subsets,
            num_iters=24,
        )

        # Coordinator OSEM
        recon_coord = run_coordinator_osem(
            coordinator,
            measured_data,
            initial_estimate.clone(),
            num_subsets=num_subsets,
            num_iters=24,
        )

        # ============================================================
        # STEP 5: Visualize
        # ============================================================
        print("\n[5/5] Generating plots...")
        plot_comparison(phantom, measured_data, recon_std, recon_coord, temp_dir)

        # ============================================================
        # Summary
        # ============================================================
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print("\nSimindCoordinator Key Features:")
        print("  1. Runs ONE SIMIND simulation for all subsets (efficient!)")
        print("  2. Distributes corrections to subset projectors")
        print("  3. Tracks global iteration count across subsets")
        print("  4. Updates corrections every N subiterations")

        print("\nWhen to Use SimindCoordinator:")
        print("  ✓ Subset-based reconstruction algorithms (OSEM, SPDHG, etc.)")
        print("  ✓ CIL-based reconstructions with multiple objectives")
        print("  ✓ When computational efficiency is important")
        print("  ✓ Advanced correction modes with cumulative additive tracking")

        print("\nWhen to Use SimindProjector Instead:")
        print("  ✓ Single projector use cases")
        print("  ✓ Simpler API preference")
        print("  ✓ Drop-in replacement for SIRF AcquisitionModel")

        print(f"\nOutput: {temp_dir}")
        print("=" * 80)


if __name__ == "__main__":
    main()

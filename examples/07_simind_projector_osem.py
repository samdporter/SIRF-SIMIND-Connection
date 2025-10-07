#!/usr/bin/env python3
"""
Example 07: SIMIND Simulation + OSEM Reconstruction with SimindProjector

This example demonstrates:
1. Running a basic SIMIND simulation to generate measured data
2. Using SimindProjector as the forward model in OSEM reconstruction
3. Comparing reconstructions with and without SIMIND corrections

Uses SCATTWIN scoring with collimator_routine=0 for fast execution.
No penetration physics - just geometric collimation.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sirf.STIR import (
    AcquisitionModelUsingRayTracingMatrix,
    OSMAPOSLReconstructor,
)

from sirf_simind_connection import SimindProjector, SimindSimulator, SimulationConfig
from sirf_simind_connection.configs import get
from sirf_simind_connection.core.components import ScoringRoutine
from sirf_simind_connection.utils import get_array
from sirf_simind_connection.utils.stir_utils import (
    create_attenuation_map,
    create_simple_phantom,
)


def create_fast_config():
    """Create SIMIND config for fast simulation."""
    config = SimulationConfig(get("AnyScan.yaml"))

    # Fast settings
    config.set_value(26, 0.1)  # number_photon_histories (10^5 photons)
    config.set_value(29, 60)  # spect_no_projections (60 views)
    config.set_value(76, 64)  # matrix_size_image_i
    config.set_value(77, 64)  # matrix_size_image_j
    config.set_value(28, 0.442)  # pixel_size_simulated_image (cm) - matches 4.42mm
    config.set_value(53, 0)  # collimator_routine = 0 (geometric only)
    config.set_value(84, 1)  # scoring_routine = 1 (SCATTWIN)

    return config


def run_standard_osem(measured_data, initial_estimate, num_subsets=6, num_iters=24):
    """Run standard OSEM without SIMIND corrections."""
    print("\n=== Standard OSEM (No SIMIND corrections) ===")

    am = AcquisitionModelUsingRayTracingMatrix()
    am.set_up(measured_data, initial_estimate)

    recon = OSMAPOSLReconstructor()
    recon.set_acquisition_model(am)
    recon.set_input(measured_data)
    recon.set_num_subsets(num_subsets)
    recon.set_num_subiterations(num_iters)

    recon.set_up(initial_estimate)
    print(
        f"  Running {num_iters} subiterations ({num_iters // num_subsets} full iterations)..."
    )
    recon.process()

    return recon.get_output()


def run_simind_osem(
    simulator, measured_data, initial_estimate, num_subsets=6, num_iters=24
):
    """Run OSEM with SimindProjector (residual corrections)."""
    print("\n=== OSEM with SimindProjector ===")

    stir_am = AcquisitionModelUsingRayTracingMatrix()

    # Create SimindProjector with residual corrections
    # Mode A: residual_correction only (fast - no penetration needed)
    simind_projector = SimindProjector(
        simind_simulator=simulator,
        stir_projector=stir_am,
        correction_update_interval=3,  # Update every 3 subiterations
        update_additive=False,
        residual_correction=True,
    )

    simind_projector.set_up(measured_data, initial_estimate)

    recon = OSMAPOSLReconstructor()
    recon.set_acquisition_model(simind_projector)
    recon.set_input(measured_data)
    recon.set_num_subsets(num_subsets)
    recon.set_num_subiterations(num_iters)

    recon.set_up(initial_estimate)
    print(f"  Running {num_iters} subiterations with SIMIND updates every 3...")
    recon.process()

    return recon.get_output()


def plot_comparison(phantom, measured_data, recon_std, recon_simind, output_dir):
    """Create comparison plots."""
    phantom_arr = get_array(phantom)
    measured_arr = get_array(measured_data)
    std_arr = get_array(recon_std)
    simind_arr = get_array(recon_simind)

    mid_z = phantom_arr.shape[0] // 2
    mid_proj = measured_arr.shape[0] // 2

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Row 1: Phantom and projection
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(phantom_arr[mid_z], cmap="hot")
    ax0.set_title("Ground Truth Phantom")
    ax0.axis("off")
    plt.colorbar(im0, ax=ax0, fraction=0.046)

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(measured_arr[mid_proj], cmap="gray", aspect="auto")
    ax1.set_title(f"SIMIND Projection (View {mid_proj})")
    ax1.set_xlabel("Detector X")
    ax1.set_ylabel("Detector Z")
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # Row 2: Reconstructions
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(std_arr[mid_z], cmap="hot")
    ax2.set_title("Standard OSEM")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(simind_arr[mid_z], cmap="hot")
    ax3.set_title("OSEM + SimindProjector")
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # Line profiles
    ax4 = fig.add_subplot(gs[0:, 2])
    mid_x = phantom_arr.shape[2] // 2
    phantom_profile = phantom_arr[mid_z, :, mid_x]
    std_profile = std_arr[mid_z, :, mid_x]
    simind_profile = simind_arr[mid_z, :, mid_x]

    ax4.plot(phantom_profile, "k-", linewidth=2.5, label="Ground Truth", alpha=0.7)
    ax4.plot(std_profile, "b--", linewidth=2, label="Standard OSEM")
    ax4.plot(simind_profile, "r--", linewidth=2, label="SIMIND OSEM")
    ax4.set_title("Central Line Profile Comparison")
    ax4.set_xlabel("Position (pixels)")
    ax4.set_ylabel("Activity")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    output_path = Path(output_dir) / "reconstruction_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to: {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("Example 07: SIMIND Simulation + OSEM with SimindProjector")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nWorking directory: {temp_dir}\n")

        # ============================================================
        # STEP 1: Create Phantom and Setup
        # ============================================================
        print("[1/5] Creating phantom and setup...")
        phantom = create_simple_phantom()
        mu_map = create_attenuation_map(phantom)

        print("  Phantom: 64×64×64 voxels, 4.42mm spacing")

        # ============================================================
        # STEP 2: Run SIMIND Simulation to Generate Measured Data
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
        simulator.set_energy_windows([126], [154], [0])  # Tc-99m

        print("  SIMIND settings:")
        print("    - Photons: 10^5 per projection")
        print("    - Projections: 60")
        print("    - Collimator: geometric only (routine=0)")
        print("    - Scoring: SCATTWIN (fast)")

        measured_data = simulator.run_simulation()

        # Add Poisson noise
        measured_arr = get_array(measured_data)
        scale = 1e6 / measured_arr.sum()
        measured_arr_noisy = np.random.poisson(measured_arr * scale) / scale
        measured_data.fill(measured_arr_noisy)

        print(f"  Total counts: {get_array(measured_data).sum():.2e}")

        # ============================================================
        # STEP 3: Create Initial Estimate
        # ============================================================
        print("\n[3/5] Creating initial estimate...")
        initial_estimate = phantom.get_uniform_copy(1.0)
        print("  Using uniform initial estimate")

        # ============================================================
        # STEP 4: Run Standard OSEM
        # ============================================================
        print("\n[4/5] Running reconstructions...")
        recon_std = run_standard_osem(
            measured_data,
            initial_estimate.clone(),
            num_subsets=6,
            num_iters=24,
        )

        # ============================================================
        # STEP 5: Run OSEM with SimindProjector
        # ============================================================
        recon_simind = run_simind_osem(
            simulator,
            measured_data,
            initial_estimate.clone(),
            num_subsets=6,
            num_iters=24,
        )

        # ============================================================
        # Visualize Results
        # ============================================================
        print("\n[5/5] Generating comparison plots...")
        plot_comparison(phantom, measured_data, recon_std, recon_simind, temp_dir)

        # ============================================================
        # Summary
        # ============================================================
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print("\nThis example demonstrated:")
        print("  1. SIMIND simulation with SCATTWIN scoring (collimator_routine=0)")
        print("  2. Standard OSEM reconstruction (STIR ray tracing)")
        print("  3. OSEM with SimindProjector (residual corrections, Mode A)")
        print("\nSimindProjector configuration:")
        print("  - Residual correction: ON")
        print("  - Additive update: OFF")
        print("  - Update interval: Every 3 subiterations")
        print("  - SIMIND runs without penetration physics (fast)")
        print("\nKey difference:")
        print("  SimindProjector corrects for resolution modeling differences")
        print("  between STIR's analytical projection and SIMIND's Monte Carlo")
        print(f"\nOutput: {temp_dir}")
        print("=" * 80)


if __name__ == "__main__":
    main()

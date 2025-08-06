#!/usr/bin/env python3
"""
Example: Schneider Density Conversion

This example demonstrates the improved Schneider2000-based density conversion
functionality, which provides much higher accuracy than the traditional
bilinear model.

The Schneider model uses 44 tissue segments with specific HU ranges and
densities, providing clinically accurate tissue density mappings for SPECT
simulation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sirf_simind_connection.converters.attenuation import (
    compare_density_methods,
    get_schneider_tissue_info,
    hu_to_density,
    hu_to_density_schneider,
)


def main():
    """Demonstrate Schneider density conversion functionality."""

    print("Schneider Density Conversion Example")
    print("=" * 50)

    # Create output directory
    output_dir = Path("output/schneider_density")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate HU test data covering the full clinical range
    hu_values = np.linspace(-1050, 2000, 1000)

    print("1. Converting HU values to densities using different methods...")

    # Compare all three methods
    hu_to_density(hu_values)
    hu_to_density_schneider(hu_values)

    print("   ✓ Bilinear model")
    print("   ✓ Schneider interpolated model")

    # Perform comparison analysis
    print("\n2. Analyzing differences between methods...")
    comparison = compare_density_methods(hu_values)

    print(
        f"   Max difference (interpolated vs bilinear): "
        f"{comparison['max_diff_interp']:.4f} g/cm³"
    )
    print(
        f"   Max difference (piecewise vs bilinear): "
        f"{comparison['max_diff_piecewise']:.4f} g/cm³"
    )
    print(
        f"   Mean absolute difference (interpolated): "
        f"{comparison['mean_diff_interp']:.4f} g/cm³"
    )
    print(
        f"   Mean absolute difference (piecewise): "
        f"{comparison['mean_diff_piecewise']:.4f} g/cm³"
    )

    # Demonstrate tissue lookup
    print("\n3. Tissue lookup examples:")
    test_hu_values = [-1000, -500, -100, 0, 50, 200, 500, 1000, 1500]

    for hu in test_hu_values:
        tissue_info = get_schneider_tissue_info(hu)
        if tissue_info:
            print(
                f"   HU {hu:4d}: {tissue_info['name']:20s} "
                f"(ρ = {tissue_info['density_g_cm3']:.3f} g/cm³)"
            )

    # Create visualization
    print("\n4. Creating comparison plots...")
    create_comparison_plots(hu_values, comparison, output_dir)

    # Create simulated CT image example
    print("\n5. Creating simulated CT image example...")
    create_ct_image_example(output_dir)

    print(f"\n✓ Example completed! Check output directory: {output_dir}")
    print("\nKey advantages of Schneider model:")
    print("  • 44 tissue segments vs 3 points (bilinear)")
    print("  • Clinically validated tissue densities")
    print("  • Better accuracy for lung, bone, and soft tissue")
    print("  • Handles metal implants and dental materials")


def create_comparison_plots(hu_values, comparison, output_dir):
    """Create comprehensive comparison plots."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Full range comparison
    axes[0, 0].plot(
        hu_values, comparison["bilinear"], "b-", label="Bilinear", linewidth=2
    )
    axes[0, 0].plot(
        hu_values,
        comparison["schneider_interpolated"],
        "r--",
        label="Schneider Interpolated",
        linewidth=2,
    )
    axes[0, 0].set_xlabel("Hounsfield Units")
    axes[0, 0].set_ylabel("Density (g/cm³)")
    axes[0, 0].set_title("Full Range Density Conversion (-1050 to 2000 HU)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Clinical range focus
    clinical_mask = (hu_values >= -200) & (hu_values <= 1500)
    hu_clinical = hu_values[clinical_mask]

    axes[0, 1].plot(
        hu_clinical,
        comparison["bilinear"][clinical_mask],
        "b-",
        label="Bilinear",
        linewidth=2,
    )
    axes[0, 1].plot(
        hu_clinical,
        comparison["schneider_interpolated"][clinical_mask],
        "r--",
        label="Schneider Interpolated",
        linewidth=2,
    )
    axes[0, 1].set_xlabel("Hounsfield Units")
    axes[0, 1].set_ylabel("Density (g/cm³)")
    axes[0, 1].set_title("Clinical Range (-200 to 1500 HU)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Difference plots
    axes[1, 0].plot(
        hu_values,
        comparison["difference_interpolated"],
        "r-",
        label="Schneider Interp - Bilinear",
        linewidth=2,
    )
    axes[1, 0].plot(
        hu_values,
        comparison["difference_piecewise"],
        "g-",
        label="Schneider Piece - Bilinear",
        linewidth=2,
    )
    axes[1, 0].axhline(y=0, color="k", linestyle="-", alpha=0.3)
    axes[1, 0].set_xlabel("Hounsfield Units")
    axes[1, 0].set_ylabel("Density Difference (g/cm³)")
    axes[1, 0].set_title("Density Differences vs Bilinear Model")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Clinical range differences
    axes[1, 1].plot(
        hu_clinical,
        comparison["difference_interpolated"][clinical_mask],
        "r-",
        label="Schneider Interp - Bilinear",
        linewidth=2,
    )
    axes[1, 1].plot(
        hu_clinical,
        comparison["difference_piecewise"][clinical_mask],
        "g-",
        label="Schneider Piece - Bilinear",
        linewidth=2,
    )
    axes[1, 1].axhline(y=0, color="k", linestyle="-", alpha=0.3)
    axes[1, 1].set_xlabel("Hounsfield Units")
    axes[1, 1].set_ylabel("Density Difference (g/cm³)")
    axes[1, 1].set_title("Clinical Range Differences")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "density_method_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def create_ct_image_example(output_dir):
    """Create a simulated CT image to demonstrate density conversion."""

    # Create a simple phantom with different tissue types
    size = 128
    phantom_hu = np.zeros((size, size))

    # Background air
    phantom_hu.fill(-1000)

    # Body outline (soft tissue)
    center = size // 2
    radius_body = size // 3
    y, x = np.ogrid[:size, :size]
    body_mask = (x - center) ** 2 + (y - center) ** 2 <= radius_body**2
    phantom_hu[body_mask] = 40  # Soft tissue

    # Lung regions
    lung_radius = size // 8
    lung1_center = (center - size // 6, center - size // 8)
    lung2_center = (center - size // 6, center + size // 8)

    lung1_mask = (x - lung1_center[1]) ** 2 + (
        y - lung1_center[0]
    ) ** 2 <= lung_radius**2
    lung2_mask = (x - lung2_center[1]) ** 2 + (
        y - lung2_center[0]
    ) ** 2 <= lung_radius**2

    phantom_hu[lung1_mask] = -800  # Lung tissue
    phantom_hu[lung2_mask] = -800

    # Bone structures
    bone_radius = size // 12
    spine_center = (center + size // 4, center)
    spine_mask = (x - spine_center[1]) ** 2 + (
        y - spine_center[0]
    ) ** 2 <= bone_radius**2
    phantom_hu[spine_mask] = 800  # Bone

    # Metal implant
    metal_radius = size // 20
    implant_center = (center, center + size // 6)
    implant_mask = (x - implant_center[1]) ** 2 + (
        y - implant_center[0]
    ) ** 2 <= metal_radius**2
    phantom_hu[implant_mask] = 2000  # Metal

    # Convert to densities using both methods
    density_bilinear = hu_to_density(phantom_hu)
    density_schneider = hu_to_density_schneider(phantom_hu)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original HU image
    im1 = axes[0, 0].imshow(phantom_hu, cmap="gray", vmin=-1000, vmax=2000)
    axes[0, 0].set_title("Simulated CT Image (HU)")
    axes[0, 0].axis("off")
    plt.colorbar(im1, ax=axes[0, 0], label="HU")

    # Bilinear density
    im2 = axes[0, 1].imshow(density_bilinear, cmap="viridis", vmin=0, vmax=2.5)
    axes[0, 1].set_title("Bilinear Density Conversion")
    axes[0, 1].axis("off")
    plt.colorbar(im2, ax=axes[0, 1], label="Density (g/cm³)")

    # Schneider density
    im3 = axes[1, 0].imshow(density_schneider, cmap="viridis", vmin=0, vmax=2.5)
    axes[1, 0].set_title("Schneider Density Conversion")
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0], label="Density (g/cm³)")

    # Difference
    difference = density_schneider - density_bilinear
    im4 = axes[1, 1].imshow(
        difference,
        cmap="RdBu_r",
        vmin=-np.max(np.abs(difference)),
        vmax=np.max(np.abs(difference)),
    )
    axes[1, 1].set_title("Density Difference\n(Schneider - Bilinear)")
    axes[1, 1].axis("off")
    plt.colorbar(im4, ax=axes[1, 1], label="Density Diff (g/cm³)")

    plt.tight_layout()
    plt.savefig(
        output_dir / "ct_phantom_density_conversion.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Save numerical data
    np.savez(
        output_dir / "phantom_data.npz",
        hu_image=phantom_hu,
        density_bilinear=density_bilinear,
        density_schneider=density_schneider,
        density_difference=difference,
    )


if __name__ == "__main__":
    main()

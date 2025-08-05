"""
Tests for Schneider density conversion functionality.
"""

import numpy as np
import pytest

from sirf_simind_connection.converters.attenuation import (
    compare_density_methods,
    get_schneider_tissue_info,
    hu_to_density,
    hu_to_density_schneider,
    hu_to_density_schneider_piecewise,
    load_schneider_data,
)


class TestSchneiderData:
    """Test Schneider data loading and structure."""

    def test_load_schneider_data(self):
        """Test that Schneider data can be loaded successfully."""
        data = load_schneider_data()

        assert isinstance(data, dict)
        assert len(data) > 0

        # Check that we have expected number of tissues (44 in Schneider2000)
        assert len(data) == 44

        # Check structure of first entry
        first_tissue = next(iter(data.values()))
        required_keys = ["density (mg/cm3)", "HU_lo (HU)", "HU_hi (HU)"]
        for key in required_keys:
            assert key in first_tissue

    def test_schneider_data_ranges(self):
        """Test that Schneider data has reasonable HU ranges and densities."""
        data = load_schneider_data()

        # Check overall HU range
        all_hu_lo = [tissue["HU_lo (HU)"] for tissue in data.values()]
        all_hu_hi = [tissue["HU_hi (HU)"] for tissue in data.values()]

        assert min(all_hu_lo) <= -1000  # Should include air range
        assert max(all_hu_hi) >= 1000  # Should include bone/metal range

        # Check density ranges are reasonable
        all_densities = [tissue["density (mg/cm3)"] for tissue in data.values()]
        assert min(all_densities) > 0  # All densities should be positive
        assert (
            max(all_densities) < 5000
        )  # Reasonable upper bound for biological tissues


class TestSchneiderDensityConversion:
    """Test Schneider density conversion functions."""

    def test_hu_to_density_schneider_basic(self):
        """Test basic functionality of Schneider interpolated conversion."""
        # Test with simple array
        hu_values = np.array([-1000, -500, 0, 500, 1000])
        densities = hu_to_density_schneider(hu_values)

        assert densities.shape == hu_values.shape
        assert np.all(densities >= 0)
        assert np.all(densities <= 3.0)  # Reasonable upper bound

    def test_hu_to_density_schneider_piecewise_basic(self):
        """Test basic functionality of Schneider piecewise conversion."""
        hu_values = np.array([-1000, -500, 0, 500, 1000])
        densities = hu_to_density_schneider_piecewise(hu_values)

        assert densities.shape == hu_values.shape
        assert np.all(densities >= 0)
        assert np.all(densities <= 3.0)

    def test_schneider_vs_bilinear_differences(self):
        """Test that Schneider methods produce different results from bilinear."""
        hu_values = np.linspace(-1000, 1000, 100)

        density_bilinear = hu_to_density(hu_values)
        density_schneider = hu_to_density_schneider(hu_values)

        # Should be different for most values
        differences = np.abs(density_schneider - density_bilinear)
        assert np.mean(differences) > 0.001  # Meaningful differences

    def test_schneider_extreme_values(self):
        """Test Schneider conversion with extreme HU values."""
        # Test values outside normal range
        extreme_hu = np.array([-2000, -1500, 3000, 5000])

        # Should not crash and should return reasonable values
        densities_interp = hu_to_density_schneider(extreme_hu)
        densities_piece = hu_to_density_schneider_piecewise(extreme_hu)

        assert np.all(densities_interp >= 0)
        assert np.all(densities_piece >= 0)
        assert np.all(densities_interp <= 3.0)
        assert np.all(densities_piece <= 3.0)

    def test_schneider_array_shapes(self):
        """Test that Schneider functions handle different array shapes."""
        # 1D array
        hu_1d = np.array([0, 100, 500])
        density_1d = hu_to_density_schneider(hu_1d)
        assert density_1d.shape == hu_1d.shape

        # 2D array
        hu_2d = np.array([[0, 100], [500, 1000]])
        density_2d = hu_to_density_schneider(hu_2d)
        assert density_2d.shape == hu_2d.shape

        # 3D array
        hu_3d = np.array([[[0, 100]], [[500, 1000]]])
        density_3d = hu_to_density_schneider(hu_3d)
        assert density_3d.shape == hu_3d.shape


class TestTissueLookup:
    """Test tissue information lookup functionality."""

    def test_get_schneider_tissue_info_valid(self):
        """Test tissue lookup for valid HU values."""
        # Test air
        air_info = get_schneider_tissue_info(-1000)
        assert air_info is not None
        assert "Air" in air_info["name"]
        assert air_info["density_g_cm3"] < 0.01  # Very low density for air

        # Test water/soft tissue
        soft_info = get_schneider_tissue_info(50)
        assert soft_info is not None
        assert soft_info["density_g_cm3"] > 0.9  # Close to water density
        assert soft_info["density_g_cm3"] < 1.2

        # Test bone
        bone_info = get_schneider_tissue_info(800)
        assert bone_info is not None
        assert bone_info["density_g_cm3"] > 1.3  # Higher density for bone

    def test_get_schneider_tissue_info_boundary(self):
        """Test tissue lookup at boundary values."""
        data = load_schneider_data()

        # Find a tissue with clear boundaries
        for tissue_name, tissue_data in data.items():
            hu_lo = tissue_data["HU_lo (HU)"]
            tissue_data["HU_hi (HU)"]

            # Test at lower boundary (should be included)
            info_lo = get_schneider_tissue_info(hu_lo)
            assert info_lo is not None
            assert info_lo["name"] == tissue_name

            # Test just below lower boundary (should be different tissue or None)
            if hu_lo > -1050:  # Not at absolute minimum
                info_below = get_schneider_tissue_info(hu_lo - 0.1)
                if info_below is not None:
                    assert info_below["name"] != tissue_name

            break  # Just test one tissue for boundaries

    def test_get_schneider_tissue_info_invalid(self):
        """Test tissue lookup for invalid/extreme HU values."""
        # Very extreme values might return None or edge tissue
        extreme_low = get_schneider_tissue_info(-5000)
        extreme_high = get_schneider_tissue_info(10000)

        # Should either return None or edge tissue data
        if extreme_low is not None:
            assert extreme_low["density_g_cm3"] >= 0
        if extreme_high is not None:
            assert extreme_high["density_g_cm3"] >= 0


class TestMethodComparison:
    """Test comparison functionality between methods."""

    def test_compare_density_methods_structure(self):
        """Test that comparison function returns proper structure."""
        hu_values = np.linspace(-500, 1000, 50)
        comparison = compare_density_methods(hu_values)

        expected_keys = [
            "bilinear",
            "schneider_interpolated",
            "schneider_piecewise",
            "difference_interpolated",
            "difference_piecewise",
            "max_diff_interp",
            "max_diff_piecewise",
            "mean_diff_interp",
            "mean_diff_piecewise",
        ]

        for key in expected_keys:
            assert key in comparison

        # Check array shapes
        for key in ["bilinear", "schneider_interpolated", "schneider_piecewise"]:
            assert comparison[key].shape == hu_values.shape

    def test_compare_density_methods_statistics(self):
        """Test that comparison statistics are reasonable."""
        hu_values = np.linspace(-1000, 1000, 100)
        comparison = compare_density_methods(hu_values)

        # Statistics should be non-negative
        assert comparison["max_diff_interp"] >= 0
        assert comparison["max_diff_piecewise"] >= 0
        assert comparison["mean_diff_interp"] >= 0
        assert comparison["mean_diff_piecewise"] >= 0

        # Max should be >= mean
        assert comparison["max_diff_interp"] >= comparison["mean_diff_interp"]
        assert comparison["max_diff_piecewise"] >= comparison["mean_diff_piecewise"]


class TestConsistency:
    """Test consistency and edge cases."""

    def test_density_monotonicity(self):
        """Test that density generally increases with HU (not strictly required but expected trend)."""
        hu_values = np.array([-1000, -500, 0, 500, 1000])
        densities = hu_to_density_schneider(hu_values)

        # Overall trend should be increasing (allowing for some local variations)
        assert densities[-1] > densities[0]  # Bone > Air
        assert densities[2] > densities[0]  # Water > Air

    def test_known_values(self):
        """Test conversion for known reference values."""
        # Water should be close to 1.0 g/cm³
        water_density = hu_to_density_schneider(np.array([0]))[0]
        assert abs(water_density - 1.0) < 0.1

        # Air should be very low density
        air_density = hu_to_density_schneider(np.array([-1000]))[0]
        assert air_density < 0.01

    def test_piecewise_vs_interpolated_consistency(self):
        """Test that piecewise and interpolated methods are reasonably consistent."""
        hu_values = np.linspace(-1000, 1000, 100)

        density_interp = hu_to_density_schneider(hu_values)
        density_piece = hu_to_density_schneider_piecewise(hu_values)

        # Should be close but not identical
        max_diff = np.max(np.abs(density_interp - density_piece))
        mean_diff = np.mean(np.abs(density_interp - density_piece))

        assert max_diff < 0.5  # Reasonable maximum difference
        assert mean_diff < 0.1  # Small average difference


@pytest.mark.integration
def test_schneider_full_workflow():
    """Integration test for complete Schneider workflow."""
    # Create a simple phantom with different tissue types
    phantom_hu = np.array(
        [
            [-1000, -800, -500],  # Air, lung variations
            [0, 50, 100],  # Water, soft tissues
            [500, 800, 1200],  # Bone variations
        ]
    )

    # Convert using all methods
    density_bilinear = hu_to_density(phantom_hu)
    density_schneider = hu_to_density_schneider(phantom_hu)
    density_piecewise = hu_to_density_schneider_piecewise(phantom_hu)

    # All should have same shape
    assert density_bilinear.shape == phantom_hu.shape
    assert density_schneider.shape == phantom_hu.shape
    assert density_piecewise.shape == phantom_hu.shape

    # All should have reasonable values
    for density_map in [density_bilinear, density_schneider, density_piecewise]:
        assert np.all(density_map >= 0)
        assert np.all(density_map <= 3.0)

    # Schneider methods should show differences from bilinear
    diff_schneider = np.abs(density_schneider - density_bilinear)
    diff_piecewise = np.abs(density_piecewise - density_bilinear)

    assert np.mean(diff_schneider) > 0.001  # Should see meaningful differences
    assert np.mean(diff_piecewise) > 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

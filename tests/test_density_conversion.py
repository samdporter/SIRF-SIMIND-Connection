# test_density_conversion.py - Unit tests for density conversion utilities
"""
test_density_conversion.py - Unit tests for density conversion utilities
"""

import pytest
import numpy as np

try:
    from sirf_simind_connection.utils import density_conversion
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not UTILS_AVAILABLE, reason="Utils module not available")
class TestDensityConversion:
    """Unit tests for density conversion utilities."""
    
    def test_hounsfield_to_density_water(self):
        """Test HU to density conversion for water."""
        hu_water = 0
        density = density_conversion.hounsfield_to_density(hu_water)
        assert density == pytest.approx(1.0, rel=1e-3)
    
    def test_hounsfield_to_density_air(self):
        """Test HU to density conversion for air."""
        hu_air = -1000
        density = density_conversion.hounsfield_to_density(hu_air)
        assert density == pytest.approx(0.0, abs=1e-3)
    
    def test_hounsfield_to_density_bone(self):
        """Test HU to density conversion for bone."""
        hu_bone = 1000
        density = density_conversion.hounsfield_to_density(hu_bone)
        assert density > 1.0  # Bone is denser than water
    
    def test_hounsfield_to_density_array(self):
        """Test HU to density conversion for arrays."""
        hu_values = np.array([-1000, 0, 1000])
        densities = density_conversion.hounsfield_to_density(hu_values)
        
        assert len(densities) == 3
        assert densities[0] < densities[1] < densities[2]  # Increasing density
    
    def test_density_to_attenuation_tc99m(self):
        """Test density to attenuation conversion for Tc-99m energy."""
        densities = np.array([0.0, 1.0, 1.8])  # air, water, bone
        energy_kev = 140
        
        mu_values = density_conversion.density_to_attenuation(densities, energy_kev)
        
        assert len(mu_values) == 3
        assert all(mu_values >= 0)  # Non-negative attenuation
        assert mu_values[0] < mu_values[1] < mu_values[2]  # Increasing attenuation
    
    def test_density_to_attenuation_different_energies(self):
        """Test density to attenuation conversion at different energies."""
        density_water = 1.0
        energies = np.array([80, 140, 511])  # Low, medium, high energy
        
        mu_values = [density_conversion.density_to_attenuation(density_water, e) for e in energies]
        
        # Generally, attenuation decreases with energy (except at edges)
        assert all(mu > 0 for mu in mu_values)
    
    def test_attenuation_to_density_round_trip(self):
        """Test round-trip conversion: density -> attenuation -> density."""
        original_density = 1.5
        energy_kev = 140
        
        mu = density_conversion.density_to_attenuation(original_density, energy_kev)
        recovered_density = density_conversion.attenuation_to_density(mu, energy_kev)
        
        assert recovered_density == pytest.approx(original_density, rel=1e-2)
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Negative energies
        with pytest.raises(ValueError):
            density_conversion.density_to_attenuation(1.0, -140)
        
        # Negative densities
        with pytest.raises(ValueError):
            density_conversion.hounsfield_to_density(-2000)  # Below air
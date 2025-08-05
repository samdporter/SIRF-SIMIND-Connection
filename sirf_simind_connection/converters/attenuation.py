"""
Attenuation coefficient conversion utilities.

This module provides functions to convert Hounsfield Units (HU) to attenuation
coefficients and densities based on the bilinear model.
"""

import importlib.resources as pkg_resources
import warnings
from pathlib import Path

import numpy as np

from sirf_simind_connection.data import data_path


def get_package_data_path(filename):
    """Get the path to a data file in the package."""
    try:
        # Python 3.9+
        files = pkg_resources.files("sirf_simind_connection.data")
        return files / filename
    except AttributeError:
        # Python 3.8
        with pkg_resources.path("sirf_simind_connection.data", filename) as path:
            return path


def interpolate_attenuation_coefficient(filename, energy):
    """Interpolate attenuation coefficient from tabulated data."""
    # Skip the header lines
    energies, coeffs = np.loadtxt(filename, unpack=True, skiprows=12)
    return np.interp(energy, energies, coeffs)


def get_attenuation_coefficient(material, energy, file_path=None):
    """
    Get attenuation coefficient for a given material and energy.

    Args:
        material (str): 'water' or 'bone'
        energy (float): Photon energy in keV
        file_path (str, optional): Override default data file path

    Returns:
        float: Linear attenuation coefficient in cm^-1
    """
    density_water = 1.0  # g/cm^3
    density_bone = 1.85  # g/cm^3 for cortical bone

    if material == "water":
        filename = data_path("h2o.atn")
    elif material == "bone":
        filename = data_path("bone.atn")
    else:
        raise ValueError("Unknown material. Accepted values are 'water' or 'bone'.")

    if file_path:
        filepath = Path(file_path) / filename
    else:
        filepath = get_package_data_path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"Attenuation data file not found: {filepath}")

    mass_attn_coeffs = interpolate_attenuation_coefficient(filepath, energy)

    if material == "water":
        return mass_attn_coeffs * density_water
    elif material == "bone":
        return mass_attn_coeffs * density_bone


def hu_to_attenuation(image_array, photon_energy, file_path=None):
    """
    Convert Hounsfield Units to attenuation coefficients.

    Args:
        image_array (np.ndarray): Array of HU values
        photon_energy (float): Photon energy in keV
        file_path (str, optional): Override default data file path

    Returns:
        np.ndarray: Attenuation map in cm^-1
    """
    # Constants
    HU_water = 0
    HU_bone = 1000

    # Convert photon_energy to MeV from keV
    photon_energy_mev = photon_energy / 1000

    # Get attenuation coefficients
    mu_water = get_attenuation_coefficient("water", photon_energy_mev, file_path)
    mu_bone = get_attenuation_coefficient("bone", photon_energy_mev, file_path)

    # For air, we assume negligible attenuation
    mu_air = 0.0

    # Bilinear model slopes
    slope_soft = (mu_water - mu_air) / (
        HU_water - (-1000)
    )  # from -1000 HU (air) to 0 HU (water)
    slope_bone = (mu_bone - mu_water) / (
        HU_bone - HU_water
    )  # from 0 HU (water) to 1000 HU (bone)

    # Compute attenuation map using the bilinear model
    attenuation_map = np.where(
        image_array <= HU_water,
        mu_air + slope_soft * (image_array + 1000),
        mu_water + slope_bone * (image_array - HU_water),
    )

    # Ensure non-negative values
    attenuation_map = np.maximum(attenuation_map, 0)

    return attenuation_map


def hu_to_density(image_array):
    """
    Convert Hounsfield Units to density values.

    Args:
        image_array (np.ndarray): Array of HU values

    Returns:
        np.ndarray: Density map in g/cm^3
    """
    # Constants
    density_air = 0.001225  # g/cm^3
    density_water = 1.0  # g/cm^3
    density_bone = 1.85  # g/cm^3

    HU_air = -1000
    HU_water = 0
    HU_bone = 1000

    slope_soft = (density_water - density_air) / (HU_water - HU_air)
    slope_bone = (density_bone - density_water) / (HU_bone - HU_water)

    density_map = np.where(
        image_array <= HU_water,
        density_air + slope_soft * (image_array - HU_air),
        density_water + slope_bone * (image_array - HU_water),
    )

    # Ensure reasonable density bounds
    density_map = np.clip(density_map, 0, 3.0)  # Max density ~3 g/cm^3

    return density_map


def attenuation_to_density(attenuation_array, photon_energy, file_path=None):
    """
    Convert attenuation coefficients to density values.

    This is an approximate inverse of the attenuation calculation.

    Args:
        attenuation_array (np.ndarray): Array of attenuation coefficients in cm^-1
        photon_energy (float): Photon energy in keV
        file_path (str, optional): Override default data file path

    Returns:
        np.ndarray: Density map in g/cm^3
    """
    # Convert photon_energy to MeV
    photon_energy_mev = photon_energy / 1000

    # Get attenuation coefficients
    mu_water = get_attenuation_coefficient("water", photon_energy_mev, file_path)
    mu_bone = get_attenuation_coefficient("bone", photon_energy_mev, file_path)

    # Densities
    # density_air = 0.001225  # g/cm^3  # Not used in this calculation
    density_water = 1.0  # g/cm^3
    density_bone = 1.85  # g/cm^3

    # Bilinear model inverse
    if mu_water > 0:
        slope_soft = density_water / mu_water
    else:
        warnings.warn("Water attenuation coefficient is zero or negative")
        slope_soft = 1.0

    if mu_bone > mu_water:
        slope_bone = (density_bone - density_water) / (mu_bone - mu_water)
    else:
        warnings.warn("Bone attenuation not greater than water")
        slope_bone = 1.0

    density_map = np.where(
        attenuation_array <= mu_water,
        slope_soft * attenuation_array,
        density_water + slope_bone * (attenuation_array - mu_water),
    )

    # Ensure reasonable bounds
    density_map = np.clip(density_map, 0, 3.0)

    return density_map

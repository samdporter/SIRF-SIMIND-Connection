import tempfile
from pathlib import Path

import numpy as np
import pytest

from sirf_simind_connection.utils.stir_utils import (
    create_attenuation_map,
    create_simple_phantom,
    create_stir_acqdata,
    create_stir_image,
    parse_interfile,
)


# Most tests require SIRF, but parse_interfile doesn't
pytestmark = pytest.mark.requires_sirf


def test_create_simple_phantom():
    """Test creating a simple phantom."""
    phantom = create_simple_phantom()

    # Check that phantom was created
    assert phantom is not None

    # Check dimensions
    dims = phantom.dimensions()
    assert len(dims) == 3
    assert all(d > 0 for d in dims)

    # Check that phantom has some activity
    phantom_array = phantom.as_array()
    assert phantom_array.max() > 0


def test_create_attenuation_map():
    """Test creating attenuation map from phantom."""
    phantom = create_simple_phantom()
    mu_map = create_attenuation_map(phantom)

    # Check that attenuation map was created
    assert mu_map is not None

    # Check dimensions match phantom
    phantom_dims = phantom.dimensions()
    mu_dims = mu_map.dimensions()
    assert phantom_dims == mu_dims

    # Check that attenuation values are reasonable
    mu_array = mu_map.as_array()
    assert mu_array.max() > 0
    assert mu_array.max() < 1.0  # Reasonable for tissue at 140keV


def test_create_stir_image():
    """Test creating STIR image with specific dimensions."""
    matrix_dim = [32, 32, 32]
    voxel_size = [2.0, 2.0, 2.0]

    image = create_stir_image(matrix_dim, voxel_size)

    # Check dimensions
    dims = image.dimensions()
    assert dims == tuple(matrix_dim)

    # Check voxel sizes
    voxels = image.voxel_sizes()
    assert np.allclose(voxels, voxel_size, rtol=1e-6)


def test_create_stir_acqdata():
    """Test creating STIR acquisition data."""
    proj_matrix = [64, 64]
    num_projections = 60
    pixel_size = [4.42, 4.42]

    acqdata = create_stir_acqdata(proj_matrix, num_projections, pixel_size)

    # Check that acquisition data was created
    assert acqdata is not None

    # Check that it has expected structure
    acq_array = acqdata.as_array()
    assert acq_array.shape[1] == proj_matrix[0]  # First dimension
    assert acq_array.shape[2] == num_projections  # Number of projections
    assert acq_array.shape[3] == proj_matrix[1]  # Second dimension


@pytest.mark.unit
def test_parse_interfile():
    """Test parsing interfile headers."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".hv", delete=False) as f:
        # Write a simple interfile header
        f.write("!INTERFILE :=\n")
        f.write("name of data file := test.v\n")
        f.write("!matrix size [1] := 64\n")
        f.write("!matrix size [2] := 64\n")
        f.write("scaling factor (mm/pixel) [1] := 4.42\n")
        f.write("!END OF INTERFILE :=\n")
        temp_file = f.name

    try:
        # Parse the file
        values = parse_interfile(temp_file)

        # Check parsed values
        assert "name of data file" in values
        assert values["name of data file"] == "test.v"
        assert "!matrix size [1]" in values
        assert values["!matrix size [1]"] == "64"
        assert "scaling factor (mm/pixel) [1]" in values
        assert values["scaling factor (mm/pixel) [1]"] == "4.42"

    finally:
        # Clean up
        Path(temp_file).unlink()


def test_phantom_properties():
    """Test that created phantom has expected properties."""
    phantom = create_simple_phantom()
    phantom_array = phantom.as_array()

    # Should have background activity (~10) and hot spot activity (~40)
    unique_values = np.unique(phantom_array)

    # Should have at least background (0), body (~10), and hot spot (~40)
    assert len(unique_values) >= 3

    # Check that there are both background and hot regions
    assert phantom_array.min() == 0  # Background
    assert phantom_array.max() > 30  # Hot spot

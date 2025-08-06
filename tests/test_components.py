import pytest

from sirf_simind_connection.core.components import (
    EnergyWindow,
    ImageGeometry,
    ImageValidator,
    PenetrateOutputType,
    RotationDirection,
    RotationParameters,
    ScoringRoutine,
    ValidationError,
)


@pytest.mark.unit
def test_energy_window():
    """Test EnergyWindow data class."""
    window = EnergyWindow(126.0, 154.0, 0, 1)
    assert window.lower_bound == 126.0
    assert window.upper_bound == 154.0
    assert window.scatter_order == 0
    assert window.window_id == 1


@pytest.mark.requires_sirf
def test_image_geometry():
    """Test ImageGeometry data class."""
    from sirf_simind_connection.utils.stir_utils import create_simple_phantom

    phantom = create_simple_phantom()
    geometry = ImageGeometry.from_image(phantom)

    assert geometry.dim_x > 0
    assert geometry.dim_y > 0
    assert geometry.dim_z > 0
    assert geometry.voxel_x > 0
    assert geometry.voxel_y > 0
    assert geometry.voxel_z > 0


@pytest.mark.unit
def test_rotation_parameters():
    """Test RotationParameters data class."""
    rotation = RotationParameters(
        direction=RotationDirection.CCW,
        rotation_angle=360.0,
        start_angle=0.0,
        num_projections=64,
    )

    assert rotation.direction == RotationDirection.CCW
    assert rotation.rotation_angle == 360.0

    # Test SIMIND parameter conversion
    switch, start_angle = rotation.to_simind_params()
    assert isinstance(switch, int)
    assert isinstance(start_angle, float)


@pytest.mark.unit
def test_scoring_routine_enum():
    """Test ScoringRoutine enum."""
    assert ScoringRoutine.SCATTWIN.value == 1
    assert ScoringRoutine.PENETRATE.value == 4


@pytest.mark.unit
def test_penetrate_output_type_enum():
    """Test PenetrateOutputType enum."""
    assert PenetrateOutputType.ALL_INTERACTIONS.value == 1
    assert PenetrateOutputType.GEOM_COLL_PRIMARY_ATT.value == 2


@pytest.mark.requires_sirf
def test_image_validator():
    """Test ImageValidator functionality."""
    from sirf_simind_connection.utils.stir_utils import create_simple_phantom

    phantom = create_simple_phantom()

    # Test validation passes for square phantom
    try:
        ImageValidator.validate_square_pixels(phantom)
    except ValidationError:
        pytest.fail("Validation should pass for square phantom")

    # Test compatibility check
    phantom2 = create_simple_phantom()
    try:
        ImageValidator.validate_compatibility(phantom, phantom2)
    except ValidationError:
        pytest.fail("Compatibility check should pass for identical phantoms")


@pytest.mark.unit
def test_validation_error():
    """Test ValidationError exception."""
    with pytest.raises(ValidationError):
        raise ValidationError("Test validation error")

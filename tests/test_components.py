import tempfile
from pathlib import Path

import pytest

from sirf_simind_connection.core.components import (
    EnergyWindow,
    ImageGeometry,
    ImageValidator,
    RotationDirection,
    RotationParameters,
)
from sirf_simind_connection.core.file_managers import OrbitFileManager
from sirf_simind_connection.core.types import (
    SIMIND_VOXEL_UNIT_CONVERSION,
    PenetrateOutputType,
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
    # Validate metadata wiring
    slugs = {member.slug for member in PenetrateOutputType}
    assert len(slugs) == len(PenetrateOutputType)
    assert PenetrateOutputType.ALL_INTERACTIONS.slug == "all_interactions"
    assert (
        "backscatter"
        in PenetrateOutputType.GEOM_COLL_PRIMARY_ATT_BACK.description.lower()
    )


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


@pytest.mark.unit
def test_orbit_file_manager_write_with_unit_conversion():
    """
    Test OrbitFileManager writes orbit file with correct mm->cm conversion.

    Regression test for bug where radii were written in mm instead of cm,
    causing SIMIND to interpret them incorrectly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = OrbitFileManager(Path(tmpdir))

        # Test data: radii in mm (STIR units)
        radii_mm = [134.0, 231.0, 311.0]
        expected_radii_cm = [13.4, 23.1, 31.1]

        # Write orbit file
        orbit_file = manager.write_orbit_file(
            radii_mm, "test_output", center_of_rotation=64
        )

        # Verify file name uses _input suffix (avoid collision with SIMIND output)
        assert orbit_file.name == "test_output_input.cor"

        # Read back and verify conversion
        with open(orbit_file) as f:
            lines = f.readlines()

        assert len(lines) == len(radii_mm)

        for i, line in enumerate(lines):
            parts = line.strip().split()
            radius_cm = float(parts[0])
            cor = int(parts[1])

            # Verify conversion from mm to cm
            assert abs(radius_cm - expected_radii_cm[i]) < 0.01
            assert cor == 64

            # Verify round-trip conversion
            radius_mm_back = radius_cm * SIMIND_VOXEL_UNIT_CONVERSION
            assert abs(radius_mm_back - radii_mm[i]) < 0.01


@pytest.mark.unit
def test_orbit_file_manager_naming_no_collision():
    """
    Test that input orbit file uses different name than SIMIND output.

    Regression test for bug where input orbit file was named {prefix}.cor,
    which was then overwritten by SIMIND's output {prefix}.cor file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = OrbitFileManager(Path(tmpdir))

        # Write input orbit file
        radii_mm = [150.0, 200.0]
        orbit_file = manager.write_orbit_file(radii_mm, "simulation")

        # Verify input uses _input suffix
        assert orbit_file.name == "simulation_input.cor"
        assert orbit_file != Path(tmpdir) / "simulation.cor"

        # Simulate SIMIND writing output file
        simind_output = Path(tmpdir) / "simulation.cor"
        simind_output.write_text("15.0\t64\n20.0\t64\n")

        # Verify input file still exists and is different
        assert orbit_file.exists()
        assert simind_output.exists()
        assert orbit_file.read_text() != simind_output.read_text()


@pytest.mark.unit
def test_orbit_file_manager_read():
    """Test OrbitFileManager reads orbit file and converts cm->mm."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test orbit file in SIMIND format (cm)
        orbit_file = Path(tmpdir) / "test.cor"
        orbit_file.write_text("13.4\t64\n23.1\t64\n31.1\t64\n")

        manager = OrbitFileManager(Path(tmpdir))
        radii_mm = manager.read_orbit_file(orbit_file)

        # Verify conversion from cm to mm
        expected_mm = [134.0, 231.0, 311.0]
        assert len(radii_mm) == len(expected_mm)

        for i, expected in enumerate(expected_mm):
            assert abs(radii_mm[i] - expected) < 0.1


@pytest.mark.unit
def test_simind_executor_orbit_file_position():
    """
    Test that orbit file is placed correctly in SIMIND command line.

    Regression test for bug where orbit file was appended after -p flag,
    causing SIMIND to ignore it. Orbit file must be 3rd/4th/5th argument.
    """
    from unittest.mock import MagicMock, patch

    from sirf_simind_connection.core.components import SimindExecutor

    executor = SimindExecutor()

    with tempfile.TemporaryDirectory() as tmpdir:
        orbit_file = Path(tmpdir) / "test_input.cor"
        orbit_file.write_text("15.0\t64\n")

        # Test MPI command with orbit file
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock()
            executor.run_simulation(
                "output",
                orbit_file=orbit_file,
                runtime_switches={"MP": 6, "CC": "ge-megp"},
            )

            # Verify command was called
            assert mock_run.called
            cmd = mock_run.call_args[0][0]

            # Find positions in command
            simind_idx = cmd.index("simind")
            orbit_idx = cmd.index(orbit_file.name)  # Just filename, not full path
            p_flag_idx = cmd.index("-p")

            # Orbit file must come BEFORE -p flag (3rd/4th/5th argument)
            assert orbit_idx < p_flag_idx
            assert orbit_idx >= simind_idx + 2  # After prefix arguments

            # Verify it's just the filename (no path)
            assert cmd[orbit_idx] == orbit_file.name
            assert "/" not in cmd[orbit_idx]

        # Test serial command with orbit file
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock()
            executor.run_simulation(
                "output", orbit_file=orbit_file, runtime_switches={"CC": "ge-megp"}
            )

            assert mock_run.called
            cmd = mock_run.call_args[0][0]

            # Verify orbit file is 3rd argument (index 2)
            # Should be just filename, not full path (since we chdir to output_dir)
            assert cmd[0] == "simind"
            assert cmd[1] == "output"
            assert cmd[2] == "output"
            assert cmd[3] == orbit_file.name
            assert "/" not in cmd[3]  # Verify no path separators

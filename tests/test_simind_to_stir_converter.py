"""Tests for SIMIND to STIR converter."""

import pytest

from sirf_simind_connection.converters.simind_to_stir import (
    ConversionConfig,
    DataFileNameRule,
    EnergyWindowRule,
    ImageDurationRule,
    NumberFormatRule,
    OrbitConversionRule,
    OrbitFileRule,
    RadiusConversionRule,
    SimindToStirConverter,
    StartAngleConversionRule,
)
from sirf_simind_connection.utils.interfile_parser import parse_interfile_line
from sirf_simind_connection.core.types import PenetrateOutputType


@pytest.mark.unit
class TestConversionRules:
    """Test individual conversion rules."""

    def test_radius_conversion_rule(self):
        """Test radius conversion with scaling."""
        rule = RadiusConversionRule(scale_factor=10.0)

        # Test matching
        assert rule.matches("Radius := 134")
        assert not rule.matches("Radii := {134}")
        assert not rule.matches("start angle := 0")

        # Test conversion
        line, context = rule.convert("Radius := 134.5", {})
        assert "Radius := 134.5" in line

    def test_start_angle_conversion_rule(self):
        """Test start angle conversion with offset."""
        rule = StartAngleConversionRule(angle_offset=180.0)

        # Test matching
        assert rule.matches("start angle := 0.000000")
        assert not rule.matches("Radius := 134")

        # Test conversion - 0 + 180 = 180
        line, context = rule.convert("start angle := 0.000000", {})
        assert "start angle := 180" in line

        # Test conversion - 90 + 180 = 270
        line, context = rule.convert("start angle := 90.000000", {})
        assert "start angle := 270" in line

        # Test wrap-around - 270 + 180 = 450 % 360 = 90
        line, context = rule.convert("start angle := 270.000000", {})
        assert "start angle := 90" in line

    def test_number_format_rule(self):
        """Test number format conversion."""
        rule = NumberFormatRule()

        # Test matching
        assert rule.matches("!number format := short float")
        assert not rule.matches("!number format := float")

        # Test conversion
        line, context = rule.convert("!number format := short float", {})
        assert line == "!number format := float"

    def test_orbit_conversion_rule(self):
        """Test orbit specification conversion."""
        rule = OrbitConversionRule()

        # Test matching
        assert rule.matches("orbit := noncircular")
        assert not rule.matches("orbit := circular")

        # Test conversion
        line, context = rule.convert("orbit := noncircular", {})
        assert line == "orbit := non-circular"

    def test_image_duration_rule(self):
        """Test image duration conversion."""
        rule = ImageDurationRule()

        # Test matching
        assert rule.matches("image duration (sec) := 120.000000")
        assert not rule.matches("time per projection (sec) := 1.0")

        # Test conversion
        line, context = rule.convert("image duration (sec) := 120.000000", {})
        assert "number of time frames := 1" in line
        assert "image duration (sec) [1] := 120.000000" in line

    def test_energy_window_rule(self):
        """Test energy window conversion."""
        rule_lower = EnergyWindowRule("lower")
        rule_upper = EnergyWindowRule("upper")

        # Test matching
        assert rule_lower.matches(";energy window lower level := 126.16")
        assert rule_upper.matches(";energy window upper level := 36.84")
        assert not rule_lower.matches(";energy window upper level := 36.84")

        # Test conversion
        line, context = rule_lower.convert(";energy window lower level := 126.16", {})
        assert line == "energy window lower level[1] := 126.16"

        line, context = rule_upper.convert(";energy window upper level := 36.84", {})
        assert line == "energy window upper level[1] := 36.84"

    def test_data_file_name_rule(self):
        """Test data file name conversion."""
        rule = DataFileNameRule()

        # Test matching
        assert rule.matches("!name of data file := /path/to/output.a00")

        # Test conversion
        line, context = rule.convert("!name of data file := /path/to/output.a00", {})
        assert line == "!name of data file := output.a00"

        # Test with override
        rule_override = DataFileNameRule(override_filename="custom.b01")
        line, context = rule_override.convert("!name of data file := output.a00", {})
        assert line == "!name of data file := custom.b01"


@pytest.mark.unit
class TestOrbitFileRule:
    """Test orbit file processing rule."""

    def test_orbit_file_rule_with_valid_file(self, tmp_path):
        """Test orbit file conversion with valid .cor file."""
        # Create a test orbit file
        orbit_file = tmp_path / "test.cor"
        radii_cm = [14.0, 15.5, 16.0, 14.0]  # cm
        with open(orbit_file, "w") as f:
            for r in radii_cm:
                f.write(f"      {r:6.3f}    64\n")

        # Create rule with input directory
        rule = OrbitFileRule(input_file_dir=tmp_path)

        # Test matching
        assert rule.matches(";# Non-Uniform Orbit File := test.cor")
        assert not rule.matches("orbit := noncircular")

        # Test conversion
        line, context = rule.convert(";# Non-Uniform Orbit File := test.cor", {})

        # Check that Radii line was created
        assert "Radii := {140, 155, 160, 140}" in line
        assert ";# Non-Uniform Orbit File := test.cor" in line

    def test_orbit_file_rule_missing_file(self, tmp_path):
        """Test orbit file rule when .cor file doesn't exist."""
        rule = OrbitFileRule(input_file_dir=tmp_path)

        # Test with non-existent file
        line, context = rule.convert(";# Non-Uniform Orbit File := missing.cor", {})

        # Should return original line when file not found
        assert line == ";# Non-Uniform Orbit File := missing.cor"

    def test_orbit_file_rule_only_processes_once(self, tmp_path):
        """Test that orbit file is only processed once."""
        orbit_file = tmp_path / "test.cor"
        with open(orbit_file, "w") as f:
            f.write("14.0    64\n15.0    64\n")

        rule = OrbitFileRule(input_file_dir=tmp_path)

        # First call should process
        assert rule.matches(";# Non-Uniform Orbit File := test.cor")
        line1, _ = rule.convert(";# Non-Uniform Orbit File := test.cor", {})
        assert "Radii" in line1

        # Second call should not match (already processed)
        assert not rule.matches(";# Non-Uniform Orbit File := test.cor")


@pytest.mark.unit
class TestSimindToStirConverter:
    """Test the main converter class."""

    def test_converter_initialization(self):
        """Test converter initialization."""
        converter = SimindToStirConverter()
        assert converter is not None
        assert converter.config is not None
        assert len(converter.rules) > 0

    def test_converter_with_custom_config(self):
        """Test converter with custom configuration."""
        config = ConversionConfig(
            radius_scale_factor=1.0,
            angle_offset=90.0,
        )
        converter = SimindToStirConverter(config)
        assert converter.config.radius_scale_factor == 1.0

    def test_penetrate_headers_apply_enum_metadata(self, tmp_path):
        """Ensure penetrate metadata from enum is propagated to headers."""
        converter = SimindToStirConverter()
        h00_file = tmp_path / "output.h00"
        h00_file.write_text(
            "\n".join(
                [
                    "!name of data file := output.a00",
                    "patient name := phantom",
                    "!study ID := study123",
                    "data description := placeholder",
                    "!END OF INTERFILE :=",
                ]
            )
        )
        component = PenetrateOutputType.COLL_SCATTER_PRIMARY_ATT_BACK
        binary_file = tmp_path / f"output.b{component.value:02d}"
        binary_file.write_bytes(b"\x00\x00\x00\x00")

        outputs = converter.create_penetrate_headers_from_template(
            str(h00_file), "output", str(tmp_path)
        )
        header_path = tmp_path / f"output_component_{component.value:02d}.hs"
        header_text = header_path.read_text()

        assert component.slug in outputs
        assert f"patient name := {component.slug}_{binary_file.name}" in header_text
        assert component.description in header_text

    def test_convert_line(self):
        """Test single line conversion."""
        converter = SimindToStirConverter()

        # Test number format conversion
        line, context = converter.convert_line("!number format := short float", {})
        assert line == "!number format := float"

        # Test start angle conversion
        line, context = converter.convert_line("start angle := 0.000000", {})
        assert "180" in line

    def test_parse_interfile_line(self):
        """Test interfile line parsing using shared parser."""
        # Test valid parameter line
        key, value = parse_interfile_line("!matrix size [1] := 128")
        assert key == "!matrix size [1]"
        assert value == "128"

        # Test with spaces
        key, value = parse_interfile_line("scaling factor (mm/pixel) [1] := 4.419600")
        assert key == "scaling factor (mm/pixel) [1]"
        assert value == "4.419600"

        # Test comment line
        key, value = parse_interfile_line(";# This is a comment")
        assert key is None
        assert value is None

        # Test section header
        key, value = parse_interfile_line("!GENERAL DATA :=")
        assert key is None
        assert value is None


@pytest.mark.unit
class TestConverterWithRealData:
    """Test converter with realistic SIMIND data."""

    def create_sample_h00_file(self, tmp_path, include_orbit=False):
        """Create a sample SIMIND .h00 file."""
        h00_file = tmp_path / "output.h00"

        content = """!INTERFILE :=
!imaging modality := nucmed
!originating system := simind
!version of keys := 3.3
program author := M Ljungberg, Lund University

!GENERAL DATA :=
!data offset in bytes := 0
!name of data file := output.a00
patient name := test_patient

!GENERAL IMAGE DATA :=
!type of data := tomographic
imagedata byte order := LITTLEENDIAN
;energy window lower level := 126.16
;energy window upper level := 36.84
!number of energy windows := 1
!matrix size [1] := 128
!matrix size [2] := 128
!number format := short float
!number of bytes per pixel := 4
scaling factor (mm/pixel) [1] := 4.419600
scaling factor (mm/pixel) [2] := 4.419600

!SPECT STUDY (General) :=
!extent of rotation := 360
!process status := acquired
!number of projections := 60
image duration (sec) := 60.000000

!SPECT STUDY (acquired data) :=
"""

        if include_orbit:
            content += "orbit := noncircular\n"
            content += ";# Non-Uniform Orbit File := output.cor\n"

            # Create corresponding .cor file
            cor_file = tmp_path / "output.cor"
            with open(cor_file, "w") as f:
                for i in range(60):
                    radius = 15.0 + (i % 10) * 0.5  # Varying radius
                    f.write(f"      {radius:6.3f}    64\n")
        else:
            content += "orbit := circular\n"

        content += """!direction of rotation := CW
start angle := 0.000000

!END OF INTERFILE :=
"""

        with open(h00_file, "w") as f:
            f.write(content)

        return h00_file

    def test_convert_file_basic(self, tmp_path):
        """Test basic file conversion."""
        h00_file = self.create_sample_h00_file(tmp_path, include_orbit=False)
        hs_file = tmp_path / "output.hs"

        converter = SimindToStirConverter()
        converter.convert_file(str(h00_file), str(hs_file))

        # Check output file exists
        assert hs_file.exists()

        # Read and verify content
        with open(hs_file, "r") as f:
            content = f.read()

            # Check key conversions
            assert "!number format := float" in content
            assert "start angle := 180" in content
            assert "orbit := circular" in content
            assert "number of time frames := 1" in content
            assert "image duration (sec) [1] := 60.000000" in content
            assert "energy window lower level[1] := 126.16" in content

    def test_convert_file_with_orbit(self, tmp_path):
        """Test file conversion with non-circular orbit."""
        h00_file = self.create_sample_h00_file(tmp_path, include_orbit=True)
        hs_file = tmp_path / "output.hs"

        converter = SimindToStirConverter()
        converter.convert_file(str(h00_file), str(hs_file))

        # Check output file exists
        assert hs_file.exists()

        # Read and verify content
        with open(hs_file, "r") as f:
            content = f.read()

            # Check orbit-specific conversions
            assert "orbit := non-circular" in content
            assert "Radii := {" in content

            # Verify Radii array has correct number of values
            radii_line = [line for line in content.split("\n") if "Radii :=" in line][0]
            radii_count = radii_line.count(",") + 1
            assert radii_count == 60  # Should match number of projections

            # Verify radii are in mm (should be 150-195 based on our test data)
            radii_str = radii_line.split("{")[1].split("}")[0]
            radii = [int(x.strip()) for x in radii_str.split(",")]
            assert all(140 <= r <= 200 for r in radii)

    def test_convert_file_default_output_name(self, tmp_path):
        """Test that output filename defaults to .hs extension."""
        h00_file = self.create_sample_h00_file(tmp_path)

        converter = SimindToStirConverter()
        converter.convert_file(str(h00_file))

        # Default output should be .hs
        hs_file = tmp_path / "output.hs"
        assert hs_file.exists()

    def test_read_parameter(self, tmp_path):
        """Test reading parameter from header file."""
        h00_file = self.create_sample_h00_file(tmp_path)
        hs_file = tmp_path / "output.hs"

        converter = SimindToStirConverter()
        converter.convert_file(str(h00_file), str(hs_file))

        # Read parameters
        matrix_size = converter.read_parameter(str(hs_file), "!matrix size [1]")
        assert matrix_size == "128"

        scaling = converter.read_parameter(
            str(hs_file), "scaling factor (mm/pixel) [1]"
        )
        assert scaling == "4.419600"

    def test_edit_parameter(self, tmp_path):
        """Test editing parameter in header file."""
        h00_file = self.create_sample_h00_file(tmp_path)
        hs_file = tmp_path / "output.hs"

        converter = SimindToStirConverter()
        converter.convert_file(str(h00_file), str(hs_file))

        # Edit a parameter
        converter.edit_parameter(str(hs_file), "!matrix size [1]", "256")

        # Verify edit
        new_value = converter.read_parameter(str(hs_file), "!matrix size [1]")
        assert new_value == "256"


@pytest.mark.unit
def test_conversion_config_defaults():
    """Test ConversionConfig default values."""
    config = ConversionConfig()

    assert config.radius_scale_factor == 10.0  # cm to mm
    assert config.angle_offset == 180.0
    assert config.default_number_format == "float"
    assert len(config.ignored_patterns) > 0
    assert "program" in config.ignored_patterns
    assert "patient" in config.ignored_patterns

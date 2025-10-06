import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# Conditional import for SIRF to avoid CI dependencies
try:
    from sirf.STIR import AcquisitionData

    SIRF_AVAILABLE = True
except ImportError:
    AcquisitionData = type(None)
    SIRF_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass
class ConversionConfig:
    """Configuration for SIMIND to STIR conversion."""

    radius_scale_factor: float = 10.0  # cm to mm
    angle_offset: float = 180.0  # degrees
    default_number_format: str = "float"
    ignored_patterns: List[str] = None

    def __post_init__(self):
        if self.ignored_patterns is None:
            self.ignored_patterns = [
                "program",
                "patient",
                "institution",
                "contact",
                "ID",
                "exam type",
                "detector head",
                "number of images/energy window",
                "time per projection",
                "data description",
                "total number of images",
                "acquisition mode",
            ]


class ConversionRule:
    """Base class for conversion rules."""

    def matches(self, line: str) -> bool:
        """Check if this rule applies to the given line."""
        raise NotImplementedError

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Convert the line and return new line plus updated context."""
        raise NotImplementedError


class RadiusConversionRule(ConversionRule):
    """Convert radius values with scaling."""

    def __init__(self, scale_factor: float = 1.0):  # Default to no scaling
        self.scale_factor = scale_factor

    def matches(self, line: str) -> bool:
        return "Radius" in line and ":=" in line

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        try:
            # Assume .h00 files contain radius in mm (SIMIND's inconsistent behavior)
            # So use radius value as-is for STIR (which expects mm)
            radius_value = float(line.split()[-1])
            return f"Radius := {radius_value}", context
        except (ValueError, IndexError) as e:
            logging.warning(f"Failed to convert radius line '{line}': {e}")
            return line, context


class OrbitFileRule(ConversionRule):
    """Process non-circular orbit file reference and insert Radii array."""

    def __init__(self, input_file_dir: Optional[Path] = None):
        self.input_file_dir = input_file_dir
        self.orbit_file_processed = False

    def matches(self, line: str) -> bool:
        return ";# Non-Uniform Orbit File" in line and not self.orbit_file_processed

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        try:
            # Extract orbit filename
            orbit_filename = line.split(":=")[-1].strip()

            # Build full path to orbit file
            if self.input_file_dir:
                orbit_path = self.input_file_dir / orbit_filename
            else:
                orbit_path = Path(orbit_filename)

            if not orbit_path.exists():
                logging.warning(f"Orbit file not found: {orbit_path}")
                self.orbit_file_processed = True
                return line, context

            # Read radii from orbit file (first column, in cm)
            radii_cm = []
            with open(orbit_path, "r") as f:
                for file_line in f:
                    parts = file_line.strip().split()
                    if parts:
                        radii_cm.append(float(parts[0]))

            # Convert cm to mm
            radii_mm = [int(round(r * 10)) for r in radii_cm]

            # Format as STIR Radii array
            radii_str = ", ".join(str(r) for r in radii_mm)
            radii_line = f"Radii := {{{radii_str}}}"

            logging.info(
                f"Converted {len(radii_mm)} radii from orbit file {orbit_filename}"
            )
            self.orbit_file_processed = True

            # Return both the commented orbit file line and the new Radii line
            return (
                f";# Non-Uniform Orbit File := {orbit_filename}\n{radii_line}",
                context,
            )

        except Exception as e:
            logging.warning(f"Failed to process orbit file from line '{line}': {e}")
            self.orbit_file_processed = True
            return line, context


class StartAngleConversionRule(ConversionRule):
    """Convert start angle with offset."""

    def __init__(self, angle_offset: float = 180.0):
        self.angle_offset = angle_offset

    def matches(self, line: str) -> bool:
        return "start angle" in line and ":=" in line

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        try:
            angle = float(line.split()[3]) + self.angle_offset
            return f"start angle := {angle % 360}", context
        except (ValueError, IndexError) as e:
            logging.warning(f"Failed to convert start angle line '{line}': {e}")
            return line, context


class RotationDirectionRule(ConversionRule):
    """Track rotation direction for context."""

    def matches(self, line: str) -> bool:
        return "CCW" in line or "CW" in line

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        if "CCW" in line:
            context["rotation_direction"] = "CCW"
        elif "CW" in line:
            context["rotation_direction"] = "CW"
        return line, context


class NumberFormatRule(ConversionRule):
    """Convert number format specifications."""

    def matches(self, line: str) -> bool:
        return "!number format := short float" in line

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        return "!number format := float", context


class OrbitConversionRule(ConversionRule):
    """Convert orbit specifications."""

    def matches(self, line: str) -> bool:
        return "orbit" in line and "noncircular" in line

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        return "orbit := non-circular", context


class ImageDurationRule(ConversionRule):
    """Convert image duration to STIR format."""

    def matches(self, line: str) -> bool:
        return "image duration" in line and ":=" in line

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        try:
            parts = line.split()
            duration = parts[4]
            return (
                f"number of time frames := 1\nimage duration (sec) [1] := {duration}",
                context,
            )
        except (IndexError, ValueError) as e:
            logging.warning(f"Failed to convert image duration line '{line}': {e}")
            return line, context


class EnergyWindowRule(ConversionRule):
    """Convert energy window specifications."""

    def __init__(self, window_type: str):
        self.window_type = window_type  # "lower" or "upper"

    def matches(self, line: str) -> bool:
        return f";energy window {self.window_type} level" in line

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        try:
            value = line.split()[-1]
            return f"energy window {self.window_type} level[1] := {value}", context
        except IndexError as e:
            logging.warning(f"Failed to convert energy window line '{line}': {e}")
            return line, context


class DataFileNameRule(ConversionRule):
    """Convert data file name references."""

    def __init__(self, override_filename: Optional[str] = None):
        self.override_filename = override_filename

    def matches(self, line: str) -> bool:
        return "!name of data file" in line

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        try:
            if self.override_filename:
                # Use the override filename
                return f"!name of data file := {self.override_filename}", context
            else:
                # Use existing filename from the header line
                file = Path(line.split()[5])
                return f"!name of data file := {file.stem + file.suffix}", context
        except IndexError as e:
            logging.warning(f"Failed to convert data file name line '{line}': {e}")
            return line, context


class IgnorePatternRule(ConversionRule):
    """Add semicolon to ignored pattern lines."""

    def __init__(self, patterns: List[str]):
        self.patterns = patterns

    def matches(self, line: str) -> bool:
        return any(pattern in line for pattern in self.patterns)

    def convert(self, line: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        return ";" + line, context


class SimindToStirConverter:
    """
    Enhanced SIMIND to STIR converter with configurable rules and editing
    capabilities.
    """

    def __init__(self, config: Optional[ConversionConfig] = None):
        self.config = config or ConversionConfig()
        self.input_file_dir = None  # Will be set during convert_file
        self.rules = self._create_rules()
        self.logger = logging.getLogger(__name__)

    def _create_rules(
        self, data_file_override: Optional[str] = None
    ) -> List[ConversionRule]:
        """Create conversion rules in order of priority."""
        return [
            IgnorePatternRule(self.config.ignored_patterns),
            OrbitFileRule(self.input_file_dir),  # Process orbit file before other rules
            RadiusConversionRule(self.config.radius_scale_factor),
            StartAngleConversionRule(self.config.angle_offset),
            RotationDirectionRule(),
            NumberFormatRule(),
            OrbitConversionRule(),
            ImageDurationRule(),
            EnergyWindowRule("lower"),
            EnergyWindowRule("upper"),
            DataFileNameRule(data_file_override),
        ]

    def convert_line(
        self, line: str, context: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Convert a single line using the first matching rule."""
        line = line.strip()

        for rule in self.rules:
            if rule.matches(line):
                return rule.convert(line, context)

        return line, context

    @contextmanager
    def _safe_file_operation(self, input_file: str, output_file: str):
        """Context manager for safe file operations with cleanup."""
        temp_file = output_file + ".tmp"
        try:
            with open(input_file, "r") as f_in, open(temp_file, "w") as f_out:
                yield f_in, f_out

            # Only replace original if conversion succeeded
            if os.path.exists(temp_file):
                if os.path.exists(output_file):
                    os.remove(output_file)
                os.rename(temp_file, output_file)
        except Exception:
            # Clean up temp file if something went wrong
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise

    @staticmethod
    def _parse_interfile_line(line: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse an interfile line and return parameter and value."""
        line = line.strip()

        # Skip comments, empty lines, and section headers
        if (
            not line
            or line.startswith(";")
            or line.startswith("#")
            or line.endswith(":=")
        ):
            return None, None

        # Handle := separator (preferred)
        if ":=" in line:
            key, _, value = line.partition(":=")
            return key.strip(), value.strip()

        return None, None

    def convert_file(
        self,
        input_filename: str,
        output_filename: Optional[str] = None,
        data_file: Optional[str] = None,
        return_object: bool = False,
    ) -> Optional[AcquisitionData]:
        """Convert a SIMIND header file to STIR format."""

        if not input_filename.endswith(".h00"):
            raise ValueError("Input file must have .h00 extension")

        if output_filename is None:
            output_filename = input_filename.replace(".h00", ".hs")

        # Set input directory for orbit file resolution
        self.input_file_dir = Path(input_filename).parent

        # Create rules with optional data file override
        if data_file is not None:
            self.rules = self._create_rules(data_file)
        else:
            # Recreate rules to pick up the new input_file_dir
            self.rules = self._create_rules()

        context = {"rotation_direction": None}

        try:
            with self._safe_file_operation(input_filename, output_filename) as (
                f_in,
                f_out,
            ):
                for line_num, line in enumerate(f_in, 1):
                    try:
                        converted_line, context = self.convert_line(line, context)
                        f_out.write(converted_line + "\n")
                    except Exception as e:
                        self.logger.error(
                            f"Error converting line {line_num}: {line.strip()}"
                        )
                        self.logger.error(f"Error details: {e}")
                        # Write original line as fallback
                        f_out.write(line)

            self.logger.info(
                f"Successfully converted {input_filename} to {output_filename}"
            )
            if data_file:
                self.logger.info(f"Used data file override: {data_file}")

            if return_object:
                return AcquisitionData(output_filename)

        except Exception as e:
            self.logger.error(f"Failed to convert {input_filename}: {e}")
            raise
        finally:
            # Reset rules to default after conversion
            if data_file is not None:
                self.rules = self._create_rules()

        return None

    def create_penetrate_headers_from_template(
        self, h00_file: str, output_prefix: str, output_dir: str
    ) -> Dict[str, AcquisitionData]:
        """
        Create multiple STIR headers for penetrate routine from single .h00 template.

        The penetrate routine creates only one .h00 file pointing to a
        non-existent .a00,
        but multiple .bXX binary files. This method creates separate .hs headers
        for each .bXX file.

        Args:
            h00_file: Path to the single .h00 template file from penetrate routine
            output_prefix: Prefix used for output files
            output_dir: Directory containing .bXX files

        Returns:
            Dictionary mapping component names to AcquisitionData objects
        """
        output_dir = Path(output_dir)
        outputs = {}

        # First convert the template .h00 to .hs format
        template_hs = h00_file.replace(".h00", "_template.hs")
        self.convert_file(h00_file, template_hs)

        # Read the template header content
        with open(template_hs, "r") as f:
            template_content = f.read()

        # Look for .bXX files and create headers for each
        for i in range(1, 20):  # b01 to b19
            binary_file = output_dir / f"{output_prefix}.b{i:02d}"

            if binary_file.exists():
                try:
                    # Create .hs file for this component
                    component_hs = output_dir / f"{output_prefix}_component_{i:02d}.hs"

                    # Modify template content for this specific binary file
                    modified_content = self._modify_header_for_binary(
                        template_content, binary_file.name, i
                    )

                    # Write the modified header
                    with open(component_hs, "w") as f:
                        f.write(modified_content)

                    # Create AcquisitionData object
                    acquisition_data = AcquisitionData(str(component_hs))

                    # Generate component name
                    component_name = self._get_penetrate_output_name(i)
                    outputs[component_name] = acquisition_data

                    self.logger.info(
                        f"Created STIR header for {component_name}: {component_hs.name}"
                    )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to create header for {binary_file}: {e}"
                    )

        # Clean up template file
        if os.path.exists(template_hs):
            os.remove(template_hs)

        return outputs

    def _modify_header_for_binary(
        self, template_content: str, binary_filename: str, component_number: int
    ) -> str:
        """
        Modify template header content to point to specific binary file.

        Args:
            template_content: Original header content from template
            binary_filename: Name of the .bXX binary file
            component_number: Component number (1-19)

        Returns:
            Modified header content
        """
        lines = template_content.split("\n")
        modified_lines = []

        for line in lines:
            # Update data file reference
            if line.startswith("!name of data file"):
                modified_lines.append(f"!name of data file := {binary_filename}")

            # Update patient name to include component info
            elif line.startswith("patient name"):
                component_name = self._get_penetrate_output_name(component_number)
                modified_lines.append(
                    f"patient name := {component_name}_{binary_filename}"
                )

            # Update study ID
            elif line.startswith("!study ID"):
                study_base = Path(binary_filename).stem
                modified_lines.append(f"!study ID := {study_base}")

            # Update data description to include component info
            elif line.startswith("data description"):
                component_desc = self._get_penetrate_component_description(
                    component_number
                )
                modified_lines.append(f"data description := {component_desc}")

            else:
                modified_lines.append(line)

        return "\n".join(modified_lines)

    def _get_penetrate_output_name(self, component_number: int) -> str:
        """Get descriptive name for penetrate output component."""
        name_mapping = {
            1: "all_interactions",
            2: "geom_coll_primary",
            3: "septal_pen_primary",
            4: "coll_scatter_primary",
            5: "coll_xray_primary",
            6: "geom_coll_scattered",
            7: "septal_pen_scattered",
            8: "coll_scatter_scattered",
            9: "coll_xray_scattered",
            10: "geom_coll_primary_back",
            11: "septal_pen_primary_back",
            12: "coll_scatter_primary_back",
            13: "coll_xray_primary_back",
            14: "geom_coll_scattered_back",
            15: "septal_pen_scattered_back",
            16: "coll_scatter_scattered_back",
            17: "coll_xray_scattered_back",
            18: "unscattered_unattenuated",
            19: "unscattered_unattenuated_geom_coll",
        }
        return name_mapping.get(component_number, f"component_{component_number:02d}")

    def _get_penetrate_component_description(self, component_number: int) -> str:
        """Get detailed description for penetrate output component."""
        descriptions = {
            1: "All type of interactions",
            2: "Geometrically collimated primary attenuated photons",
            3: "Septal penetration from primary attenuated photons",
            4: "Collimator scatter from primary attenuated photons",
            5: "X-rays from collimator (primary attenuated photons)",
            6: "Geometrically collimated scattered photons",
            7: "Septal penetration from scattered photons",
            8: "Collimator scatter from scattered photons",
            9: "X-rays from collimator (scattered photons)",
            10: (
                "Geometrically collimated primary attenuated photons (with backscatter)"
            ),
            11: "Septal penetration from primary attenuated photons (with backscatter)",
            12: "Collimator scatter from primary attenuated photons (with backscatter)",
            13: "X-rays from collimator, primary attenuated photons (with backscatter)",
            14: "Geometrically collimated scattered photons (with backscatter)",
            15: "Septal penetration from scattered photons (with backscatter)",
            16: "Collimator scatter from scattered photons (with backscatter)",
            17: "X-rays from collimator, scattered photons (with backscatter)",
            18: "Photons without scattering and attenuation in phantom",
            19: "Photons without scattering/attenuation, geometrically collimated",
        }
        return descriptions.get(
            component_number, f"Penetrate component {component_number}"
        )

    def find_penetrate_h00_file(
        self, output_prefix: str, output_dir: str
    ) -> Optional[str]:
        """
        Find the single .h00 file created by penetrate routine.

        Args:
            output_prefix: Prefix used for output files
            output_dir: Directory containing output files

        Returns:
            Path to the .h00 file, or None if not found
        """
        output_dir = Path(output_dir)

        # Look for .h00 file with the output prefix
        h00_files = list(output_dir.glob(f"{output_prefix}*.h00"))

        if len(h00_files) == 1:
            return str(h00_files[0])
        elif len(h00_files) == 0:
            self.logger.warning(f"No .h00 file found with prefix {output_prefix}")
            return None
        else:
            # Multiple .h00 files - this might be scattwin, not penetrate
            self.logger.warning(
                "Multiple .h00 files found - this may not be penetrate routine output"
            )
            # Return the first one as fallback
            return str(h00_files[0])

    def read_parameter(self, filename: str, parameter: str) -> Optional[str]:
        """Read a parameter from a header file."""
        if not filename.endswith((".hs", ".h00")):
            self.logger.error("File must have .hs or .h00 extension")
            return None

        try:
            with open(filename, "r") as f:
                for line in f:
                    key, value = self._parse_interfile_line(line)
                    if key == parameter:
                        return value
        except FileNotFoundError:
            self.logger.error(f"File not found: {filename}")
        except Exception as e:
            self.logger.error(f"Error reading file {filename}: {e}")

        return None

    def edit_parameter(
        self,
        filename: str,
        parameter: str,
        value: Union[str, float],
        return_object: bool = False,
    ) -> Optional[AcquisitionData]:
        """Edit a parameter in a header file."""
        if not filename.endswith((".hs", ".h00")):
            self.logger.error("File must have .hs or .h00 extension")
            return None

        temp_filename = filename + ".tmp"

        try:
            with open(filename, "r") as f_in, open(temp_filename, "w") as f_out:
                parameter_found = False
                for line in f_in:
                    key, current_value = self._parse_interfile_line(line)
                    if key == parameter:
                        f_out.write(f"{parameter} := {value}\n")
                        parameter_found = True
                    else:
                        f_out.write(line)

                if not parameter_found:
                    self.logger.warning(
                        f"Parameter '{parameter}' not found in {filename}"
                    )

            # Replace original file
            os.replace(temp_filename, filename)
            self.logger.info(f"Parameter {parameter} set to {value}")

            return AcquisitionData(filename) if return_object else None

        except Exception as e:
            # Clean up temp file if it exists
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            self.logger.error(f"Error editing parameter in {filename}: {e}")
            raise

    def add_parameter(
        self,
        filename: str,
        parameter: str,
        value: Union[str, float],
        line_number: int = 0,
        return_object: bool = False,
    ) -> Optional[AcquisitionData]:
        """Add a parameter at a specific line number in an Interfile header file."""
        if not filename.endswith((".hs", ".h00")):
            self.logger.error("File must have .hs or .h00 extension")
            return None

        # First test if parameter already exists
        existing_value = self.read_parameter(filename, parameter)
        if existing_value is not None:
            self.logger.info(
                f"Parameter {parameter} already exists, editing instead of adding"
            )
            return self.edit_parameter(filename, parameter, value, return_object)

        temp_filename = filename + ".tmp"
        parameter_line = f"{parameter} := {value}\n"

        try:
            with open(filename, "r") as f_in:
                lines = f_in.readlines()

            with open(temp_filename, "w") as f_out:
                for i, line in enumerate(lines):
                    if i == line_number:
                        f_out.write(parameter_line)
                    f_out.write(line)

                # If line_number is beyond file length, append at end
                if len(lines) <= line_number:
                    f_out.write(parameter_line)

            # Replace original file
            os.replace(temp_filename, filename)
            self.logger.info(
                f"Parameter {parameter} added with value {value} at line {line_number}"
            )

            return AcquisitionData(filename) if return_object else None

        except Exception as e:
            # Clean up temp file if it exists
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            self.logger.error(f"Error adding parameter to {filename}: {e}")
            raise

    def validate_and_fix_scaling_factors(
        self, filename: str, image_data, tolerance: float = 0.1
    ) -> bool:
        """
        Validate scaling factors against image voxel sizes and fix if they differ
        within tolerance.

        Args:
            filename: Path to the header file
            image_data: SIRF ImageData object to get voxel sizes from
            tolerance: Maximum allowed difference in mm (default 0.1mm)

        Returns:
            bool: True if scaling factors were within tolerance, False if they were
                corrected
        """
        if not filename.endswith((".hs", ".h00")):
            self.logger.error("File must have .hs or .h00 extension")
            return False

        # Get voxel sizes from image data
        voxel_sizes = image_data.voxel_sizes()
        image_voxel_x = voxel_sizes[0]  # mm
        image_voxel_y = voxel_sizes[1]  # mm

        # Read current scaling factors from file
        current_scaling_x = self.read_parameter(
            filename, "scaling factor (mm/pixel) [1]"
        )
        current_scaling_y = self.read_parameter(
            filename, "scaling factor (mm/pixel) [2]"
        )

        if current_scaling_x is None or current_scaling_y is None:
            self.logger.warning(f"Could not read scaling factors from {filename}")
            # Set them to image voxel sizes
            self.edit_parameter(
                filename, "scaling factor (mm/pixel) [1]", image_voxel_x
            )
            self.edit_parameter(
                filename, "scaling factor (mm/pixel) [2]", image_voxel_y
            )
            self.logger.info(
                f"Set scaling factors to image voxel sizes: "
                f"[{image_voxel_x}, {image_voxel_y}]"
            )
            return False

        try:
            current_x = float(current_scaling_x)
            current_y = float(current_scaling_y)

            # Check if they're different but within tolerance
            diff_x = abs(current_x - image_voxel_x)
            diff_y = abs(current_y - image_voxel_y)

            if diff_x <= tolerance and diff_y <= tolerance:
                self.logger.debug(
                    f"Scaling factors are within tolerance: current=[{current_x}, "
                    f"{current_y}], image=[{image_voxel_x}, {image_voxel_y}]"
                )
                return True
            else:
                # Fix the scaling factors to match image voxel sizes
                self.logger.info(
                    f"Scaling factors differ beyond tolerance ({tolerance}mm)"
                )
                self.logger.info(f"  Current: [{current_x}, {current_y}]")
                self.logger.info(f"  Image:   [{image_voxel_x}, {image_voxel_y}]")
                self.logger.info(f"  Diff:    [{diff_x:.6f}, {diff_y:.6f}]")

                self.edit_parameter(
                    filename, "scaling factor (mm/pixel) [1]", image_voxel_x
                )
                self.edit_parameter(
                    filename, "scaling factor (mm/pixel) [2]", image_voxel_y
                )
                self.logger.info("Updated scaling factors to match image voxel sizes")
                return False

        except ValueError as e:
            self.logger.error(f"Error parsing scaling factors: {e}")
            # Set them to image voxel sizes as fallback
            self.edit_parameter(
                filename, "scaling factor (mm/pixel) [1]", image_voxel_x
            )
            self.edit_parameter(
                filename, "scaling factor (mm/pixel) [2]", image_voxel_y
            )
            return False

    def add_custom_rule(self, rule: ConversionRule, priority: int = None):
        """Add a custom conversion rule."""
        if priority is None:
            self.rules.append(rule)
        else:
            self.rules.insert(priority, rule)

    def validate_and_correct_radius(
        self,
        output_file: str,
        template_file: Optional[str] = None,
        tolerance_factor: float = 2.0,
    ) -> bool:
        """
        Validate radius in output file against template and correct if needed.

        Args:
            output_file: Path to the output .hs file
            template_file: Path to the template .hs file for comparison
            tolerance_factor: Factor by which radius can differ before correction

        Returns:
            bool: True if radius was within tolerance, False if corrected
        """
        if template_file is None or not os.path.exists(template_file):
            self.logger.debug(
                "No template file provided or found - skipping radius validation"
            )
            return True

        try:
            # Read radius from template
            template_radius = self.read_parameter(template_file, "Radius")
            if template_radius is None:
                self.logger.debug("No radius found in template file")
                return True

            template_radius = float(template_radius)

            # Read radius from output
            output_radius = self.read_parameter(output_file, "Radius")
            if output_radius is None:
                self.logger.warning(f"No radius found in output file {output_file}")
                return True

            output_radius = float(output_radius)

            # Check if radii are within tolerance
            ratio = output_radius / template_radius
            if 1 / tolerance_factor <= ratio <= tolerance_factor:
                self.logger.debug(
                    f"Radius validation passed: template={template_radius}, output={output_radius}"
                )
                return True

            # Radius mismatch detected - attempt correction
            self.logger.warning(
                f"Radius mismatch detected in {output_file}:\n"
                f"  Template: {template_radius} mm\n"
                f"  Output:   {output_radius} mm\n"
                f"  Ratio:    {ratio:.2f}"
            )

            # Correct the radius by setting it to template value
            self.edit_parameter(output_file, "Radius", template_radius)
            self.logger.info(
                f"Corrected radius in {output_file} to {template_radius} mm"
            )

            return False

        except (ValueError, TypeError) as e:
            self.logger.error(f"Error during radius validation: {e}")
            return True

### This file contains a few useful functions for converting SIMIND output to STIR format.
### It should probably be cleaned up and object-orientedified at some point.

### Author: Sam Porter, Efstathios Varzakis

import os
import re
import subprocess
import warnings

import numpy as np

# Conditional import for SIRF to avoid CI dependencies
try:
    from sirf.STIR import AcquisitionData, ImageData

    SIRF_AVAILABLE = True
except ImportError:
    # Create dummy types for type hints when SIRF is not available
    AcquisitionData = type(None)
    ImageData = type(None)
    SIRF_AVAILABLE = False


def parse_sinogram(template_sinogram):
    template_sinogram.write("tmp.hs")
    values = parse_interfile("tmp.hs")
    os.remove("tmp.hs")
    return values


def parse_interfile(filename):
    """
    Parses STIR interfile file and returns dictionary of key value pairs

    Args:
        filename (string): file name of interfile file

    Returns:
        dict: dictionary of key value pairs
    """

    values = {}
    with open(filename, "r") as file:
        for line in file:
            # Use regex to find lines with ':='
            match = re.search(r"([^;].*?)\s*:=\s*(.*)", line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                values[key] = value
    return values


def get_sirf_attenuation_from_simind(
    attn_filename, photopeak_energy=0.12, attn_type="mu"
):
    """Reads attenuation data from simind attenuation file and returns SIRF ImageData object

    Args:
        attn_filename (string): file name of simind attenuation file header
        new_filename (string): file name of new attenuation file header
        attn_type (str, optional): unit of attenuation binary file. Defaults to 'mu'.

    Returns:
        image: SIRF ImageData object containing attenuation data
    """

    if attn_type == "mu":
        data_type = np.float32
    elif attn_type == "rho*1000":
        data_type = np.uint16

    # remove suffix from filename if present
    if attn_filename[-3:] == "ict" or attn_filename[-3:] == "hct":
        attn_filename = attn_filename[:-4]

    attn = np.fromfile(attn_filename + ".ict", dtype=data_type)
    image = ImageData()

    header_dict = parse_interfile(attn_filename + ".hct")
    dim = [int(header_dict["!matrix size [%d]" % i]) for i in range(1, 4)][::-1]

    vsize = [
        float(header_dict["scaling factor (mm/pixel) [%d]" % i]) for i in range(1, 3)
    ]
    vsize.append(header_dict["# scaling factor (mm/pixel) [3]"])

    # origin looks like '-128.0000 -128.0000 128.000000'
    origin_string = header_dict["# Image Position First image"]
    origin = [float(i) for i in origin_string.split(" ")][::-1]

    image.initialise(dim=tuple(dim), vsize=tuple(vsize), origin=tuple(origin))
    attn = attn.reshape(dim)

    if attn_type == "mu":
        image.fill(attn)
    elif attn_type == "rho*100":
        image.fill(attn / 1000 * photopeak_energy)

    return image


def get_sirf_sinogram_from_simind(
    simind_header_filepath, script_path=".", circular=True
):
    """

    ** DEPRECATED METHOD - may still be useful down the line so retained for now **


    Reads sinogram data from simind header file and returns SIRF AcquisitionData object
    Outputs sinogram with counts/MBq/s as units

    Args:
        simind_header_filepath (string): file name of simind header file
        script_path (string, optional): path to simind conversion scripts. Defaults to ".".
        circular (bool, optional): whether to use circular or non-circular conversion script. Defaults to True.
    Returns:
        AcquisitionData: SIRF AcquisitionData object containing sinogram data
    """

    if circular:
        script = os.path.join(script_path, "convertSIMINDToSTIR.sh")
    else:
        script = os.path.join(script_path, "convertSIMINDToSTIR_noncirc.sh")
    subprocess.run(["sh", os.path.join(script_path, script), simind_header_filepath])

    return AcquisitionData(simind_header_filepath[:-4] + ".hs")


def convert_value(val: str):
    """
    Attempt to convert a string to int or float.
    Also converts a list of numbers if enclosed in { }.
    Otherwise, returns the stripped string.
    """
    val = val.strip()
    if val.startswith("{") and val.endswith("}"):
        try:
            return [float(x.strip()) for x in val[1:-1].split(",")]
        except Exception:
            return val
    try:
        if re.fullmatch(r"[-+]?\d+", val):
            return int(val)
        # Try float conversion if the value contains a decimal point or exponent.
        return float(val) if re.search(r"[.\deE+-]", val) else val
    except ValueError:
        return val


### PLEASE NOTE that the following functions are far from perfect and things may have been missed
### The naming converntions are a bit funny so there needs to be some cleaning up

STIR_ATTRIBUTE_MAPPING = {
    "number_of_views": "number_of_projections",
    "azimuthal_angle_extent": "extent_of_rotation",
    "view_offset": "start_angle",
    "calibration_factor": "calibration_factor",
    "radionuclide": "isotope_name",
    "energy_window_lower": "energy_window_lower",
    "energy_window_upper": "energy_window_upper",
    "scanner_type": "scanner_type",
    "number_of_rings": "number_of_rings",
    "number_of_detectors_per_ring": "number_of_detectors_per_ring",
    "inner_ring_diameter": "height_to_detector_surface",
    "tangential_sampling": "default_bin_size",
}


def harmonize_stir_attributes(attributes: dict) -> dict:
    """
    Standardizes STIR attributes by renaming fields according to STIR_ATTRIBUTE_MAPPING.

    Args:
        attributes (dict): Extracted attributes from STIR header file or get_info().

    Returns:
        dict: Standardized attribute dictionary.
    """
    harmonized_attributes = {}
    for key, value in attributes.items():
        standard_key = STIR_ATTRIBUTE_MAPPING.get(
            key, key
        )  # Rename if mapping exists, otherwise keep original
        harmonized_attributes[standard_key] = value

    # Special derived attributes
    if "inner_ring_diameter" in harmonized_attributes:
        harmonized_attributes["height_to_detector_surface"] = (
            harmonized_attributes["inner_ring_diameter"] / 2
        )

    return harmonized_attributes


def extract_attributes_from_stir(sinogram) -> dict:
    if isinstance(sinogram, str):
        return extract_attributes_from_stir_headerfile(sinogram)
    elif isinstance(sinogram, AcquisitionData):
        return extract_attributes_from_stir_sinogram(sinogram)

    raise ValueError("Input must be a file path or AcquisitionData object.")


def extract_attributes_from_stir_sinogram(sinogram: "AcquisitionData") -> dict:
    """
    Parse a STIR sinogram info string (from sinogram.get_info()) and extract attributes.

    Note that this probably isn't exhaustive but does work for PET and SPECT sinograms.

    Parameters
    ----------
    sinogram : AcquisitionData
        Object providing the get_info() method.

    Returns
    -------
    dict
        Dictionary of extracted attributes.
    """
    attributes = {}
    info_str = sinogram.get_info()  # assuming this returns a string
    lines = info_str.splitlines()

    # Define generic patterns: each tuple is (regex, converter, attribute key)
    patterns = [
        (
            re.compile(r"^Modality:\s*(\S+)", re.IGNORECASE),
            lambda m: m.group(1).strip(),
            "modality",
        ),
        (
            re.compile(r"^Calibration Factor:\s*([-+]?[0-9]*\.?[0-9]+)", re.IGNORECASE),
            lambda m: float(m.group(1)),
            "calibration_factor",
        ),
        (
            re.compile(r"^Radionuclide:\s*(.+)", re.IGNORECASE),
            lambda m: m.group(1).strip(),
            "radionuclide",
        ),
        (
            re.compile(r"^Energy\s+([-+]?[0-9]*\.?[0-9]+)", re.IGNORECASE),
            lambda m: float(m.group(1)),
            "energy",
        ),
        (
            re.compile(r"^Half-life:\s*([-+]?[0-9]*\.?[0-9]+)", re.IGNORECASE),
            lambda m: float(m.group(1)),
            "half_life",
        ),
        (
            re.compile(r"^Branching ratio:\s*([-+]?[0-9]*\.?[0-9]+)", re.IGNORECASE),
            lambda m: float(m.group(1)),
            "branching_ratio",
        ),
        (
            re.compile(r"^Patient position:\s*(.+)", re.IGNORECASE),
            lambda m: m.group(1).strip(),
            "patient_position",
        ),
        (
            re.compile(r"^Scan start time:\s*([-+]?[0-9]*\.?[0-9]+)", re.IGNORECASE),
            lambda m: float(m.group(1)),
            "scan_start_time",
        ),
        (
            re.compile(r"number of energy windows\s*[:=]+\s*(\d+)", re.IGNORECASE),
            lambda m: int(m.group(1)),
            "number_of_energy_windows",
        ),
        (
            re.compile(
                r"energy window lower level\[\d+\]\s*[:=]+\s*([-+]?[0-9]*\.?[0-9]+)",
                re.IGNORECASE,
            ),
            lambda m: float(m.group(1)),
            "energy_window_lower",
        ),
        (
            re.compile(
                r"energy window upper level\[\d+\]\s*[:=]+\s*([-+]?[0-9]*\.?[0-9]+)",
                re.IGNORECASE,
            ),
            lambda m: float(m.group(1)),
            "energy_window_upper",
        ),
        # Scanner parameters:
        (
            re.compile(r"Scanner type\s*[:=]+\s*(\S+)", re.IGNORECASE),
            lambda m: m.group(1).strip(),
            "scanner_type",
        ),
        (
            re.compile(r"Number of rings\s*[:=]+\s*(\d+)", re.IGNORECASE),
            lambda m: int(m.group(1)),
            "number_of_rings",
        ),
        (
            re.compile(
                r"Number of detectors per ring\s*[:=]+\s*([-]?\d+)", re.IGNORECASE
            ),
            lambda m: int(m.group(1)),
            "number_of_detectors_per_ring",
        ),
        (
            re.compile(
                r"Inner ring diameter\s*\(cm\)\s*[:=]+\s*([-+]?[0-9]*\.?[0-9]+)",
                re.IGNORECASE,
            ),
            lambda m: float(m.group(1)),
            "inner_ring_diameter",
        ),
        (
            re.compile(
                r"Average depth of interaction\s*\(cm\)\s*[:=]+\s*([-+]?[0-9]*\.?[0-9]+)",
                re.IGNORECASE,
            ),
            lambda m: float(m.group(1)),
            "average_depth_of_interaction",
        ),
        (
            re.compile(
                r"Distance between rings\s*\(cm\)\s*[:=]+\s*([-+]?[0-9]*\.?[0-9]+)",
                re.IGNORECASE,
            ),
            lambda m: float(m.group(1)),
            "distance_between_rings",
        ),
        (
            re.compile(
                r"Default bin size\s*\(cm\)\s*[:=]+\s*([-+]?[0-9]*\.?[0-9]+)",
                re.IGNORECASE,
            ),
            lambda m: float(m.group(1)),
            "default_bin_size",
        ),
        (
            re.compile(
                r"View offset\s*\(degrees\)\s*[:=]+\s*([-+]?[0-9]*\.?[0-9]+)",
                re.IGNORECASE,
            ),
            lambda m: float(m.group(1)),
            "view_offset",
        ),
        (
            re.compile(
                r"Maximum number of non-arc-corrected bins\s*[:=]+\s*(\d+)",
                re.IGNORECASE,
            ),
            lambda m: int(m.group(1)),
            "max_non_arc_corrected_bins",
        ),
        (
            re.compile(
                r"Default number of arc-corrected bins\s*[:=]+\s*(\d+)", re.IGNORECASE
            ),
            lambda m: int(m.group(1)),
            "default_arc_corrected_bins",
        ),
        (
            re.compile(
                r"Number of blocks per bucket in transaxial direction\s*[:=]+\s*([-]?\d+)",
                re.IGNORECASE,
            ),
            lambda m: int(m.group(1)),
            "number_of_blocks_per_bucket_transaxial",
        ),
        (
            re.compile(
                r"Number of blocks per bucket in axial direction\s*[:=]+\s*([-]?\d+)",
                re.IGNORECASE,
            ),
            lambda m: int(m.group(1)),
            "number_of_blocks_per_bucket_axial",
        ),
        (
            re.compile(
                r"Number of crystals per block in axial direction\s*[:=]+\s*([-]?\d+)",
                re.IGNORECASE,
            ),
            lambda m: int(m.group(1)),
            "number_of_crystals_per_block_axial",
        ),
        (
            re.compile(
                r"Number of crystals per block in transaxial direction\s*[:=]+\s*([-]?\d+)",
                re.IGNORECASE,
            ),
            lambda m: int(m.group(1)),
            "number_of_crystals_per_block_transaxial",
        ),
        (
            re.compile(r"Number of detector layers\s*[:=]+\s*(\d+)", re.IGNORECASE),
            lambda m: int(m.group(1)),
            "number_of_detector_layers",
        ),
        (
            re.compile(
                r"Number of crystals per singles unit in axial direction\s*[:=]+\s*([-]?\d+)",
                re.IGNORECASE,
            ),
            lambda m: int(m.group(1)),
            "number_of_crystals_per_singles_unit_axial",
        ),
        (
            re.compile(
                r"Number of crystals per singles unit in transaxial direction\s*[:=]+\s*([-]?\d+)",
                re.IGNORECASE,
            ),
            lambda m: int(m.group(1)),
            "number_of_crystals_per_singles_unit_transaxial",
        ),
        (
            re.compile(r"Scanner geometry\s*\(.+?\)\s*[:=]+\s*(\S+)", re.IGNORECASE),
            lambda m: m.group(1).strip(),
            "scanner_geometry",
        ),
        # Other parameters:
        (
            re.compile(
                r"start vertical bed position\s*\(mm\)\s*[:=]+\s*([-+]?[0-9]*\.?[0-9]+)",
                re.IGNORECASE,
            ),
            lambda m: float(m.group(1)),
            "start_vertical_bed_position",
        ),
        (
            re.compile(
                r"start horizontal bed position\s*\(mm\)\s*[:=]+\s*([-+]?[0-9]*\.?[0-9]+)",
                re.IGNORECASE,
            ),
            lambda m: float(m.group(1)),
            "start_horizontal_bed_position",
        ),
        (
            re.compile(
                r"TOF mashing factor in data\s*[:=]+\s*([-+]?[0-9]*\.?[0-9]+)",
                re.IGNORECASE,
            ),
            lambda m: float(m.group(1)),
            "tof_mashing_factor",
        ),
        (
            re.compile(
                r"Number of TOF positions in data\s*[:=]+\s*(\d+)", re.IGNORECASE
            ),
            lambda m: int(m.group(1)),
            "number_of_tof_positions",
        ),
        (
            re.compile(r"Number of Views:\s*(\d+)", re.IGNORECASE),
            lambda m: int(m.group(1)),
            "number_of_views",
        ),
        (
            re.compile(
                r"Number of axial positions per seg:\s*\{?\s*(\d+)\s*\}?", re.IGNORECASE
            ),
            lambda m: int(m.group(1)),
            "number_of_axial_positions_per_seg",
        ),
        (
            re.compile(r"Number of tangential positions:\s*(\d+)", re.IGNORECASE),
            lambda m: int(m.group(1)),
            "number_of_tangential_positions",
        ),
        (
            re.compile(
                r"Azimuthal angle increment\s*\(deg\)\s*[:=]+\s*([-+]?[0-9]*\.?[0-9]+)",
                re.IGNORECASE,
            ),
            lambda m: float(m.group(1)),
            "azimuthal_angle_increment",
        ),
        (
            re.compile(
                r"Azimuthal angle extent\s*\(deg\)\s*[:=]+\s*([-+]?[0-9]*\.?[0-9]+)",
                re.IGNORECASE,
            ),
            lambda m: float(m.group(1)),
            "azimuthal_angle_extent",
        ),
        (
            re.compile(
                r"tangential sampling\s*[:=]+\s*([-+]?[0-9]*\.?[0-9]+)", re.IGNORECASE
            ),
            lambda m: float(m.group(1)),
            "tangential_sampling",
        ),
    ]

    # Special handling for "ring differences per segment:" which appears on one line, followed by a tuple in the next.
    ring_diff_pattern = re.compile(r"\(\s*([-+]?\d+)\s*,\s*([-+]?\d+)\s*\)")

    skip_next = False  # flag to skip a line already processed (for ring differences)
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        line = line.strip()
        if not line:
            continue

        # Skip block header lines
        if (
            line.startswith("Radionuclide Parameters:")
            or line.startswith("Scanner parameters:=")
            or line.startswith("End scanner parameters:=")
            or line.startswith("ProjDataInfoCylindricalArcCorr")
        ):
            continue

        # Handle "ring differences per segment:" (the value is on the next non-empty line)
        if line.lower().startswith("ring differences per segment"):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                m = ring_diff_pattern.search(lines[j])
                if m:
                    attributes["ring_differences_per_segment"] = (
                        int(m.group(1)),
                        int(m.group(2)),
                    )
                skip_next = True
            continue

        # Try generic patterns.
        for pattern, converter, key in patterns:
            m = pattern.search(line)
            if m:
                try:
                    attributes[key] = converter(m)
                except Exception as e:
                    warnings.warn(
                        f"Error converting key '{key}' from '{m.group(0)}': {e}"
                    )
                break

    # Mapping differences
    attributes["number_of_projections"] = attributes.get("number_of_views")
    attributes["height_to_detector_surface"] = (
        attributes.get("inner_ring_diameter", 0) / 2 * 10
    )
    attributes["extent_of_rotation"] = attributes.get("azimuthal_angle_extent")
    attributes["start_angle"] = attributes.get("view_offset")
    attributes["direction_of_rotation"] = (
        "CCW" if attributes.get("azimuthal_angle_increment", 1) > 0 else "CW"
    )

    return harmonize_stir_attributes(attributes)


def extract_attributes_from_stir_headerfile(filename: str) -> dict:
    """
    Parse a STIR header file and extract relevant attributes.

    Parameters
    ----------
    filename : str
        Path to the header file.

    Returns
    -------
    dict
        Dictionary of extracted attributes.
    """
    attributes = {
        "matrix_sizes": {},
        "scaling_factors": {},
    }

    # Define generic patterns: each tuple contains (compiled regex, converter, attribute key)
    patterns = [
        (
            re.compile(r"!imaging modality\s*:=\s*(.+)", re.IGNORECASE),
            lambda m: m.group(1).strip(),
            "modality",
        ),
        (
            re.compile(r"!type of data\s*:=\s*(.+)", re.IGNORECASE),
            lambda m: m.group(1).strip(),
            "data_type",
        ),
        (
            re.compile(r"imagedata byte order\s*:=\s*(.+)", re.IGNORECASE),
            lambda m: m.group(1).strip(),
            "byte_order",
        ),
        (
            re.compile(r"!number format\s*:=\s*(.+)", re.IGNORECASE),
            lambda m: m.group(1).strip(),
            "number_format",
        ),
        (
            re.compile(r"!number of bytes per pixel\s*:=\s*(\d+)", re.IGNORECASE),
            lambda m: int(m.group(1)),
            "bytes_per_pixel",
        ),
        (
            re.compile(r"calibration factor\s*:=\s*(.+)", re.IGNORECASE),
            lambda m: float(m.group(1)),
            "calibration_factor",
        ),
        (
            re.compile(r"isotope name\s*:=\s*(.+)", re.IGNORECASE),
            lambda m: m.group(1).strip(),
            "isotope_name",
        ),
        (
            re.compile(r"number of dimensions\s*:=\s*(\d+)", re.IGNORECASE),
            lambda m: int(m.group(1)),
            "number_of_dimensions",
        ),
        (
            re.compile(r"!number of projections\s*:=\s*(\d+)", re.IGNORECASE),
            lambda m: int(m.group(1)),
            "number_of_projections",
        ),
        (
            re.compile(r"number of time frames\s*:=\s*(\d+)", re.IGNORECASE),
            lambda m: int(m.group(1)),
            "number_of_time_frames",
        ),
        (
            re.compile(
                r"!image duration\s*\(sec\)[\[\w\s]*\]\s*:=\s*(\d+)", re.IGNORECASE
            ),
            lambda m: int(m.group(1)),
            "image_duration",
        ),
        (
            re.compile(r"!extent of rotation\s*:=\s*(.+)", re.IGNORECASE),
            lambda m: float(m.group(1)),
            "extent_of_rotation",
        ),
        (
            re.compile(r"!direction of rotation\s*:=\s*(.+)", re.IGNORECASE),
            lambda m: m.group(1).strip(),
            "direction_of_rotation",
        ),
        (
            re.compile(r"start angle\s*:=\s*(.+)", re.IGNORECASE),
            lambda m: float(m.group(1)),
            "start_angle",
        ),
        (
            re.compile(r"!name of data file\s*:=\s*(.+)", re.IGNORECASE),
            lambda m: m.group(1).strip(),
            "data_file",
        ),
        (
            re.compile(r"energy window lower level\[\d+\]\s*:=\s*(.+)", re.IGNORECASE),
            lambda m: float(m.group(1)),
            "energy_window_lower",
        ),
        (
            re.compile(r"energy window upper level\[\d+\]\s*:=\s*(.+)", re.IGNORECASE),
            lambda m: float(m.group(1)),
            "energy_window_upper",
        ),
    ]

    with open(filename, "r") as file:
        for line in file:
            # Process matrix sizes: "!matrix size [axis] := <int>"
            ms_match = re.search(
                r"!matrix size\s*\[(.+?)\]\s*:=\s*(\d+)", line, re.IGNORECASE
            )
            if ms_match:
                axis = ms_match.group(1).strip()
                attributes["matrix_sizes"][axis] = int(ms_match.group(2))
                continue

            # Process scaling factors: "!scaling factor (mm/pixel) [axis] := <float>"
            sf_match = re.search(
                r"!scaling factor\s*\(mm/pixel\)\s*\[(.+?)\]\s*:=\s*(.+)",
                line,
                re.IGNORECASE,
            )
            if sf_match:
                axis = sf_match.group(1).strip()
                attributes["scaling_factors"][axis] = float(sf_match.group(2).strip())
                continue

            # Process orbit/radius information.
            if re.search(r"(radius|radii)\s*:=\s*(.+)", line, re.IGNORECASE):
                r_match = re.search(r"(radius|radii)\s*:=\s*(.+)", line, re.IGNORECASE)
                tmp = r_match.group(2).strip()
                if tmp.startswith("{") and tmp.endswith("}") or "," in tmp:
                    # Remove braces and split by comma.
                    tmp = tmp.strip("{}")
                    values = [float(v.strip()) for v in tmp.split(",")]
                    mean_value = np.mean(values)
                    std_of_mean_value = np.std(values) / mean_value
                    # If the radii vary, flag non-circular orbit.
                    if std_of_mean_value > 1e-6:
                        attributes["orbit"] = "non-circular"
                        attributes["radii"] = values
                        attributes["height_to_detector_surface"] = mean_value
                        warnings.warn(
                            "Non-circular orbit detected. Handle this case manually.",
                            UserWarning,
                        )
                    else:
                        attributes["orbit"] = "Circular"
                    attributes["height_to_detector_surface"] = mean_value
                else:
                    attributes["orbit"] = "Circular"
                    attributes["height_to_detector_surface"] = float(tmp)
                continue

            # Process generic orbit specification if not already set.
            orbit_match = re.search(r"orbit\s*:=\s*(.+)", line, re.IGNORECASE)
            if orbit_match:
                attributes["orbit"] = orbit_match.group(1).strip()
                continue

            # Try generic patterns.
            for pattern, converter, attr_key in patterns:
                m = pattern.search(line)
                if m:
                    attributes[attr_key] = converter(m)
                    break

    return harmonize_stir_attributes(attributes)


def create_stir_image(matrix_dim: list, voxel_size: list):
    """
    Creates a uniform (zeros) STIR ImageData object given specified parameters.

    Parameters:
    matrix_dim (list of int): A three element list containing the matrix size for each dimension of the image.
    voxel_size (list of float): A three element list describing the voxel size in the image (mm).

    Returns [ImageData]: The ImageData object.

    """
    img = np.zeros(matrix_dim, dtype=np.float32)

    header = {}
    header["!INTERFILE"] = ""
    header["!imaging modality"] = "nucmed"
    header["!version of keys"] = "STIR4.0"

    header["!GENERAL DATA"] = ""
    header["!name of data file"] = "temp.v"

    header["!GENERAL IMAGE DATA"] = ""
    header["!type of data"] = "Tomographic"
    header["imagedata byte order"] = "LITTLEENDIAN"

    header["!SPECT STUDY (general)"] = ""
    header["!process status"] = "reconstructed"
    header["!number format"] = "float"
    header["!number of bytes per pixel"] = "4"
    header["number of dimensions"] = str(np.size(img.shape))
    header["matrix axis label [1]"] = "x"
    header["matrix axis label [2]"] = "y"
    header["matrix axis label [3]"] = "z"
    header["!matrix size [1]"] = str(matrix_dim[2])
    header["!matrix size [2]"] = str(matrix_dim[1])
    header["!matrix size [3]"] = str(matrix_dim[0])
    header["scaling factor (mm/pixel) [1]"] = str(voxel_size[2])
    header["scaling factor (mm/pixel) [2]"] = str(voxel_size[1])
    header["scaling factor (mm/pixel) [3]"] = str(voxel_size[0])
    header["number of time frames"] = "1"

    header["!END OF INTERFILE"] = ""

    line = 0
    header_path = os.path.join("temp.hv")
    with open(header_path, "w") as f:
        for k in header.keys():
            if k.islower() or line == 0:
                tempStr = str(k) + " := " + str(header[str(k)]) + "\n"
                line += 1
            else:
                tempStr = "\n" + str(k) + " := " + str(header[str(k)]) + "\n"
            f.write(tempStr)
            # print(k, ":=", header[str(k)])

    f.close()

    raw_file_path = os.path.join("temp.v")
    img.tofile(raw_file_path)

    print("Image written to: " + header_path)

    template_image = ImageData(header_path)
    os.remove(header_path)
    os.remove(raw_file_path)

    return template_image


def create_stir_acqdata(proj_matrix: list, num_projections: int, pixel_size: list):
    """
    Creates a uniform (zeros) STIR AcquisitionData object given specified parameters.

    Parameters:
    proj_matrix (list of int): A two element list containing the matrix size for each dimension of the projections.
    num_projections (int): The number of projections in the acquisition data file.
    pixel_size (list of float): A two element list describing the pixel size in the projections (mm).

    Returns [AcquisiitonData]: The AcquisitionData object.

    """
    acq = np.zeros(
        (1, proj_matrix[0], num_projections, proj_matrix[1]), dtype=np.float32
    )

    header = {}
    header["!INTERFILE"] = ""
    header["!imaging modality"] = "NM"
    header["name of data file"] = "temp.s"
    header["!version of keys"] = "3.3"

    header["!GENERAL DATA"] = ""

    header["!GENERAL IMAGE DATA"] = ""
    header["!type of data"] = "Tomographic"
    header["imagedata byte order"] = "LITTLEENDIAN"

    header["!SPECT STUDY (General)"] = ""
    header["!number format"] = "float"
    header["!number of bytes per pixel"] = "4"
    header["!number of projections"] = str(num_projections)
    header["!extent of rotation"] = "360"
    header["process status"] = "acquired"

    header["!SPECT STUDY (acquired data)"] = ""
    header["!direction of rotation"] = "CW"
    header["start angle"] = "180"
    header["orbit"] = "Circular"
    header["Radius"] = "200"

    header["!matrix size [1]"] = str(proj_matrix[0])
    header["scaling factor (mm/pixel) [1]"] = str(pixel_size[0])
    header["!matrix size [2]"] = str(proj_matrix[1])
    header["scaling factor (mm/pixel) [2]"] = str(pixel_size[1])

    header["!END OF INTERFILE"] = ""

    header_path = os.path.join("temp.hs")
    with open(header_path, "w") as f:
        for k in header.keys():
            tempStr = str(k) + " := " + str(header[str(k)]) + "\n"
            f.write(tempStr)

    f.close()

    raw_file_path = os.path.join("temp.s")
    acq.tofile(raw_file_path)

    print("Acquisition Data written to: " + header_path)

    template_acqdata = AcquisitionData(header_path)
    os.remove(header_path)
    os.remove(raw_file_path)

    return template_acqdata


def create_simple_phantom():
    """Create a simple cylindrical phantom with a hot sphere."""
    # Create a 64x64x64 image with 4.42mm voxels
    matrix_dim = [64, 64, 64]
    voxel_size = [4.42, 4.42, 4.42]  # mm

    # Create empty image
    phantom = create_stir_image(matrix_dim, voxel_size)
    phantom_array = phantom.as_array()

    # Add cylindrical background (body)
    center = [32, 32, 32]
    radius = 10  # pixels
    height = 20  # pixels

    for z in range(center[0] - height // 2, center[0] + height // 2):
        for y in range(matrix_dim[1]):
            for x in range(matrix_dim[2]):
                if (x - center[1]) ** 2 + (y - center[2]) ** 2 <= radius**2:
                    phantom_array[z, y, x] = 10  # Background activity

    # Add hot sphere (tumor)
    sphere_center = [32, 32, 36]
    sphere_radius = 3  # pixels

    for z in range(matrix_dim[0]):
        for y in range(matrix_dim[1]):
            for x in range(matrix_dim[2]):
                if (x - sphere_center[2]) ** 2 + (y - sphere_center[1]) ** 2 + (
                    z - sphere_center[0]
                ) ** 2 <= sphere_radius**2:
                    phantom_array[z, y, x] = 40  # Hot spot activity

    phantom.fill(phantom_array)
    return phantom


def create_attenuation_map(phantom):
    """Create a simple attenuation map from the phantom."""
    # For simplicity, use uniform attenuation where phantom > 0
    mu_water_140keV = 0.15  # cm^-1 approximate for 140 keV

    attn_array = phantom.as_array().copy()
    attn_array[attn_array > 0] = mu_water_140keV

    mu_map = phantom.clone()
    mu_map.fill(attn_array)
    return mu_map

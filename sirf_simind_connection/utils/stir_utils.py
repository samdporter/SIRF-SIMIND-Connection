# This file contains a few useful functions for converting SIMIND output to
# STIR format.
### It should probably be cleaned up and object-orientedified at some point.

### Author: Sam Porter, Efstathios Varzakis

import contextlib
import os
import re
import subprocess
import tempfile

import numpy as np

from . import get_array
from .import_helpers import get_sirf_types
from .interfile_parser import parse_interfile_header, parse_interfile_line
from sirf_simind_connection.utils.backend_access import BACKEND_AVAILABLE, BACKENDS

# Conditional import for SIRF to avoid CI dependencies
ImageData, AcquisitionData, SIRF_AVAILABLE = get_sirf_types()

# Unpack interfaces needed by stir_utils
create_acquisition_data = BACKENDS.factories.create_acquisition_data
create_image_data = BACKENDS.factories.create_image_data


def _parse_interfile_text(text: str):
    """Parse interfile-formatted text and return a dictionary."""
    values = {}
    for line in text.splitlines():
        key, value = parse_interfile_line(line)
        if key is not None:
            values[key] = value
    return values


def parse_sinogram(template_sinogram):
    """Parse sinogram metadata without colliding temp files.

    Accepts either a path to an interfile header or an acquisition object that
    exposes ``get_info`` or ``write``. The parser reuses the shared
    ``parse_interfile_header`` helper for consistent behaviour.
    """
    if isinstance(template_sinogram, (str, os.PathLike)):
        return parse_interfile_header(str(template_sinogram))

    if hasattr(template_sinogram, "get_info") and callable(
        template_sinogram.get_info  # type: ignore[attr-defined]
    ):
        return _parse_interfile_text(template_sinogram.get_info())  # type: ignore[attr-defined]

    if hasattr(template_sinogram, "filename"):
        return parse_interfile_header(template_sinogram.filename)

    if hasattr(template_sinogram, "write") and callable(template_sinogram.write):
        with tempfile.NamedTemporaryFile(suffix=".hs", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            template_sinogram.write(tmp_path)
            return parse_interfile_header(tmp_path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(tmp_path)

    raise TypeError(
        "template_sinogram must be a path or acquisition object with "
        "get_info()/write() methods"
    )


def parse_interfile(filename):
    """Parse STIR interfile and return dictionary of key-value pairs.

    This function is kept for backward compatibility.
    New code should use parse_interfile_header() from interfile_parser module.

    Args:
        filename: Path to interfile file

    Returns:
        Dictionary of key-value pairs
    """
    return parse_interfile_header(filename)


def get_sirf_attenuation_from_simind(
    attn_filename, photopeak_energy=0.12, attn_type="mu"
):
    """Reads attenuation data from simind attenuation file and returns SIRF
    ImageData object

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
    if attn_filename[-3:] in ["ict", "hct"]:
        attn_filename = attn_filename[:-4]

    attn = np.fromfile(f"{attn_filename}.ict", dtype=data_type)
    image = ImageData()

    header_dict = parse_interfile(f"{attn_filename}.hct")
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


STIR_ATTRIBUTE_MAPPING = {
    "number_of_views": "number_of_projections",
    "azimuthal_angle_extent": "extent_of_rotation",
    "view_offset": "start_angle",
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
    """Apply canonical naming and derived values to raw STIR attributes."""
    harmonized_attributes = {}

    for key, value in attributes.items():
        standard_key = STIR_ATTRIBUTE_MAPPING.get(key, key)
        harmonized_attributes[standard_key] = value

    if "inner_ring_diameter" in harmonized_attributes:
        harmonized_attributes["height_to_detector_surface"] = (
            harmonized_attributes["inner_ring_diameter"] / 2
        )

    return harmonized_attributes


def extract_attributes_from_stir(header_filepath: str) -> dict:
    """Extract attributes from a STIR header file (.hs)."""
    if not isinstance(header_filepath, str):
        raise ValueError(
            f"extract_attributes_from_stir() only accepts string filepaths. "
            f"Got {type(header_filepath)}. If you have an acquisition object, "
            f"write it to a file first using .write() method."
        )

    return extract_attributes_from_stir_headerfile(header_filepath)


def extract_attributes_from_stir_headerfile(filename: str) -> dict:
    """Parse a STIR header file and extract relevant attributes."""
    attributes = {
        "matrix_sizes": {},
        "scaling_factors": {},
    }

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
            if ms_match := re.search(
                r"!matrix size\s*\[(.+?)\]\s*:=\s*(\d+)", line, re.IGNORECASE
            ):
                axis = ms_match[1].strip()
                attributes["matrix_sizes"][axis] = int(ms_match[2])
                continue

            if sf_match := re.search(
                r"!scaling factor\s*\(mm/pixel\)\s*\[(.+?)\]\s*:=\s*(.+)",
                line,
                re.IGNORECASE,
            ):
                axis = sf_match[1].strip()
                attributes["scaling_factors"][axis] = float(sf_match[2].strip())
                continue

            if re.search(r"(radius|radii)\s*:=\s*(.+)", line, re.IGNORECASE):
                r_match = re.search(r"(radius|radii)\s*:=\s*(.+)", line, re.IGNORECASE)
                tmp = r_match[2].strip()
                if tmp.startswith("{") and tmp.endswith("}") or "," in tmp:
                    tmp = tmp.strip("{}")
                    values = [float(v.strip()) for v in tmp.split(",")]
                    mean_value = float(np.mean(values))
                    std_of_mean_value = np.std(values) / mean_value
                    if std_of_mean_value > 1e-6:
                        attributes["orbit"] = "non-circular"
                        attributes["radii"] = values
                        attributes["height_to_detector_surface"] = mean_value
                    else:
                        attributes["orbit"] = "Circular"
                        attributes["height_to_detector_surface"] = mean_value
                else:
                    attributes["orbit"] = "Circular"
                    attributes["height_to_detector_surface"] = float(tmp)
                continue

            if orbit_match := re.search(r"orbit\s*:=\s*(.+)", line, re.IGNORECASE):
                attributes["orbit"] = orbit_match[1].strip()
                continue

            for pattern, converter, attr_key in patterns:
                if match := pattern.search(line):
                    attributes[attr_key] = converter(match)
                    break

    return harmonize_stir_attributes(attributes)


def create_stir_image(matrix_dim: list, voxel_size: list, backend="STIR"):
    """
    Creates a uniform (zeros) STIR ImageData object given specified parameters.

    Parameters:
    matrix_dim (list of int): A three element list containing the matrix size
        for each dimension of the image.
    voxel_size (list of float): A three element list describing the voxel size
        in the image (mm).

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
        for k in header:
            if k.islower() or line == 0:
                tempStr = f"{str(k)} := {str(header[str(k)])}" + "\n"
                line += 1
            else:
                tempStr = "\n" + str(k) + " := " + str(header[str(k)]) + "\n"
            f.write(tempStr)
            # print(k, ":=", header[str(k)])

    f.close()

    raw_file_path = os.path.join("temp.v")
    img.tofile(raw_file_path)

    print(f"Image written to: {header_path}")

    template_image = create_image_data(header_path)
    os.remove(header_path)
    os.remove(raw_file_path)

    return template_image


def create_stir_acqdata(proj_matrix: list, num_projections: int, pixel_size: list):
    """
    Creates a uniform (zeros) STIR AcquisitionData object given specified parameters.

    Parameters:
    proj_matrix (list of int): A two element list containing the matrix size for
        each dimension of the projections.
    num_projections (int): The number of projections in the acquisition data file.
    pixel_size (list of float): A two element list describing the pixel size
        in the projections (mm).

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
        for k in header:
            tempStr = f"{str(k)} := {str(header[str(k)])}" + "\n"
            f.write(tempStr)

    f.close()

    raw_file_path = os.path.join("temp.s")
    acq.tofile(raw_file_path)

    print(f"Acquisition Data written to: {header_path}")

    if BACKEND_AVAILABLE:
        # Return wrapped backend-agnostic object
        template_acqdata = create_acquisition_data(header_path)
    else:
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
    phantom_array = get_array(phantom)

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

    attn_array = get_array(phantom).copy()
    attn_array[attn_array > 0] = mu_water_140keV

    mu_map = phantom.clone()
    mu_map.fill(attn_array)
    return mu_map

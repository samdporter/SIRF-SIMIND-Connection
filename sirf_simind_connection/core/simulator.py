"""
This file contains classes for SIMIND simulations with SIRF data.
It reads in a template SMC file and runs a simulation with the given source
and mu map. The output is then converted to STIR format.

Author: Sam Porter
"""

import logging
import os
import re
import subprocess
from numbers import Number
from pathlib import Path

import numpy as np
from sirf.STIR import AcquisitionData, ImageData

from sirf_simind_connection.converters.attenuation import \
    attenuation_to_density
from sirf_simind_connection.converters.simind_to_stir import Converter
from sirf_simind_connection.utils.simind_utils import create_window_file
from sirf_simind_connection.utils.stir_utils import \
    extract_attributes_from_stir

from .config import RuntimeSwitches, SimulationConfig


class SimindSimulator:
    """Class to run SIMIND simulations with SIRF data.

    Currently only for circular orbit, but should be trivial to add non-circular orbit.
    """

    def __init__(
        self,
        template_smc_file_path,
        output_dir,
        output_prefix="output",
        source=None,
        mu_map=None,
        template_sinogram=None,
        photon_multiplier=1,
    ):
        """Initialises SimindSimulator object.

        Args:
            template_smc_file_path (str): Path to the template SMC file.
            output_dir (str): Path to the output directory.
            output_prefix (str): Prefix for the output file.
            source (str or ImageData, optional): Source image.
            mu_map (str or ImageData, optional): Mu map image.
            template_sinogram (str or AcquisitionData, optional): Template sinogram.
            photon_multiplier (int, optional): Photon multiplier.
        """
        # Ensure the input template file exists
        self.check_files_exist([template_smc_file_path])

        # Convert the template file path to a Path object and resolve it
        self.template_smc_file_path = Path(template_smc_file_path).resolve()
        self.input_dir = self.template_smc_file_path.parent

        # Define the path for the output SMC file
        self.smc_file_path = (self.input_dir / "simind.smc").resolve()

        # Ensure the output directory exists, create it if not
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.output_dir.resolve()

        # Define the path for the output file with the given prefix
        self.output_filepath = self.output_dir / output_prefix

        self.config = SimulationConfig(template_smc_file_path)
        self.runtime_switches = RuntimeSwitches()

        self.source = source
        self.mu_map = mu_map
        self.template_sinogram = template_sinogram
        self.time_per_projection = None

        self.output = None

        self.non_circular_orbit = False

        if source is not None:
            self.set_source(source)
        if mu_map is not None:
            self.set_mu_map(mu_map)
        if template_sinogram is not None:
            self.set_template_sinogram(template_sinogram)

        self.runtime_switches.set_switch("NN", photon_multiplier)

        # Set appropriate indices and flags for voxelised phantom
        self.config.set_flag(5, True)  # SPECT study
        self.config.set_value(15, -1)  # source type
        self.config.set_value(14, -1)  # phantom type
        self.config.set_flag(14, True)  # write to interfile header

        self.window_set = False

    def check_files_exist(self, filepaths):
        """Check that each file in filepaths exists."""
        for f in filepaths:
            assert os.path.exists(f), f"{f} does not exist"

    @staticmethod
    def find_parent_directory(file_path):
        """Return the parent directory of the directory containing file_path."""
        absolute_path = os.path.abspath(file_path)
        directory_name = os.path.dirname(absolute_path)
        parent_directory = os.path.dirname(directory_name)
        return parent_directory

    def set_windows(self, lower_bounds, upper_bounds, scatter_orders):
        """Set energy windows for SIMIND simulation."""
        output_filename = os.path.join(self.output_filepath)
        create_window_file(lower_bounds, upper_bounds, scatter_orders, output_filename)
        self.window_set = True

    def check_images_match(self, image0, image1):
        """Check that source and mu map have the same dimensions and voxel sizes."""
        assert (
            image0.voxel_sizes() == image1.voxel_sizes()
        ), "Source and mu map must have same voxel sizes"
        assert (
            image0.dimensions() == image1.dimensions()
        ), "Source and mu map must have same dimensions"

    def check_square_pixels_and_image(self, image):
        """Check that image has square pixels and the same x, y dimensions."""
        assert (
            image.voxel_sizes()[2] == image.voxel_sizes()[1]
        ), "Image must have square pixels"
        assert (
            image.dimensions()[1] == image.dimensions()[2]
        ), "Image must have same x,y dimensions"

    def add_index(self, index, value):
        """Add an index value to the simulation."""
        self.config.set_value(index, value)

    def add_flag(self, flag, value: bool):
        """Add a flag to the simulation."""
        self.config.set_flag(flag, value)

    def add_text_variable(self, variable, value):
        """Add a text variable to the simulation."""
        self.config.set_text_variable(variable, value)

    def add_data_file(self, data_file):
        """Add a data file to the simulation."""
        self.config.set_data_file(data_file)

    def add_comment(self, comment):
        """Add a comment to the simulation."""
        self.config.set_comment(comment)

    def add_runtime_switch(self, switch, value):
        """Add a runtime switch to the simulation."""
        self.runtime_switches.set_switch(switch, value)

    def set_source(self, source):
        if isinstance(source, str):
            self.source = ImageData(source)
        elif isinstance(source, ImageData):
            self.source = source
        else:
            raise TypeError("source must be a string or SIRF ImageData object")

        # Get dimensions and voxel sizes
        dim_z, dim_xy, vox_xy, vox_z = self.get_dimensions_and_voxel_sizes(self.source)
        # Divide voxel size by 10 to get cm
        vox_xy /= 10
        vox_z /= 10

        self.add_index(2, vox_z * dim_z / 2)
        self.add_index(3, vox_xy * dim_xy / 2)
        self.add_index(4, vox_xy * dim_xy / 2)

        self.add_index(28, vox_xy)
        self.add_index(76, dim_xy)
        self.add_index(77, dim_xy)

    def set_mu_map(self, mu_map):
        if isinstance(mu_map, str):
            self.mu_map = ImageData(mu_map)
        elif isinstance(mu_map, ImageData):
            self.mu_map = mu_map
        else:
            raise TypeError("mu_map must be a string or SIRF ImageData object")

        # Get dimensions and voxel sizes
        dim_z, dim_xy, vox_xy, vox_z = self.get_dimensions_and_voxel_sizes(self.mu_map)

        # Divide voxel size by 10 to get cm
        vox_xy /= 10
        vox_z /= 10

        self.add_index(5, vox_z * dim_z / 2)
        self.add_index(6, vox_xy * dim_xy / 2)
        self.add_index(7, vox_xy * dim_xy / 2)

        self.add_index(31, vox_xy)  # pixel size density images
        self.add_index(33, 1)  # first image density images
        self.add_index(34, dim_z)  # number density images
        self.add_index(78, dim_xy)  # matrix size density map i
        self.add_index(79, dim_xy)  # matrix size density map j

        self.runtime_switches.set_switch("PX", vox_xy)
        self.runtime_switches.set_switch("TH", vox_z)

    def set_rotation_in_stir_geometry(
        self, rotation_angle: Number, start_angle: Number, direction: str
    ):
        if direction.lower() == "ccw":
            # SIMIND needs angles to be negative for CCW rotation
            rotation_angle = -rotation_angle
            start_angle = -start_angle
        elif direction.lower() == "cw":
            rotation_angle = rotation_angle
            start_angle = start_angle
        else:
            raise ValueError("direction must be 'CW' or 'CCW'")

        # SIMIND requires a specific number for each direction and amount of rotation.
        # STIR can handle any number, but SIMIND can't.
        # STIR sinograms can contain floats for start_angle and extent_of_rotation,
        # SIMIND sinograms can only contain integers - so we need to convert.
        if np.isclose(rotation_angle, -360, rtol=0, atol=1e-1):
            rotation_switch = 0
        elif np.isclose(rotation_angle, -180, rtol=0, atol=1e-1):
            rotation_switch = 1
        elif np.isclose(rotation_angle, 360, rtol=0, atol=1e-1):
            rotation_switch = 2
        elif np.isclose(rotation_angle, 180, rtol=0, atol=1e-1):
            rotation_switch = 3
        else:
            raise ValueError("rotation_angle must be -360, -180, 180 or 360")

        # Start angles in SIMIND and STIR are opposite
        start_angle += 180
        if start_angle >= 360:
            start_angle -= 360

        return rotation_switch, start_angle

    def write_orbit_file(self, orbits, centre_of_rotation=None):
        if centre_of_rotation is None:
            # centre of rotation is middle of the image (x/y axis)
            centre_of_rotation = self.source.dimensions()[1] / 2
        with open(self.output_filepath.with_suffix(".cor"), "w") as f:
            # write line by line with tab radius tab centre of rotation tab
            for orbit in orbits:
                f.write(f"{orbit}\t{centre_of_rotation}\t\n")

    def read_orbit_file(self, orbit_file):
        orbits = []
        with open(orbit_file, "r") as f:
            for line in f:
                orbit, _ = line.split()
                orbits.append(orbit)
        return orbits

    def set_template_sinogram(self, template_sinogram):
        """Set the template sinogram for the simulation.

        This will overwrite any other settings with those found in the template sinogram.
        Settings include:
          - number of projections
          - height to detector surface
          - rotation direction
          - start angle (converted to SIMIND geometry)
        """
        print(
            "Warning: This will overwrite any other settings with those found in the "
            "template sinogram."
        )

        if isinstance(template_sinogram, AcquisitionData):
            print("Warning: Only circular orbit is supported.")

        attribute_dict = extract_attributes_from_stir(template_sinogram)

        self.add_index(29, attribute_dict["number_of_projections"])
        self.add_index(
            12, attribute_dict["height_to_detector_surface"] / 10
        )  # convert to cm
        rotation_switch, start_angle = self.set_rotation_in_stir_geometry(
            attribute_dict["extent_of_rotation"],
            attribute_dict["start_angle"],
            attribute_dict["direction_of_rotation"],
        )
        self.add_index(30, rotation_switch)
        self.add_index(41, start_angle)

        if isinstance(template_sinogram, str):
            if attribute_dict["orbit"] == "non-circular":
                self.non_circular_orbit = True
                self.write_orbit_file(attribute_dict["radii"])

        try:
            self.time_per_projection = (
                attribute_dict["image_duration"]
                / attribute_dict["number_of_projections"]
            )
        except:
            print(
                "Warning: Could not find time per projection in template sinogram. Header file will be incorrect."
            )

        if isinstance(template_sinogram, str):
            self.template_sinogram = AcquisitionData(template_sinogram)
        elif isinstance(template_sinogram, AcquisitionData):
            self.template_sinogram = template_sinogram.clone()

    @staticmethod
    def update_linux_path_strings(path):
        """Update path strings for Linux by converting slashes to backslashes."""
        return path.as_posix().replace("/", "\\")

    @staticmethod
    def reset_linux_path_strings(path):
        """Reset path strings to POSIX style and convert them back to Path objects."""
        return Path(path.replace("\\", "/"))

    def get_dimensions_and_voxel_sizes(self, image):
        """Get dimensions and voxel sizes of an image."""
        dim_z, _, dim_xy = image.dimensions()
        vox_xy, _, vox_z = image.voxel_sizes()
        return dim_z, dim_xy, vox_xy, vox_z

    def run_simulation(self):
        """
        Run SIMIND simulation.
        Currently only supports square pixels.
        """
        self.output = None
        self.files_converted = (
            False  # flag to check if output files have been converted to STIR
        )

        if not self.window_set:
            raise ValueError(
                "Energy windows must be set before running simulation\n"
                "Use set_windows method"
            )

        self.check_images_match(self.source, self.mu_map)
        self.check_square_pixels_and_image(self.source)
        self.check_square_pixels_and_image(self.mu_map)

        # SIMIND can only take short path strings
        # Therefore, we need to convert the paths to short strings
        # We do this by changing to the directory containing the files
        # and then changing back to the original directory after the simulation
        cwd = os.getcwd()
        os.chdir(self.output_dir)

        if self.config.get_flag(11):
            mu_map_arr = self.mu_map.as_array()
            mu_map_arr = (
                attenuation_to_density(
                    mu_map_arr, self.config.get_value("photon_energy"), self.input_dir
                )
                * 1000
            )
        else:
            mu_map_arr = np.zeros(self.mu_map.shape)

        mu_map_arr = mu_map_arr.astype(np.uint16)
        mu_map_arr.tofile(self.output_filepath.name + "_dns.dmi")
        self.config.set_data_file(11, self.output_filepath.name + "_dns")

        # Make dynamic range of source 0-100
        source_arr = self.source.as_array()
        source_arr /= self.source.max()
        source_arr *= 100
        source_arr = np.round(source_arr).astype(np.uint16)
        source_arr.tofile(self.output_filepath.name + "_src.smi")
        self.config.set_data_file(12, self.output_filepath.name + "_src")

        # Write SMC file
        self.config.save_file(self.output_filepath)

        command = ["simind", self.output_filepath.name, self.output_filepath.name]

        # if non-circular orbit, add the orbit file including .cor extension
        # but without directory
        if self.non_circular_orbit:
            print("Non-circular orbit detected. Adding orbit file to command")
            command.append(self.output_filepath.name + ".cor")

        # Add switches
        switches = ""
        for key, value in self.runtime_switches.switches.items():
            switches += f"/{key}:{str(value)}"
        command.append(switches)

        print(f"Running simind with command: {' '.join(command)}")
        subprocess.run(command)

        # Remove temporary files
        os.remove(self.output_filepath.name + "_dns.dmi")
        os.remove(self.output_filepath.name + "_src.smi")

        # Check if output files have been put in the output directory
        if len(os.listdir(self.output_dir)) == 0:
            print(
                "No output files found in output directory. SIMIND isn't very good at this\n"
                "Manually moving files. Sorry if this moves files you don't want moved"
            )
            for f in os.listdir(self.input_dir):
                if f.endswith(".h00") or f.endswith(".a00"):
                    os.rename(
                        os.path.join(self.input_dir, f),
                        os.path.join(self.output_dir, f),
                    )
        os.chdir(cwd)

    def get_output(self):
        """Get output files from SIMIND simulation."""
        if self.output is not None and len(self.output) > 0:
            return self.output

        output_strings = ["_air_w", "_sca_w", "_tot_w", "_pri_w"]

        if not self.files_converted:
            # get all h00 files in the output directory
            # that match the output prefix and contain one of the output strings
            h00_files = [
                f
                for f in os.listdir(self.output_dir)
                if f.endswith(".h00")
                and any(s in f for s in output_strings)
                and self.output_filepath.name in f
            ]
            # convert all h00 files to hs files
            # Mostly naming conventions between SIMIND and STIR
            for f in h00_files:
                if self.time_per_projection is not None:
                    Converter.edit_parameter(
                        os.path.join(self.output_dir, f),
                        "!image duration (sec)[1]",
                        self.config.get_value(29) * self.time_per_projection,
                    )
                # unfortunatlely, SIMIND .h00 fiels don't always get the radius of rotation right.
                # We'll need to do this manually
                if self.non_circular_orbit:
                    orbits = self.read_orbit_file(
                        os.path.join(self.output_filepath.with_suffix(".cor"))
                    )
                    orbits_string = "{" + ",".join([f"{o}" for o in orbits]) + "}"
                    Converter.add_parameter(
                        os.path.join(self.output_dir, f),
                        "Radii",
                        orbits_string,
                        59,
                    )
                else:
                    expected_radius = self.config.get_value(12) * 10
                    if Converter.read_line(f)[1] != expected_radius:
                        logging.info(
                            "Radius not set correctly in header file. Setting manually"
                        )
                        Converter.edit_parameter(
                            os.path.join(self.output_dir, f),
                            ";# Radius",
                            expected_radius,
                        )
                Converter.convert(os.path.join(self.output_dir, f))
                logging.info(f"Converted {f}")
            self.files_converted = True

        # get all hs files in the output directory
        # that match the output prefix and contain one of the output strings
        hs_files = [
            f
            for f in os.listdir(self.output_dir)
            if f.endswith(".hs")
            and any(s in f for s in output_strings)
            and self.output_filepath.name in f
        ]

        def extract_window_number(filename):
            match = re.search(r"w(\d+)\.hs$", filename)
            return int(match.group(1)) if match else float("inf")

        # sort the hs files by window number
        hs_files.sort(key=extract_window_number)

        output = {}
        for f in hs_files:
            # order output files by scatter type and window number
            f_split = f.split("_")
            scat_type = f_split[-2]
            window = f_split[-1].split(".")[0]
            file_path = os.path.join(self.output_dir, f)
            output_key = f"{scat_type}_{window}"
            output[output_key] = AcquisitionData(file_path)

        self.output = output
        return output

    def get_output_total(self, window=1):
        """Get total output file from SIMIND simulation."""
        outputs = self.get_output()
        return outputs["tot_w" + str(window)]

    def get_output_scatter(self, window=1):
        """Get scatter output file from SIMIND simulation."""
        outputs = self.get_output()
        return outputs["sca_w" + str(window)]

    def get_output_air(self, window=1):
        """Get air output file from SIMIND simulation."""
        outputs = self.get_output()
        return outputs["air_w" + str(window)]

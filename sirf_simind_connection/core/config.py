### This file contains a wrapper to access, edit and save simulation configuration files for the Simind Monte Carlo simulation software.
### It can work as a standalone to make the .smc files accessible and editable in a more user-friendly way.
### Or you can use it with the Simulator class to run the simulation with SIRF in python.

### Author: Sam Porter

import re
from pathlib import Path

import yaml


class SimulationConfig:
    """
    SimulationConfig Class with YAML import/export capabilities

    This class is designed to parse, manipulate, and save simulation configuration files. It provides easy access
    to configuration parameters, including index-based data, simulation flags, text variables, and associated data files.
    Now includes YAML export/import for better visualization and editing.

    Attributes:
        filepath (str): Path to the simulation configuration file.
        index_dict (dict): Dictionary mapping indices to parameter names for basic change data.
        flag_dict (dict): Dictionary mapping indices to simulation flags.
        data_file_dict (dict): Dictionary mapping indices to data file descriptions.
        data (list): List of basic change data values.
        flags (str): String representing simulation flags as 'T' (True) or 'F' (False).
        text_variables (dict): Dictionary of text variables.
        data_files (dict): Dictionary of associated data files.
        comment (str): Comment section from the configuration file.
    """

    def __init__(self, filepath):
        """
        Initialize the SimulationConfig instance.

        Args:
            filepath (str): Path to the simulation configuration file.
        """
        self.filepath = filepath
        self.index_dict = {
            1: "photon_energy",
            2: "source_half_length",
            3: "source_half_width",
            4: "source_half_height",
            5: "phantom_half_length",
            6: "phantom_half_width",
            7: "phantom_half_height",
            8: "crystal_half_length_radius",
            9: "crystal_thickness",
            10: "crystal_half_width",
            11: "backscattering_material_thickness",
            12: "height_to_detector_surface",
            13: "cover_thickness",
            14: "phantom_type",
            15: "source_type",
            16: "shift_source_x",
            17: "shift_source_y",
            18: "shift_source_z",
            19: "photon_direction",
            20: "upper_window_threshold",
            21: "lower_window_threshold",
            22: "energy_resolution",
            23: "intrinsic_resolution",
            24: "emitted_photons_per_decay",
            25: "source_activity",
            26: "number_photon_histories",
            27: "kev_per_channel",
            28: "pixel_size_simulated_image",
            29: "spect_no_projections",
            30: "spect_rotation",
            31: "pixel_size_density_images",
            32: "orientation_density_images",
            33: "first_image_density_images",
            34: "number_density_images",
            35: "density_limit_border",
            36: "shift_density_images_y",
            37: "shift_density_images_z",
            38: "step_size_photon_path_simulation",
            39: "shift_density_images_x",
            40: "density_threshold_soft_bone",
            41: "spect_starting_angle",
            42: "spect_orbital_rotation_fraction",
            43: "camera_offset_x",
            44: "camera_offset_y",
            45: "code_definitions_zubal_phantom",
            46: "hole_size_x",
            47: "hole_size_y",
            48: "distance_between_holes_x",
            49: "distance_between_holes_y",
            50: "shift_center_hole_x",
            51: "shift_center_hole_y",
            52: "collimator_thickness",
            53: "collimator_routine",
            54: "hole_shape",
            55: "type",
            56: "distance_collimator_detector",
            57: "unused_parameter_1",
            58: "unused_parameter_2",
            59: "random_collimator_movement",
            60: "unused_parameter_3",
            76: "matrix_size_image_i",
            77: "matrix_size_image_j",
            78: "matrix_size_density_map_i",
            79: "matrix_size_source_map_i",
            80: "energy_spectra_channels",
            81: "matrix_size_density_map_j",
            82: "matrix_size_source_map_j",
            83: "cutoff_energy_terminate_photon_history",
            84: "scoring_routine",
            85: "csv_file_content",
            91: "voltage",
            92: "mobility_life_electrons",
            93: "mobility_life_holes",
            94: "contact_pad_size",
            95: "anode_element_pitch",
            96: "exponential_decay_constant_tau",
            97: "components_hecht_formula",
            98: "energy_resolution_model",
            99: "cloud_mobility",
            100: "detector_array_size_i",
            101: "detector_array_size_j",
        }
        self.flag_dict = {
            1: "write_results_to_screen",
            2: "write_images_to_files",
            3: "write_pulse_height_distribution_to_file",
            4: "include_collimator",
            5: "simulate_spect_study",
            6: "include_characteristic_xray_emissions",
            7: "include_backscattering_material",
            8: "use_random_seed_value",
            9: "currently_not_in_use",
            10: "include_interactions_in_cover",
            11: "include_interactions_in_phantom",
            12: "include_energy_resolution_in_crystal",
            13: "include_forced_interactions_in_crystal",
            14: "write_interfile_header_files",
            15: "save_aligned_phantom_images",
        }
        self.data_file_dict = {
            1: "phantom_soft_tissue",
            2: "phantom_bone",
            3: "cover_material",
            4: "crystal_material",
            5: "image_file_phantom",
            6: "image_file_source",
            7: "backscatter_material",
            8: "energy_resolution_file",
            9: "unknown_file_1",
            10: "unknown_file_2",
            11: "unknown_file_3",
            12: "unknown_file_4",
        }

        # Create organized parameter groups for better YAML structure
        self.parameter_groups = {
            "source": [1, 2, 3, 4, 15, 16, 17, 18, 19, 24, 25, 26, 79, 82],
            "phantom": [5, 6, 7, 14, 31, 32, 33, 34, 35, 36, 37, 39, 40, 45, 78, 81],
            "detector_crystal": [
                8,
                9,
                10,
                12,
                13,
                22,
                23,
                91,
                92,
                93,
                94,
                95,
                96,
                97,
                98,
                99,
                100,
                101,
            ],
            "collimator": [46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 59],
            "energy_analysis": [20, 21, 27, 80, 83],
            "spect_imaging": [28, 29, 30, 41, 42, 43, 44, 76, 77],
            "simulation_control": [38, 55, 84, 85],
            "unused_parameters": [57, 58, 60],
        }

        self.data = None
        self.flags = None
        self.text_variables = {}
        self.data_files = {}
        self.comment = None

        print(str(filepath))

        # Detect file type and load accordingly
        if str(filepath).endswith(".yaml") or str(filepath).endswith(".yml"):
            # Initialize with default values first
            self._initialise_yaml_defaults()
            self.import_yaml(filepath)
        else:
            # Assume .smc format
            self.import_smc(filepath)

    def _initialise_yaml_defaults(self):
        """Initialize with default values for YAML loading."""
        # Set up defaults for when loading from YAML
        self.data = [0.0] * 101  # Initialize with 101 zeros
        self.flags = "F" * 15  # Initialize with 15 False flags
        self.text_variables = {i: "none" for i in range(1, 13)}
        self.data_files = {}
        self.comment = "Loaded from YAML"

    def _initialise_sms_defaults(self):
        """Initialize with default values for SMC loading."""
        self.comment = "Loaded from SMC"

    def import_smc(self, filepath):
        """
        Parse the simulation configuration file and populate attributes.
        """
        with open(filepath, "r") as file:
            lines = file.readlines()
            self.comment = lines[1].strip()

            # Parsing Basic Change data
            data_lines = lines[3:27]
            data_string = " ".join(data_lines).replace("\n", "")
            self.data = [
                float(val) for val in re.findall(r"-?\d+\.\d+E[+-]\d+", data_string)
            ]

            # Parsing Simulation flags
            self.flags = lines[28].strip().replace(" ", "")

            # Parsing Text Variables
            text_variables_start = 29
            text_variables_count = int(lines[text_variables_start].split()[0])
            text_variables_lines = lines[
                text_variables_start
                + 1 : text_variables_start
                + 1
                + text_variables_count
            ]
            self.text_variables = {
                i + 1: text_variables_lines[i].strip()
                for i in range(text_variables_count)
            }

            # Parsing Data files
            data_files_start = 38
            data_files_count = int(lines[data_files_start].split()[0])
            data_files_lines = lines[
                data_files_start + 1 : data_files_start + 1 + data_files_count
            ]
            self.data_files = {
                i + 1: data_files_lines[i].strip()
                for i in range(data_files_count)  # Start from 1, not 7
            }

    def to_yaml_dict(self):
        """
        Convert the configuration to a structured dictionary suitable for YAML export.

        Returns:
            dict: Organized configuration data
        """
        yaml_dict = {
            "metadata": {"comment": self.comment, "source_file": str(self.filepath)},
            "parameters": {},
            "simulation_flags": {},
            "text_variables": dict(self.text_variables),
            "data_files": {},
        }

        # Organize parameters by groups
        for group_name, indices in self.parameter_groups.items():
            yaml_dict["parameters"][group_name] = {}
            for idx in indices:
                if idx in self.index_dict and idx <= len(self.data):
                    param_name = self.index_dict[idx]
                    value = self.data[idx - 1]
                    yaml_dict["parameters"][group_name][param_name] = {
                        "index": idx,
                        "value": float(value),
                        "description": self._get_parameter_description(param_name),
                    }

        # Add simulation flags
        for idx, flag_name in self.flag_dict.items():
            if idx <= len(self.flags):
                yaml_dict["simulation_flags"][flag_name] = {
                    "index": idx,
                    "enabled": self.flags[idx - 1] == "T",
                }

        # Add data files with descriptions
        for idx, file_desc in self.data_file_dict.items():
            if idx in self.data_files:
                yaml_dict["data_files"][file_desc] = {
                    "index": idx,
                    "filepath": self.data_files[idx],
                }

        return yaml_dict

    def _get_parameter_description(self, param_name):
        """
        Get a human-readable description for parameters.

        WARNING: Parameter descriptions are based on research of SIMIND documentation
        and may vary between SIMIND versions. SIMIND 7.0+ introduced significant
        parameter reorganization. Always verify against your specific SIMIND version's
        official manual for accurate parameter definitions and valid ranges.

        For definitive parameter specifications, consult:
        - Official SIMIND manual at simind.blogg.lu.se
        - "The SIMIND Monte Carlo Program" chapter in Monte Carlo Calculations
          in Nuclear Medicine (CRC Press, 2012)
        """
        descriptions = {
            # Source parameters
            "photon_energy": "Photon energy in keV (e.g., 140 for 99mTc, 208 for 177Lu)",
            "source_half_length": "Source half-length in cm",
            "source_half_width": "Source half-width in cm",
            "source_half_height": "Source half-height in cm",
            "source_type": "Source type code (0=sphere, 1=cylinder, etc.) - check SIMIND manual",
            "shift_source_x": "Shift of source in x-direction (cm)",
            "shift_source_y": "Shift of source in y-direction (cm)",
            "shift_source_z": "Shift of source in z-direction (cm)",
            "photon_direction": "Photon direction code (2=isotropic typical)",
            "emitted_photons_per_decay": "Number of photons emitted per decay",
            "source_activity": "Source activity in MBq",
            "number_photon_histories": "Number of photon histories to simulate (10^6 typical minimum)",
            "matrix_size_source_map_i": "Matrix size for source map (i-direction) - 128x128 standard",
            "matrix_size_source_map_j": "Matrix size for source map (j-direction) - 128x128 standard",
            # Phantom parameters
            "phantom_half_length": "Phantom half-length in cm",
            "phantom_half_width": "Phantom half-width in cm",
            "phantom_half_height": "Phantom half-height in cm",
            "phantom_type": "Phantom type code",
            "pixel_size_density_images": "Pixel size for density images (cm)",
            "orientation_density_images": "Orientation of density images",
            "first_image_density_images": "First image number for density images",
            "number_density_images": "Number of density images",
            "density_limit_border": "Density limit at border",
            "shift_density_images_x": "Shift of density images in x-direction (cm)",
            "shift_density_images_y": "Shift of density images in y-direction (cm)",
            "shift_density_images_z": "Shift of density images in z-direction (cm)",
            "density_threshold_soft_bone": "Density threshold for soft bone",
            "code_definitions_zubal_phantom": "Code definitions for Zubal phantom",
            "matrix_size_density_map_i": "Matrix size for density map (i-direction)",
            "matrix_size_density_map_j": "Matrix size for density map (j-direction)",
            # Detector/Crystal parameters
            "crystal_half_length_radius": "Crystal half-length/radius in cm (circular detectors use radius)",
            "crystal_thickness": "Crystal thickness in cm (NaI(Tl) typical: 0.95cm)",
            "crystal_half_width": "Crystal half-width in cm (for rectangular crystals)",
            "height_to_detector_surface": "Height from collimator to detector surface (cm)",
            "cover_thickness": "Cover thickness in cm (typically Al or Be window)",
            "energy_resolution": "Energy resolution FWHM (%) at reference energy (9-12% typical at 140keV)",
            "intrinsic_resolution": "Intrinsic spatial resolution FWHM (cm) (3-4mm typical)",
            "voltage": "Applied voltage (V) for semiconductor detectors",
            "mobility_life_electrons": "Mobility-life product for electrons (semiconductor detectors)",
            "mobility_life_holes": "Mobility-life product for holes (semiconductor detectors)",
            "contact_pad_size": "Contact pad size (cm) for pixelated detectors",
            "anode_element_pitch": "Anode element pitch (cm) for pixelated detectors",
            "exponential_decay_constant_tau": "Exponential decay constant tau for charge collection",
            "components_hecht_formula": "Components for Hecht formula (charge collection efficiency)",
            "energy_resolution_model": "Energy resolution model code (check SIMIND manual for options)",
            "cloud_mobility": "Cloud mobility parameter for charge collection",
            "detector_array_size_i": "Detector array size (i-direction) for pixelated systems",
            "detector_array_size_j": "Detector array size (j-direction) for pixelated systems",
            # Collimator parameters
            "hole_size_x": "Collimator hole diameter (cm) - LEHR: 0.111cm, HEGP: 0.24cm",
            "hole_size_y": "Collimator hole diameter (cm) - should match hole_size_x for round holes",
            "distance_between_holes_x": "Distance between hole centers (cm) - LEHR: 0.16cm septal thickness",
            "distance_between_holes_y": "Distance between hole centers (cm) - hexagonal pattern spacing",
            "shift_center_hole_x": "Shift of center hole in x-direction (cm) for alignment",
            "shift_center_hole_y": "Shift of center hole in y-direction (cm) for alignment",
            "collimator_thickness": "Collimator thickness (cm) - LEHR: 2.405cm, HEGP: 5.9cm",
            "collimator_routine": "Collimator routine code (0=no collimator, 1=parallel holes, etc.)",
            "hole_shape": "Hole shape code (0=round, 1=square, 2=hexagonal)",
            "distance_collimator_detector": "Distance from collimator face to detector surface (cm)",
            "random_collimator_movement": "Random collimator movement parameter (for manufacturing variations)",
            # Energy analysis parameters
            "upper_window_threshold": "Upper energy window threshold (keV) - set to -100 for automatic",
            "lower_window_threshold": "Lower energy window threshold (keV) - set to -100 for automatic",
            "kev_per_channel": "keV per channel for energy spectrum binning",
            "energy_spectra_channels": "Number of energy spectra channels (512 typical)",
            "cutoff_energy_terminate_photon_history": "Cutoff energy to terminate photon history (keV)",
            # SPECT imaging parameters
            "pixel_size_simulated_image": "Pixel size for simulated images (cm) - affects resolution vs FOV",
            "spect_no_projections": "Number of SPECT projections (64, 120, 128 typical)",
            "spect_rotation": "SPECT rotation parameter (2=360° typical)",
            "spect_starting_angle": "SPECT starting angle (degrees) - 0° = anterior",
            "spect_orbital_rotation_fraction": "SPECT orbital rotation fraction (1.0 = full orbit)",
            "camera_offset_x": "Camera offset in x-direction (cm) from rotation center",
            "camera_offset_y": "Camera offset in y-direction (cm) from rotation center",
            "matrix_size_image_i": "Matrix size for images (i-direction) - 128x128 standard",
            "matrix_size_image_j": "Matrix size for images (j-direction) - 128x128 standard",
            # Simulation control parameters
            "step_size_photon_path_simulation": "Step size for photon path simulation (cm) - smaller = more accurate",
            "type": "General type parameter - check SIMIND manual for current meaning",
            "scoring_routine": "Scoring routine code - affects output data collection",
            "csv_file_content": "CSV file content parameter - for custom data output",
            "backscattering_material_thickness": "Backscattering material thickness (cm)",
            # Unused parameters - NOTE: May be used in newer SIMIND versions
            "unused_parameter_1": "Unused parameter 1 - reserved for future use",
            "unused_parameter_2": "Unused parameter 2 - reserved for future use",
            "unused_parameter_3": "Unused parameter 3 - reserved for future use",
        }
        return descriptions.get(param_name, "No description available")

    def export_yaml(self, filepath):
        """
        Export the configuration to a YAML file.

        Args:
            filepath (str): Path for the output YAML file
        """
        yaml_dict = self.to_yaml_dict()

        filepath = Path(filepath)
        if filepath.suffix != ".yaml":
            filepath = filepath.with_suffix(".yaml")

        with open(filepath, "w") as file:
            yaml.dump(
                yaml_dict, file, default_flow_style=False, indent=2, sort_keys=False
            )

        print(f"Configuration exported to {filepath}")

    def import_yaml(self, filepath):
        """
        Import configuration from a YAML file.

        Args:
            filepath (str): Path to the input YAML file
        """
        with open(filepath, "r") as file:
            yaml_dict = yaml.safe_load(file)

        # Update comment
        if "metadata" in yaml_dict and "comment" in yaml_dict["metadata"]:
            self.comment = yaml_dict["metadata"]["comment"]

        # Update parameters
        if "parameters" in yaml_dict:
            for group_name, group_params in yaml_dict["parameters"].items():
                for param_name, param_data in group_params.items():
                    if "index" in param_data and "value" in param_data:
                        idx = param_data["index"]
                        value = param_data["value"]
                        if idx in self.index_dict and idx <= len(self.data):
                            self.data[idx - 1] = float(value)

        # Update flags
        if "simulation_flags" in yaml_dict:
            flags = list(self.flags)
            for flag_name, flag_data in yaml_dict["simulation_flags"].items():
                if "index" in flag_data and "enabled" in flag_data:
                    idx = flag_data["index"]
                    enabled = flag_data["enabled"]
                    if idx in self.flag_dict and idx <= len(flags):
                        flags[idx - 1] = "T" if enabled else "F"
            self.flags = "".join(flags)

        # Update data files
        if "data_files" in yaml_dict:
            for file_desc, file_data in yaml_dict["data_files"].items():
                if "index" in file_data and "filepath" in file_data:
                    idx = file_data["index"]
                    f = file_data["filepath"]
                    if idx in self.data_file_dict:
                        self.data_files[idx] = f

        # Update text variables
        if "text_variables" in yaml_dict:
            self.text_variables = yaml_dict["text_variables"]

        print(f"Configuration imported from {f}")

    def validate_parameters(self):
        """
        Basic parameter validation based on typical SIMIND ranges.

        NOTE: This provides basic sanity checks only. Consult official SIMIND
        documentation for complete parameter validation rules and constraints.
        """
        warnings = []

        # Energy validation
        if (
            self.get_value("photon_energy") < 10
            or self.get_value("photon_energy") > 500
        ):
            warnings.append("Photon energy outside typical range (10-500 keV)")

        # Matrix size validation
        matrix_i = self.get_value("matrix_size_image_i")
        matrix_j = self.get_value("matrix_size_image_j")
        if matrix_i != matrix_j:
            warnings.append(
                "Non-square matrix sizes may cause issues in some SIMIND versions"
            )

        # Energy window validation
        upper_window = self.get_value("upper_window_threshold")
        lower_window = self.get_value("lower_window_threshold")
        if upper_window > 0 and lower_window > 0 and lower_window >= upper_window:
            warnings.append("Lower energy window >= upper energy window")

        # Collimator validation
        if self.get_flag("include_collimator"):
            if self.get_value("collimator_thickness") <= 0:
                warnings.append("Collimator enabled but thickness <= 0")

        # Crystal validation
        if self.get_value("crystal_thickness") <= 0:
            warnings.append("Crystal thickness <= 0")

        if warnings:
            print("Parameter validation warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("Basic parameter validation passed")

        return len(warnings) == 0

    def get_simind_version_info(self):
        """
        Extract version information from comment or suggest manual verification.
        """
        print("SIMIND Version Detection:")
        print(f"Comment field: '{self.comment}'")
        print("\nIMPORTANT: Parameter meanings may vary between SIMIND versions.")
        print("Version 7.0+ introduced significant parameter reorganization.")
        print("Always verify parameters against your specific SIMIND version's manual.")
        print("Official documentation: https://simind.blogg.lu.se/")

    def print_config(self):
        """
        Print the configuration details, including comments, basic change data, flags, text variables, and data files.
        """
        print(f"Comment: {self.comment}")
        print("Basic Change data:")
        for key, val in self.index_dict.items():
            print(f"index {key}: {val}: {self.data[key - 1]}")
        print("Simulation flags:")
        for key, val in self.flag_dict.items():
            print(f"flag {key}: {val}: {self.flags[key - 1]}")
        print("Text Variables:")
        for key, val in self.text_variables.items():
            print(f"{key}: {val}")
        print("Data Files:")
        for key, val in self.data_files.items():
            print(f"{key}: {val}")

    def get_value(self, index):
        """
        Get the value of a parameter by its index or description.

        Args:
            index (int or str): Parameter index or description.

        Returns:
            float: Parameter value.
        """
        if isinstance(index, int) and index in self.index_dict:
            return self.data[index - 1]
        elif isinstance(index, str) and index in self.index_dict.values():
            for key, val in self.index_dict.items():
                if val == index:
                    return self.data[key - 1]
        else:
            raise ValueError("index must be a valid integer or string")

    def set_value(self, index, value):
        """
        Set the value of a parameter by its index or description.

        Args:
            index (int or str): Parameter index or description.
            value (float): New value for the parameter.
        """
        if isinstance(index, int) and index in self.index_dict:
            self.data[index - 1] = value
        elif isinstance(index, str) and index in self.index_dict.values():
            for key, val in self.index_dict.items():
                if val == index:
                    self.data[key - 1] = value
        else:
            raise ValueError("index must be an integer or string")

    def get_flag(self, index):
        """
        Get the value of a simulation flag by its index or description.

        Args:
            index (int or str): Flag index or description.

        Returns:
            bool: True if the flag is set, False otherwise.
        """
        if isinstance(index, int) and index in self.flag_dict:
            return self.flags[index - 1] == "T"
        elif isinstance(index, str) and index in self.flag_dict.values():
            for key, val in self.flag_dict.items():
                if val == index:
                    return self.flags[key - 1] == "T"
        else:
            raise ValueError("index must be a valid integer or string")

    def set_flag(self, index, value):
        """
        Set the value of a simulation flag by its index or description.

        Args:
            index (int or str): Flag index or description.
            value (bool): True to set the flag, False to clear it.
        """
        if isinstance(index, int) and index in self.flag_dict:
            flags = list(self.flags)
            flags[index - 1] = "T" if value else "F"
            self.flags = "".join(flags)
        elif isinstance(index, str) and index in self.flag_dict.values():
            for key, val in self.flag_dict.items():
                if val == index:
                    flags = list(self.flags)
                    flags[key - 1] = "T" if value else "F"
                    self.flags = "".join(flags)
        else:
            raise ValueError("index must be an integer or string")

    def set_data_file(self, index, filepath):
        """
        Set the path to a data file by its index.

        Args:
            index (int or str): Data file index or description.
            filepath (str): Path to the data file.
        """
        if isinstance(index, int) and index in self.data_file_dict:
            if index in self.data_files:
                self.data_files[index] = filepath
        elif isinstance(index, str) and index in self.data_file_dict.values():
            for key, val in self.data_files.items():
                if val == index:
                    self.data_files[key] = filepath
        else:
            raise ValueError("index must be an integer or string")

    def get_data_file(self, index):
        """
        Get the path to a data file by its index or description.

        Args:
            index (int or str): Data file index or description.

        Returns:
            str: Path to the data file.
        """
        if isinstance(index, int) and index in self.data_file_dict:
            return self.data_files[index]
        elif isinstance(index, str) and index in self.data_file_dict.values():
            for key, val in self.data_files.items():
                if val == index:
                    return key
        else:
            raise ValueError("index must be an integer or string")

    def get_comment(self):
        return self.comment

    def set_comment(self, comment):
        self.comment = comment

    def save_file(self, filepath):
        # Ensure filepath is a Path object
        filepath = Path(filepath)

        # Check if the file has the correct suffix, add it if missing
        if filepath.suffix != ".smc":
            filepath = filepath.with_suffix(".smc")

        with open(filepath, "w") as file:
            comment = self.comment + " " * (70 - len(self.comment))
            file.write(f"SMCV2\n{comment}\n")
            file.write("   120  # Basic Change data\n")

            for i in range(0, 120, 5):  # Force exactly 120 values
                line = ""
                for j in range(5):
                    if i + j < len(self.data):
                        val = self.data[i + j]
                    else:
                        val = 0.0  # Pad with zeros if needed

                    # Format the value in scientific notation with 5 decimal places
                    formatted_val = f"{val:.5E}"
                    if val != 0:
                        # Split the formatted value into its components: sign, digit, and exponent
                        sign = "-" if val < 0 else " "
                        parts = formatted_val.split("E")
                        digits = parts[0].replace("-", "")
                        # remove final 0
                        digits = digits[:-1]
                        exponent = int(parts[1])

                        # Ensure it starts with '0' after the sign
                        if "." in digits:
                            digits = digits.replace(".", "")

                        # Since we've moved the decimal place one position to the right, increment the exponent
                        new_exponent = exponent + 1

                        # Reconstruct the formatted value
                        formatted_val = f"{sign}0.{digits}E{new_exponent:+03d}"
                    else:
                        # If the value is 0, we don't need to format it
                        formatted_val = f" {val:.5E}"

                    # Add the formatted value to the line
                    line += f"{formatted_val}"

                # Write the formatted line to the file
                file.write(f"{line}\n")

            file.write(f"    30  # Simulation flags\n{self.flags}\n")
            file.write(f"     {len(self.text_variables)}  # Text Variables\n")
            for i in range(1, len(self.text_variables) + 1):
                file.write(f"{self.text_variables[i]}\n")
            file.write(f"    {len(self.data_files)} # Data files\n")
            for i in range(1, 13):
                filename = self.data_files.get(i, "none")
                file.write(f"{filename:<60}\n")

        return filepath.with_suffix(".smc")


class RuntimeSwitches:
    def __init__(self):
        self.standard_switch_dict = {
            "CC": "Collimator code",
            "DF": "Density file segment",
            "ES": "Energy offset",
            "FE": "Energy resolution file",
            "FZ": "Zubal file",
            "FI": "Input file",
            "FD": "Density map base name",
            "FS": "Source map base name",
            "I2": "Image files stored as 16-bit integer matrices",
            "IN": "Change simind.ini value",
            "LO": "Photon histories before printout",
            "LF": "Linear sampling of polar angle for photon direction",
            "MP": "MPI parallel run",
            "OR": "Change orientation of the density map",
            "PR": "Start simulation at projection number",
            "PU": "Shift of the source in pixel units",
            "QF": "Quit simulation if earlier result file exists",
            "RR": "Random number generator seed",
            "SC": "Maximum number of scatter orders",
            "SF": "Segment for source map",
            "TS": "Time shift for interfile header",
            "UA": "Set density equal to data buffer or 1.0",
            "WB": "Whole-body simulation of anterior and posterior views",
            "Xn": "Change cross sections",
        }

        self.image_based_switch_dict = {
            "PX": "Pixel size of the source maps",
            "DI": "General direction of the source map",
            "TH": "Slice thickness for the images",
            "SB": "Start block when reading source maps",
            "1S": "Position of the first image to be used",
            "NN": "Multiplier for scaling the number of counts",
            "IF": "Input tumour file",
        }

        self.myocardiac_switch_dict = {
            "A1": "Shift of the heart in the xy-direction",
            "A2": "Shift of the heart in the yz-direction",
            "A3": "Shift of the heart in the zx-direction",
            "L1": "Location of defect",
            "L2": "Angular size of the defect",
            "L3": "Start from Base",
            "L4": "Extent of defect in axis direction",
            "L5": "Transgression in %",
            "L6": "Activity ratio in defect",
            "M1": "Thickness of the myocardial wall",
            "M2": "Thickness of the plastic wall",
            "M3": "Total length of the chamber",
            "M4": "Total diameter of the chamber",
        }

        self.multiple_spheres_switch_dict = {
            "C1": "Number of spheres",
            "C2": "Radius of spheres",
            "C3": "Activity of spheres",
            "C4": "Shift of spheres in the x-direction",
            "C5": "Shift of spheres in the y-direction",
            "C6": "Shift of spheres in the z-direction",
        }
        self.switches = {}

        self.switch_dict = {
            "Standard": self.standard_switch_dict,
            "Image-based": self.image_based_switch_dict,
            "Myocardiac": self.myocardiac_switch_dict,
            "Multiple spheres": self.multiple_spheres_switch_dict,
        }

    @property
    def combined_switch_dict(self):
        combined_dict = {}
        for sub_dict in self.switch_dict.values():
            combined_dict.update(sub_dict)
        return combined_dict

    def _set_switch_by_switch(self, switch, value):
        if switch in self.combined_switch_dict:
            self.switches[switch] = value
        else:
            raise ValueError(f"Switch {switch} is not recognised.")

    def _set_switch_by_name(self, name, value):
        for switch, description in self.combined_switch_dict.items():
            if description == name:
                self.switches[switch] = value
                return
        raise ValueError(f"Switch {name} is not recognised.")

    def set_switch(self, identifier, value):
        if identifier in self.combined_switch_dict.values():
            self._set_switch_by_name(identifier, value)
        elif identifier in self.combined_switch_dict.keys():
            self._set_switch_by_switch(identifier, value)
        else:
            raise ValueError(f"Switch {identifier} is not recognised.")

    def print_switches(self):
        for switch, value in self.switches.items():
            description = self.combined_switch_dict[switch]
            print(f"{switch} ({description}): {value}")

    def print_available_switches(self):
        for switch_dict in self.switch_dict.values():
            print(f"Switches for {switch_dict}:")
            for switch, description in switch_dict.items():
                print(f"{switch}: {description}")

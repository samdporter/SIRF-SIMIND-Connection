import fileinput
import logging
import os
import re
import sys
from pathlib import Path

from sirf.STIR import AcquisitionData

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class Converter:
    """
    Class to convert SIMIND header files to STIR header files.
    """

    @staticmethod
    def convert_line(line, dir_switch):
        """
        Converts a single line from SIMIND to STIR format.
        This basically adds a semicolon to the beginning of irrelevant lines
        and converts relevant lines to STIR naming conventions.
        It also adjusts the direction of the angles and the start angle.
        """
        patterns = {
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
        }

        if any(pattern in line for pattern in patterns):
            return ";" + line, dir_switch

        if "Radius" in line:
            return f"Radius := {float(line.split()[-1])}", dir_switch
        elif "orbit" in line and "noncircular" in line:
            return "orbit := non-circular", dir_switch
        elif "!number format := short float" in line:
            return "!number format := float", dir_switch
        elif "image duration" in line:
            parts = line.split()
            return (
                f"number of time frames := 1\nimage duration (sec) [1] := {parts[4]}",
                dir_switch,
            )
        elif ";energy window lower level" in line:
            return f"energy window lower level[1] := {line.split()[-1]}", dir_switch
        elif ";energy window upper level" in line:
            return f"energy window upper level[1] := {line.split()[-1]}", dir_switch
        elif "CCW" in line:
            return line, -1
        elif "start angle" in line:
            angle = dir_switch * float(line.split()[3]) + 180
            return f"start angle := {angle % 360}", dir_switch
        elif "!name of data file" in line:
            file = Path(line.split()[5])
            return f"!name of data file := {file.stem + file.suffix}", dir_switch

        return line, dir_switch

    def edit_line(line, parameter, value):
        """
        Edit a parameter in a line.
        """
        if parameter in line:
            return f"{parameter} := {value}"
        return line

    def read_line(line):
        """
        Read a line and return the parameter and value.
        """
        if ":=" in line:
            key, _, value = line.partition(":=")
            return key.strip(), value.strip()
        return None, None

    @staticmethod
    def convert(filename, return_object=False):
        """
        Converts a SIMIND header file to a STIR header file.
        """
        if not filename.endswith(".h00"):
            logging.error("USAGE: script filename.h00")
            sys.exit(1)

        stirfilename = filename.replace(".h00", ".hs")
        dir_switch = 1

        with open(filename, "r") as f_in, open(stirfilename, "w") as f_out:
            for line in f_in:
                write_line, dir_switch = Converter.convert_line(
                    line.strip(), dir_switch
                )
                f_out.write(write_line + "\n")

        logging.info(f"Output written to {stirfilename}")
        return AcquisitionData(stirfilename) if return_object else None

    @staticmethod
    def edit_parameter(filename, parameter, value, return_object=False):
        """
        Edit a parameter in a header file.
        """
        if not filename.endswith((".hs", ".h00")):
            logging.error("USAGE: script filename.hs")
            sys.exit(1)

        with open(filename, "r") as f_in, open("tmp.hs", "w") as f_out:
            for line in f_in:
                f_out.write(Converter.edit_line(line.strip(), parameter, value) + "\n")

        os.remove(filename)
        os.rename("tmp.hs", filename)
        logging.info(f"Parameter {parameter} set to {value}")

        return AcquisitionData(filename) if return_object else None

    @staticmethod
    def read_parameter(filename, parameter):
        """
        Read a parameter from a header file.
        """
        if not filename.endswith((".hs", ".h00")):
            logging.error("USAGE: script filename.hs")
            sys.exit(1)

        with open(filename, "r") as f:
            for line in f:
                key, value = Converter.read_line(line.strip())
                if key == parameter:
                    return value
        return None

    @staticmethod
    def add_parameter(filename, parameter, value, line_number=0, return_object=False):
        """
        Add a parameter at a specific line number in an Interfile header file.
        """
        if not filename.endswith((".hs", ".h00")):
            logging.error("USAGE: script filename.hs or filename.h00")
            sys.exit(1)

        # first test if parameter already exists
        with open(filename, "r") as f:
            for i, line in enumerate(f):
                key, _ = Converter.read_line(line.strip())
                if key == parameter:
                    Converter.edit_parameter(filename, parameter, value)

        # temp_filename with correct extension
        temp_filename = (
            "tmp"
            + filename.endswith(".hs") * ".hs"
            + filename.endswith(".h00") * ".h00"
        )
        parameter_line = f"{parameter} := {value}\n"

        with open(filename, "r") as f_in, open(temp_filename, "w") as f_out:
            lines = f_in.readlines()

        # Modify content
        with open(temp_filename, "w") as f_out:
            for i, line in enumerate(lines):
                if i == line_number:
                    f_out.write(parameter_line)
                f_out.write(line)

            if len(lines) <= line_number:
                f_out.write(parameter_line)

        os.remove(filename)
        os.rename(temp_filename, filename)
        logging.info(f"Parameter {parameter} set to {value} at line {line_number}")

        return AcquisitionData(filename) if return_object else None

    ### The below is meant to deal with the case where SIMIND rounds values, meaning they differ from the original values.
    ### This is not currently used bacause it's crap and doesn't work. It's here for reference and possible future improvement.
    # TODO: Fix this crap
    @staticmethod
    def adjust_values(
        reference_file, file_to_adjust, threshold=None, output_adjusted_file=None
    ):
        """
        Adjust values in file_to_adjust based on reference_file, either replacing exactly or within a threshold.
        """
        if output_adjusted_file is None:
            output_adjusted_file = file_to_adjust[:-4] + "_adjusted.h00"

        if isinstance(reference_file, AcquisitionData):
            reference_file.write("tmp_ref.hs")
            reference_file = "tmp_ref.hs"

        with open(reference_file, "r") as ref_file, open(
            file_to_adjust, "r"
        ) as to_adjust_file:
            reference_lines = {}
            for line in ref_file:
                if ":=" in line:
                    key, _, value = line.partition(":=")
                    reference_lines[key.strip()] = value.strip()
            adjust_lines = to_adjust_file.readlines()

        for i, line in enumerate(adjust_lines):
            if ":=" in line:
                key, _, value = line.partition(":=")
                key, value = key.strip(), value.strip()
                if key in reference_lines:
                    try:
                        ref_value, adj_value = float(reference_lines[key]), float(value)
                        if threshold is None or abs(ref_value - adj_value) <= threshold:
                            adjust_lines[i] = f"{key} := {ref_value}\n"
                    except ValueError:
                        pass

        with open(output_adjusted_file, "w") as out:
            out.writelines(adjust_lines)

        if isinstance(reference_file, str) and "tmp_ref.hs" in reference_file:
            os.remove(reference_file)

    @staticmethod
    def replace_sinogram_values(reference_sinogram, sinogram_to_adjust):
        """
        Replace values in a sinogram file based on a reference sinogram.
        """
        ref_filename, adj_filename = "tmp_ref.hs", "tmp_adjust.hs"
        reference_sinogram.write(ref_filename)
        sinogram_to_adjust.write(adj_filename)
        result = Converter.adjust_values(ref_filename, adj_filename)
        for tmp in [ref_filename, adj_filename, "tmp_ref.s", "tmp_adjust.s"]:
            os.remove(tmp)
        return result

    @staticmethod
    def convert_sinogram_parameter(sinogram, parameter, value):
        """
        Modify a sinogram parameter in-place.
        """
        filename = "tmp.hs"
        sinogram.write(filename)
        pattern = re.compile(rf"^!?{re.escape(parameter)}\s*:=", re.IGNORECASE)

        with fileinput.input(filename, inplace=True) as file:
            for line in file:
                if pattern.match(line.strip()):
                    print(f"{parameter} := {value}")
                else:
                    print(line, end="")

        sinogram = AcquisitionData(filename)
        os.remove(filename)
        return sinogram

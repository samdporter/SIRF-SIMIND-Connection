from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pydicom

from sirf_simind_connection.utils.backend_access import BACKEND_AVAILABLE, BACKENDS

# Conditional import for SIRF types
from sirf_simind_connection.utils.import_helpers import get_sirf_types
from sirf_simind_connection.utils.io_utils import temporary_directory


_, AcquisitionData, SIRF_AVAILABLE = get_sirf_types()

# Unpack interfaces needed by builder
create_acquisition_data = BACKENDS.factories.create_acquisition_data


class STIRSPECTAcquisitionDataBuilder:
    """
    A builder class for creating a uniform STIR AcquisitionData object.

    Default header parameters for the STIR format are set during initialization but
    can be
    overridden via constructor parameters or by using the `update_header` method.
    """

    def __init__(
        self,
        header_overrides=None,
        backend: Optional[Literal["sirf", "stir"]] = None,
    ):
        if backend is not None and backend not in {"sirf", "stir"}:
            raise ValueError("backend must be one of: 'sirf', 'stir', or None")
        self.backend = backend

        # Define default header for all keys
        self.header = {
            "!INTERFILE": "",
            "!imaging modality": "NM",
            "name of data file": "temp.s",
            "!version of keys": "3.3",
            "data_offset_in_bytes": "0",
            "!GENERAL DATA": "",
            "!GENERAL IMAGE DATA": "",
            "!type of data": "Tomographic",
            "imagedata byte order": "LITTLEENDIAN",
            "!SPECT STUDY (General)": "",
            "!number format": "float",
            "!number of bytes per pixel": "4",
            "!number of projections": "1",
            "!extent of rotation": "360",
            "process status": "acquired",
            "!SPECT STUDY (acquired data)": "",
            "!direction of rotation": "CW",
            "start angle": "180",
            "Radius": "200",
            "!matrix size [1]": "128",
            "scaling factor (mm/pixel) [1]": "1",
            "!matrix size [2]": "128",
            "scaling factor (mm/pixel) [2]": "1",
        }

        # Apply header overrides if provided
        if header_overrides is not None:
            self.header.update(header_overrides)

        self.pixel_array = None

    def update_header(self, updates):
        """
        Update the header dictionary with new key-value pairs.

        Parameters:
            updates (dict): Dictionary of header key updates.
        """
        self.header.update(updates)

    def build(self, output_path: Optional[str | Path] = None):
        """
        Build and return the STIR AcquisitionData object.
        """

        # Create a zeros array for acquisition data.
        # Dimensions follow the order: [ToF (not used), axial, projections, tangential]
        matrix_size_1 = int(self.header.get("!matrix size [1]", 128))
        matrix_size_2 = int(self.header.get("!matrix size [2]", 128))
        num_projections = int(self.header.get("!number of projections", 1))
        if self.pixel_array is None:
            self.pixel_array = np.zeros(
                (1, matrix_size_1, num_projections, matrix_size_2), dtype=np.float32
            )
        else:
            self.pixel_array = np.array(self.pixel_array, dtype=np.float32)

        def _write(base_path: Path, cleanup: bool) -> AcquisitionData:
            header_path = base_path.with_suffix(".hs")
            raw_file_path = base_path.with_suffix(".s")

            self.header["name of data file"] = raw_file_path.name
            self.header["!END OF INTERFILE"] = ""

            with open(header_path, "w") as f:
                for key, value in self.header.items():
                    f.write(f"{key} := {value}\n")

            self.pixel_array.tofile(raw_file_path)

            acqdata = self._load_acquisition(str(header_path))

            flipped = np.flip(self.pixel_array, axis=-1)
            acqdata = acqdata.clone()
            acqdata.fill(flipped)
            acqdata.write(str(header_path))

            if cleanup:
                header_path.unlink(missing_ok=True)
                raw_file_path.unlink(missing_ok=True)

            return self._unwrap_native(acqdata)

        if output_path is None:
            with temporary_directory() as tmp_dir:
                return _write(Path(tmp_dir) / "spect_acq", cleanup=False)

        return _write(Path(output_path), cleanup=False)

    def _load_acquisition(self, header_path: str):
        """Load acquisition data with optional explicit backend selection."""
        if BACKEND_AVAILABLE and create_acquisition_data is not None:
            if (
                self.backend is not None
                and BACKENDS.detection.set_backend is not None
                and BACKENDS.detection.get_backend is not None
            ):
                previous_backend = None
                try:
                    previous_backend = BACKENDS.detection.get_backend()
                except Exception:
                    previous_backend = None
                BACKENDS.detection.set_backend(self.backend)
                try:
                    return create_acquisition_data(header_path)
                finally:
                    if (
                        previous_backend is not None
                        and previous_backend != self.backend
                    ):
                        try:
                            BACKENDS.detection.set_backend(previous_backend)
                        except Exception:
                            pass
            return create_acquisition_data(header_path)

        if self.backend == "stir":
            raise ImportError(
                "Requested STIR backend for acquisition loading, but backend wrappers "
                "are unavailable."
            )

        if SIRF_AVAILABLE:
            return AcquisitionData(header_path)

        raise ImportError(
            "Unable to load acquisition data: neither SIRF nor STIR Python "
            "backends are available."
        )

    @staticmethod
    def _unwrap_native(obj):
        """Return native backend object when a wrapper is provided."""
        native = getattr(obj, "native_object", None)
        return native if native is not None else obj

    def build_multi_energy(self, output_path_base="temp", multiple_data_files=True):
        """
        If multiple energy windows are available (as extracted in self.energy_windows),
        build and save separate AcquisitionData files for each energy window. At the
        moment,
        this is the only way to do this. In the future, we may consider adding an option
        to change the data offset and save data to a single file.

        Files are saved with a suffix indicating the energy window number.

        Returns:
            list: A list of AcquisitionData objects (one per energy window).
        """
        if not hasattr(self, "energy_windows") or not self.energy_windows:
            warnings.warn("No energy window information found. Using standard build().")
            acqdata = self.build(output_path=output_path_base)
            return [acqdata]

        # number of projections needs dividing by number of energy windows
        num_projections = int(self.header.get("!number of projections", 1))
        num_projections //= len(self.energy_windows)
        self.header["!number of projections"] = str(num_projections)

        # split pixel_array into energy windows along 3rd axis
        pixel_array_list = np.array_split(
            self.pixel_array, len(self.energy_windows), axis=2
        )

        acqdata_list = []
        for idx, ew in enumerate(self.energy_windows):
            # Update header for this energy window.
            self.header["energy window lower level[1]"] = ew["lower"]
            self.header["energy window upper level[1]"] = ew["upper"]
            suffix = f"_ew{idx + 1}"
            output_path = output_path_base + suffix
            self.pixel_array = pixel_array_list[idx]
            acqdata = self.build(output_path=output_path)
            acqdata_list.append(acqdata)
        return acqdata_list

    def update_header_from_dicom(self, dicom_filepath):
        """
        Update header values from a DICOM file. Extracts as many relevant values as
        possible,
        with warnings if an expected tag is missing.

        Also extracts energy window information into self.energy_windows.

        Parameters:
            dicom_filepath (str): Path to the DICOM file.
        """
        ds = pydicom.dcmread(dicom_filepath)

        # (Existing updates for modality, matrix sizes, pixel spacing, number of
        # projections, etc.)
        try:
            self.header["!imaging modality"] = ds.Modality
        except AttributeError:
            warnings.warn(
                "Modality not found in DICOM. Retaining default '!imaging modality'."
            )
        try:
            self.header["!matrix size [1]"] = str(ds.Rows)
            self.header["!matrix size [2]"] = str(ds.Columns)
        except AttributeError:
            warnings.warn(
                "Rows/Columns not found in DICOM. Retaining default matrix sizes."
            )
        try:
            pixel_spacing = ds.PixelSpacing
            self.header["scaling factor (mm/pixel) [1]"] = str(pixel_spacing[0])
            self.header["scaling factor (mm/pixel) [2]"] = str(pixel_spacing[1])
        except AttributeError:
            warnings.warn(
                "PixelSpacing not found in DICOM. Retaining default scaling factors."
            )
        try:
            num_frames = ds.get("NumberOfFrames", None)
            if num_frames is not None:
                self.header["!number of projections"] = str(num_frames)
            else:
                warnings.warn(
                    "NumberOfFrames not found in DICOM. Retaining default "
                    "'!number of projections'."
                )
        except Exception as e:
            warnings.warn("Error accessing NumberOfFrames from DICOM: " + str(e))

            # Extract energy window information
        try:
            ewi_seq_tag = (0x0054, 0x0012)
            self.energy_windows = []  # list to hold each energy window's info
            if ewi_seq_tag in ds:
                ewi_seq = ds[ewi_seq_tag].value
                for ewi_item in ewi_seq:
                    # Look for the Energy Window Range Sequence (tag 0054,0013)
                    rwr_tag = (0x0054, 0x0013)
                    if rwr_tag in ewi_item:
                        rwr_seq = ewi_item[rwr_tag].value
                        # It is common that there is one item per energy window.
                        for rwr_item in rwr_seq:
                            lower = rwr_item.get((0x0054, 0x0014), None)
                            upper = rwr_item.get((0x0054, 0x0015), None)
                            if lower is not None and upper is not None:
                                self.energy_windows.append(
                                    {
                                        "lower": str(lower.value),
                                        "upper": str(upper.value),
                                    }
                                )
                self.header["!number of energy windows"] = str(len(self.energy_windows))
                if len(self.energy_windows) == 1:
                    # For a single window, also set header keys for convenience.
                    self.header["energy window lower level[1]"] = self.energy_windows[
                        0
                    ]["lower"]
                    self.header["energy window upper level[1]"] = self.energy_windows[
                        0
                    ]["upper"]
            else:
                warnings.warn("Energy Window Information Sequence not found in DICOM.")
        except Exception as e:
            warnings.warn(
                "Error processing Energy Window Information Sequence: " + str(e)
            )

        # Rotation Information Sequence processing
        try:
            if (0x0054, 0x0052) in ds:
                rot_seq = ds[(0x0054, 0x0052)].value
                if len(rot_seq) > 0:
                    rot_item = rot_seq[0]

                    # Extract basic rotation parameters
                    if "StartAngle" in rot_item:
                        self.header["start angle"] = str(rot_item.StartAngle)
                    elif (0x0054, 0x0200) in rot_item:
                        self.header["start angle"] = str(
                            rot_item[(0x0054, 0x0200)].value
                        )

                    if (0x0018, 0x1242) in rot_item:
                        time_per_projection = str(
                            rot_item[(0x0018, 0x1242)].value / 1000
                        )

                    if num_frames is not None:
                        self.header["number of time frames"] = str(1)
                        self.header["!image duration (sec)[1]"] = str(
                            int(
                                np.round(
                                    float(time_per_projection) * float(num_frames), 0
                                )
                            )
                        )
                    else:
                        self.header["!time per projection (sec)[1]"] = (
                            time_per_projection
                        )

                    if "RotationDirection" in rot_item:
                        rd = str(rot_item.RotationDirection)
                        self.header["!direction of rotation"] = (
                            "CCW" if rd == "CC" else ("CW" if rd == "C" else rd)
                        )
                    elif (0x0018, 0x1140) in rot_item:
                        rd = str(rot_item[(0x0018, 0x1140)].value)
                        self.header["!direction of rotation"] = (
                            "CCW" if rd == "CC" else ("CW" if rd == "C" else rd)
                        )

                    if "ScanArc" in rot_item:
                        self.header["!extent of rotation"] = str(rot_item.ScanArc)
                    elif (0x0018, 0x1143) in rot_item:
                        self.header["!extent of rotation"] = str(
                            rot_item[(0x0018, 0x1143)].value
                        )

                    # Handle radial position - this is the key section
                    mean_radial_position = 0.0  # default
                    radial_processed = False

                    if (0x0018, 0x1142) in rot_item:
                        rp_val = rot_item[(0x0018, 0x1142)].value

                        # Check if rp_val is an array with multiple values
                        if (
                            hasattr(rp_val, "__iter__")
                            and not isinstance(rp_val, str)
                            and len(rp_val) > 1
                        ):
                            # Use the array values directly as the radii of rotation
                            rp_list = [float(x) for x in rp_val]

                            # Check if all radii are the same (circular orbit) using
                            # proper tolerance
                            if len(set(rp_list)) == 1 or all(
                                abs(r - rp_list[0]) < 1e-3 for r in rp_list
                            ):
                                self.header["Radius"] = str(rp_list[0])
                                self.header["orbit"] = "circular"
                                # Remove Radii key if it exists
                                self.header.pop("Radii", None)
                            else:
                                self.header["Radii"] = (
                                    "{" + ",".join(str(r) for r in rp_list) + "}"
                                )
                                self.header["orbit"] = "non-circular"
                                # Remove Radius key if it exists
                                self.header.pop("Radius", None)

                            radial_processed = True
                            print(
                                f"Debug: Direct radii processed - all same: "
                                f"{len(set(rp_list)) == 1}, values: {rp_list[:5]}..."
                            )

                        else:
                            # Single value
                            mean_radial_position = float(rp_val)
                            print(
                                f"Debug: Single radial position: {mean_radial_position}"
                            )
                    else:
                        warnings.warn(
                            "Mean radial position not found in Rotation Information "
                            "Sequence. Using default 0.0."
                        )
                        mean_radial_position = 0.0

                    # Only process tomo view offset if we haven't already processed
                    # direct radii
                    if not radial_processed:
                        print(
                            "Debug: Processing tomo view offset since direct radii "
                            "not processed"
                        )

                        # Process Detector Information Sequence & Tomo View Offset
                        det_info_seq_tag = (0x0055, 0x1022)
                        tomo_view_offset_tag = (0x0013, 0x101E)

                        if det_info_seq_tag in ds:
                            det_seq = ds[det_info_seq_tag].value
                            if len(det_seq) > 0:
                                det_item = det_seq[0]
                                if tomo_view_offset_tag in det_item:
                                    tvo = det_item[tomo_view_offset_tag].value
                                    if (
                                        hasattr(tvo, "__iter__")
                                        or isinstance(tvo, (list, tuple))
                                    ) and len(tvo) > 1:
                                        # Compute radial positions by adding tomo view
                                        # offsets to the mean radial position
                                        radial_positions = [
                                            mean_radial_position + float(tvo[i])
                                            for i in range(2, min(len(tvo), 360), 3)
                                        ]

                                        # Handle empty radial_positions list
                                        if len(radial_positions) == 0:
                                            self.header["Radius"] = str(
                                                mean_radial_position
                                            )
                                            self.header["orbit"] = "circular"
                                        elif len(radial_positions) == 1 or all(
                                            abs(r - radial_positions[0]) < 1e-3
                                            for r in radial_positions
                                        ):
                                            self.header["Radius"] = str(
                                                radial_positions[0]
                                            )
                                            self.header["orbit"] = "circular"
                                        else:
                                            self.header["Radii"] = (
                                                "{"
                                                + ",".join(
                                                    str(r) for r in radial_positions
                                                )
                                                + "}"
                                            )
                                            self.header["orbit"] = "non-circular"
                                            self.header.pop("Radius", None)
                                    else:
                                        self.header["Radius"] = str(
                                            mean_radial_position + float(tvo)
                                        )
                                        self.header["orbit"] = "circular"
                        else:
                            self.header["Radius"] = str(mean_radial_position)
                            self.header["orbit"] = "circular"
                    else:
                        print(
                            "Debug: Skipping tomo view offset processing - direct "
                            "radii already processed"
                        )

                else:
                    warnings.warn("Rotation Information Sequence is empty.")
            else:
                warnings.warn("Rotation Information Sequence not found in DICOM.")

        except Exception as e:
            warnings.warn("Error processing Rotation Information Sequence: " + str(e))

        # (Remaining updates: acquisition date & time, acquisition number,
        # manufacturer, etc.)
        try:
            self.header[";#acquisition date"] = ds.AcquisitionDate
        except AttributeError:
            warnings.warn("StudyDate not found in DICOM.")
        try:
            self.header[";#acquisition time"] = ds.AcquisitionTime
        except AttributeError:
            warnings.warn("StudyTime not found in DICOM.")
        try:
            acq_num = ds.get("AcquisitionNumber", None)
            if acq_num is not None:
                self.header[";#acquisition number"] = str(acq_num)
            else:
                warnings.warn("AcquisitionNumber not found in DICOM.")
        except Exception as e:
            warnings.warn("Error updating AcquisitionNumber: " + str(e))
        try:
            self.header[";#manufacturer"] = ds.Manufacturer
        except AttributeError:
            warnings.warn("Manufacturer not found in DICOM.")
        try:
            self.header[";#institution name"] = ds.InstitutionName
        except AttributeError:
            warnings.warn("InstitutionName not found in DICOM.")
        try:
            self.header[";#patient name"] = ds.PatientName
        except AttributeError:
            warnings.warn("PatientName not found in DICOM.")
        try:
            self.header[";#study name"] = ds.StudyDescription
        except AttributeError:
            warnings.warn("StudyDescription not found in DICOM.")
        try:
            self.pixel_array = ds.pixel_array
            print(self.pixel_array.shape)
            self.pixel_array = np.transpose(self.pixel_array, (2, 0, 1))
            # rotate the image by 90 degrees cW in axis 1
            self.pixel_array = np.rot90(self.pixel_array, 3, axes=(0, 2))
            self.pixel_array = np.expand_dims(self.pixel_array, axis=0)
        except AttributeError:
            warnings.warn("Pixel data not found in DICOM.")

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from sirf_simind_connection.utils.backend_access import BACKEND_AVAILABLE, BACKENDS
from sirf_simind_connection.utils.import_helpers import get_sirf_types
from sirf_simind_connection.utils.io_utils import temporary_directory


ImageData, _, SIRF_AVAILABLE = get_sirf_types()


class STIRSPECTImageDataBuilder:
    """
    Builder class for creating a uniform STIR ImageData object.

    Default header parameters are set during initialization but may be overridden
    via constructor parameters or using the update_header method.
    """

    def __init__(self, header_overrides=None):
        # Define default header keys.
        self.header = {
            "!INTERFILE": "",
            "!imaging modality": "nucmed",
            "!version of keys": "STIR4.0",
            "!GENERAL DATA": "",
            "!name of data file": "temp.v",
            "!GENERAL IMAGE DATA": "",
            "!type of data": "Tomographic",
            "imagedata byte order": "LITTLEENDIAN",
            "!SPECT STUDY (general)": "",
            "!process status": "reconstructed",
            "!number format": "float",
            "!number of bytes per pixel": "4",
            "number of dimensions": "3",
            "matrix axis label [1]": "x",
            "matrix axis label [2]": "y",
            "matrix axis label [3]": "z",
            "!matrix size [1]": "128",
            "!matrix size [2]": "128",
            "!matrix size [3]": "128",
            "scaling factor (mm/pixel) [1]": "1",
            "scaling factor (mm/pixel) [2]": "1",
            "scaling factor (mm/pixel) [3]": "1",
            "number of time frames": "1",
            "!END OF INTERFILE": "",
        }

        self.pixel_array: Optional[np.ndarray] = None

        # Override defaults if header_overrides is provided.
        if header_overrides is not None:
            self.header.update(header_overrides)

    def update_header(self, updates):
        """
        Update the header dictionary with new key-value pairs.

        Parameters:
            updates (dict): Dictionary of header key updates.
        """
        self.header.update(updates)

    def set_pixel_array(self, array: np.ndarray) -> None:
        """Provide image data to be written to disk."""
        self.pixel_array = np.asarray(array, dtype=np.float32)

    def _resolve_data_array(self) -> np.ndarray:
        if self.pixel_array is not None:
            return np.asarray(self.pixel_array, dtype=np.float32)

        dim_x = int(self.header["!matrix size [1]"])
        dim_y = int(self.header["!matrix size [2]"])
        dim_z = int(self.header["!matrix size [3]"])
        return np.zeros((dim_z, dim_y, dim_x), dtype=np.float32)

    def build(self, output_path: Optional[str | Path] = None):
        """
        Build and return the STIR ImageData object.

        Returns:
            ImageData: The constructed STIR ImageData object.
        """
        data = self._resolve_data_array()

        def _write(base_path: Path, cleanup: bool):
            header_path = base_path.with_suffix(".hv")
            raw_path = base_path.with_suffix(".v")
            self.header["!name of data file"] = raw_path.name

            with open(header_path, "w") as f:
                line = 0
                for key, value in self.header.items():
                    if key.islower() or line == 0:
                        temp_str = f"{key} := {value}\n"
                        line += 1
                    else:
                        temp_str = f"\n{key} := {value}\n"
                    f.write(temp_str)

            data.tofile(raw_path)

            image_data = self._load_image(str(header_path))

            if cleanup:
                header_path.unlink(missing_ok=True)
                raw_path.unlink(missing_ok=True)
            return image_data

        if output_path is None:
            with temporary_directory() as tmp_dir:
                return _write(Path(tmp_dir) / "spect_image", cleanup=False)

        return _write(Path(output_path), cleanup=False)

    @staticmethod
    def _load_image(header_path: str):
        """Load an image using whichever backend is available."""
        if BACKEND_AVAILABLE and BACKENDS.factories.create_image_data is not None:
            return BACKENDS.factories.create_image_data(header_path)

        if SIRF_AVAILABLE:
            return ImageData(header_path)

        raise ImportError(
            "Unable to load image data: neither SIRF nor STIR Python backends are available."
        )

    @staticmethod
    def create_spect_uniform_image_from_sinogram(sinogram, origin=None):
        """
        Create a uniform image for SPECT data based on the sinogram dimensions.

        Adjusts the z-direction voxel size and image dimensions to create a template
        image.

        Args:
            sinogram (AcquisitionData): The SPECT sinogram.
            origin (tuple, optional): The origin of the image. Defaults to (0, 0, 0)
                if not provided.

        Returns:
            ImageData: A uniform SPECT image initialized with the computed dimensions
                and voxel sizes.
        """
        if not SIRF_AVAILABLE:
            raise ImportError(
                "create_spect_uniform_image_from_sinogram requires the SIRF backend."
            )

        # Create a uniform image from the sinogram and adjust z-voxel size.
        image = sinogram.create_uniform_image(1)
        voxel_size = list(image.voxel_sizes())
        voxel_size[0] *= 2  # Adjust z-direction voxel size.

        # Compute new dimensions based on the uniform image.
        dims = list(image.dimensions())
        dims[0] = (
            dims[0] // 2 + dims[0] % 2
        )  # Halve the first dimension (with rounding)
        dims[1] -= dims[1] % 2  # Ensure even number for second dimension
        dims[2] = dims[1]  # Set third dimension equal to second dimension

        if origin is None:
            origin = (0, 0, 0)

        # Initialize a new image with computed dimensions, voxel sizes, and origin.
        new_image = ImageData()
        new_image.initialise(tuple(dims), tuple(voxel_size), tuple(origin))
        return new_image

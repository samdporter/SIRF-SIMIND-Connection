import os

from sirf.STIR import ImageData


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

        self.pixel_array = None

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

    def build(self):
        """
        Build and return the STIR ImageData object.

        Returns:
            ImageData: The constructed STIR ImageData object.
        """
        # Write header file with specific formatting.
        header_path = "temp.hv"
        with open(header_path, "w") as f:
            line = 0
            for key in self.header.keys():
                if key.islower() or line == 0:
                    temp_str = f"{key} := {self.header[key]}\n"
                    line += 1
                else:
                    temp_str = f"\n{key} := {self.header[key]}\n"
                f.write(temp_str)

        # Write raw image data to a temporary file.
        raw_file_path = "temp.v"
        self.img.tofile(raw_file_path)

        print("Image written to: " + header_path)

        # Create the ImageData object from the header file.
        image_data = ImageData(header_path)

        # Clean up temporary files.
        os.remove(header_path)
        os.remove(raw_file_path)

        return image_data

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

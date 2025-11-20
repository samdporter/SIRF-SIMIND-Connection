"""
Base interfaces for SIRF and STIR Python backends.

This module defines abstract base classes that provide a unified interface
for working with both SIRF and STIR Python libraries.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np


class ImageDataInterface(ABC):
    """Abstract interface for image data objects (ImageData or FloatVoxelsOnCartesianGrid)."""

    @classmethod
    @abstractmethod
    def read_from_file(cls, filepath: str) -> "ImageDataInterface":
        """Read image data from file.

        Args:
            filepath: Path to image file (.hv)

        Returns:
            ImageDataInterface: Wrapped image object
        """
        pass

    @abstractmethod
    def write(self, filepath: str) -> None:
        """Write image data to file.

        Args:
            filepath: Output file path
        """
        pass

    @abstractmethod
    def fill(self, data: Union[np.ndarray, float]) -> None:
        """Fill image with data.

        Args:
            data: Numpy array or scalar value
        """
        pass

    @abstractmethod
    def clone(self) -> "ImageDataInterface":
        """Create a deep copy of the image.

        Returns:
            ImageDataInterface: Cloned image object
        """
        pass

    @abstractmethod
    def get_uniform_copy(self, value: float) -> "ImageDataInterface":
        """Create a uniform copy with specified value.

        Args:
            value: Value to fill the image with

        Returns:
            ImageDataInterface: New uniform image
        """
        pass

    @abstractmethod
    def dimensions(self) -> Tuple[int, int, int]:
        """Get image dimensions.

        Returns:
            Tuple[int, int, int]: (z, y, x) dimensions
        """
        pass

    @abstractmethod
    def voxel_sizes(self) -> Tuple[float, float, float]:
        """Get voxel sizes in mm.

        Returns:
            Tuple[float, float, float]: (z, y, x) voxel sizes
        """
        pass

    @abstractmethod
    def as_array(self) -> np.ndarray:
        """Convert to numpy array.

        Returns:
            np.ndarray: Image data as numpy array
        """
        pass

    @abstractmethod
    def sum(self) -> float:
        """Sum all values in the image.

        Returns:
            float: Sum of all voxel values
        """
        pass

    @abstractmethod
    def max(self) -> float:
        """Return the maximum voxel value.

        Returns:
            float: Maximum voxel value
        """
        pass

    @abstractmethod
    def maximum(self, value: float) -> None:
        """Apply element-wise maximum (clipping).

        Args:
            value: Minimum value to clip to
        """
        pass

    @property
    @abstractmethod
    def native_object(self):
        """Get the underlying native object (ImageData or FloatVoxelsOnCartesianGrid)."""
        pass

    # Arithmetic operations
    @abstractmethod
    def __add__(self, other):
        """Add two images or image and scalar."""
        pass

    @abstractmethod
    def __sub__(self, other):
        """Subtract two images or image and scalar."""
        pass

    @abstractmethod
    def __mul__(self, other):
        """Multiply image by scalar or element-wise."""
        pass

    @abstractmethod
    def __truediv__(self, other):
        """Divide image by scalar or element-wise."""
        pass

    @abstractmethod
    def __radd__(self, other):
        """Right-hand addition."""
        pass

    @abstractmethod
    def __rsub__(self, other):
        """Right-hand subtraction."""
        pass

    @abstractmethod
    def __rmul__(self, other):
        """Right-hand multiplication."""
        pass

    @abstractmethod
    def __rtruediv__(self, other):
        """Right-hand division."""
        pass

    @abstractmethod
    def __neg__(self):
        """Negate image."""
        pass

    @abstractmethod
    def sapyb(
        self, a: float, y: "ImageDataInterface", b: float
    ) -> "ImageDataInterface":
        """Compute self * a + y * b (element-wise), modifying self in place.

        This method modifies self in place: self = self * a + y * b

        Args:
            a: Scalar multiplier for self
            y: Other image
            b: Scalar multiplier for y

        Returns:
            ImageDataInterface: self after modification (for chaining)
        """
        pass


class AcquisitionDataInterface(ABC):
    """Abstract interface for acquisition data (AcquisitionData or ProjData)."""

    @classmethod
    @abstractmethod
    def read_from_file(cls, filepath: str) -> "AcquisitionDataInterface":
        """Read acquisition data from file.

        Args:
            filepath: Path to header file (.hs)

        Returns:
            AcquisitionDataInterface: Wrapped acquisition object
        """
        pass

    @abstractmethod
    def write(self, filepath: str) -> None:
        """Write acquisition data to file.

        Args:
            filepath: Output file path
        """
        pass

    @abstractmethod
    def fill(self, data: Union[np.ndarray, float]) -> None:
        """Fill acquisition data.

        Args:
            data: Numpy array or scalar value
        """
        pass

    @abstractmethod
    def clone(self) -> "AcquisitionDataInterface":
        """Create a deep copy.

        Returns:
            AcquisitionDataInterface: Cloned object
        """
        pass

    @abstractmethod
    def get_uniform_copy(self, value: float) -> "AcquisitionDataInterface":
        """Create a uniform copy with specified value.

        Args:
            value: Value to fill the acquisition data with

        Returns:
            AcquisitionDataInterface: New uniform acquisition data
        """
        pass

    @abstractmethod
    def dimensions(self) -> Tuple:
        """Get acquisition data dimensions.

        Returns:
            Tuple: Dimensions (varies by format)
        """
        pass

    @property
    def shape(self) -> Tuple:
        """Get acquisition data shape (alias for dimensions()).

        Returns:
            Tuple: Dimensions (varies by format)
        """
        return self.dimensions()

    @abstractmethod
    def as_array(self) -> np.ndarray:
        """Convert to numpy array.

        Returns:
            np.ndarray: Acquisition data as numpy array
        """
        pass

    @abstractmethod
    def sum(self) -> float:
        """Sum all values.

        Returns:
            float: Sum of all counts
        """
        pass

    @abstractmethod
    def max(self) -> float:
        """Return the maximum bin value.

        Returns:
            float: Maximum bin value
        """
        pass

    @abstractmethod
    def get_info(self) -> str:
        """Get metadata information.

        Returns:
            str: Metadata string
        """
        pass

    @abstractmethod
    def get_energy_window_bounds(self) -> Tuple[float, float]:
        """Get the energy window bounds for the acquisition.

        Returns:
            Tuple[float, float]: (lower_threshold, upper_threshold) in keV
        """
        pass

    @property
    @abstractmethod
    def native_object(self):
        """Get the underlying native object (AcquisitionData or ProjData)."""
        pass

    # Arithmetic operations
    @abstractmethod
    def __add__(self, other):
        """Add two acquisition data or acquisition data and scalar."""
        pass

    @abstractmethod
    def __sub__(self, other):
        """Subtract two acquisition data or acquisition data and scalar."""
        pass

    @abstractmethod
    def __mul__(self, other):
        """Multiply acquisition data by scalar or element-wise."""
        pass

    @abstractmethod
    def __truediv__(self, other):
        """Divide acquisition data by scalar or element-wise."""
        pass

    @abstractmethod
    def __radd__(self, other):
        """Right-hand addition."""
        pass

    @abstractmethod
    def __rsub__(self, other):
        """Right-hand subtraction."""
        pass

    @abstractmethod
    def __rmul__(self, other):
        """Right-hand multiplication."""
        pass

    @abstractmethod
    def __rtruediv__(self, other):
        """Right-hand division."""
        pass

    @abstractmethod
    def __neg__(self):
        """Negate acquisition data."""
        pass

    @abstractmethod
    def sapyb(
        self, a: float, y: "AcquisitionDataInterface", b: float
    ) -> "AcquisitionDataInterface":
        """Compute self * a + y * b (element-wise), modifying self in place.

        This method modifies self in place: self = self * a + y * b

        Args:
            a: Scalar multiplier for self
            y: Other acquisition data
            b: Scalar multiplier for y

        Returns:
            AcquisitionDataInterface: self after modification (for chaining)
        """
        pass

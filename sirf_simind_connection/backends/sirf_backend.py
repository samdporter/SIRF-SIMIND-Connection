"""
SIRF backend implementation.

Provides wrappers around sirf.STIR.ImageData and sirf.STIR.AcquisitionData
that conform to the backend interface.
"""

from typing import Tuple, Union

import numpy as np

from sirf_simind_connection.utils.import_helpers import get_sirf_types

ImageData, AcquisitionData, SIRF_AVAILABLE = get_sirf_types()

from .base import AcquisitionDataInterface, ImageDataInterface


class SirfImageData(ImageDataInterface):
    """Wrapper for sirf.STIR.ImageData."""

    def __init__(self, sirf_obj: ImageData):
        """Initialize wrapper.

        Args:
            sirf_obj: SIRF ImageData object
        """
        if not SIRF_AVAILABLE:
            raise ImportError("SIRF is not available")
        self._obj = sirf_obj

    @classmethod
    def read_from_file(cls, filepath: str) -> "SirfImageData":
        """Read image from file."""
        return cls(ImageData(filepath))

    @classmethod
    def create_empty(cls) -> "SirfImageData":
        """Create empty image."""
        return cls(ImageData())

    def write(self, filepath: str) -> None:
        """Write to file."""
        self._obj.write(filepath)

    def fill(self, data: Union[np.ndarray, float, "SirfImageData"]) -> None:
        """Fill with data."""
        # Unwrap if data is a wrapped object
        if isinstance(data, SirfImageData):
            self._obj.fill(data._obj)
        else:
            self._obj.fill(data)

    def clone(self) -> "SirfImageData":
        """Clone the image."""
        return SirfImageData(self._obj.clone())

    def get_uniform_copy(self, value: float) -> "SirfImageData":
        """Create uniform copy."""
        return SirfImageData(self._obj.get_uniform_copy(value))

    def dimensions(self) -> Tuple[int, int, int]:
        """Get dimensions as (z, y, x)."""
        return self._obj.dimensions()

    def voxel_sizes(self) -> Tuple[float, float, float]:
        """Get voxel sizes as (z, y, x) in mm."""
        return self._obj.voxel_sizes()

    def as_array(self) -> np.ndarray:
        """Convert to numpy array."""
        # Try asarray first (newer SIRF), fall back to as_array
        if hasattr(self._obj, "asarray"):
            return self._obj.asarray()
        else:
            return self._obj.as_array()

    def sum(self) -> float:
        """Sum all values."""
        return float(self._obj.sum())

    def max(self) -> float:
        """Return the maximum voxel value."""
        if hasattr(self._obj, "max"):
            return float(self._obj.max())
        return float(np.max(self.as_array()))

    def maximum(self, value: float) -> None:
        """Apply element-wise maximum."""
        self._obj.maximum(value)

    @property
    def native_object(self):
        """Get underlying SIRF object."""
        return self._obj

    def __repr__(self) -> str:
        return f"SirfImageData(dims={self.dimensions()})"

    # Arithmetic operations - delegate to native SIRF object
    def __add__(self, other):
        if isinstance(other, SirfImageData):
            return SirfImageData(self._obj + other._obj)
        else:
            return SirfImageData(self._obj + other)

    def __sub__(self, other):
        if isinstance(other, SirfImageData):
            return SirfImageData(self._obj - other._obj)
        else:
            return SirfImageData(self._obj - other)

    def __mul__(self, other):
        if isinstance(other, SirfImageData):
            return SirfImageData(self._obj * other._obj)
        else:
            return SirfImageData(self._obj * other)

    def __truediv__(self, other):
        if isinstance(other, SirfImageData):
            return SirfImageData(self._obj / other._obj)
        else:
            return SirfImageData(self._obj / other)

    def __radd__(self, other):
        return SirfImageData(other + self._obj)

    def __rsub__(self, other):
        return SirfImageData(other - self._obj)

    def __rmul__(self, other):
        return SirfImageData(other * self._obj)

    def __rtruediv__(self, other):
        return SirfImageData(other / self._obj)

    def __neg__(self):
        return SirfImageData(-self._obj)

    def sapyb(self, a: float, y: "SirfImageData", b: float) -> "SirfImageData":
        """Compute self * a + y * b (element-wise), modifying self in place.

        Args:
            a: Scalar multiplier for self
            y: Other image
            b: Scalar multiplier for y

        Returns:
            SirfImageData: self after modification (for chaining)
        """
        y_obj = y._obj if isinstance(y, SirfImageData) else y
        self._obj.sapyb(a, y_obj, b)
        return self


class SirfAcquisitionData(AcquisitionDataInterface):
    """Wrapper for sirf.STIR.AcquisitionData."""

    def __init__(self, sirf_obj: AcquisitionData):
        """Initialize wrapper.

        Args:
            sirf_obj: SIRF AcquisitionData object
        """
        if not SIRF_AVAILABLE:
            raise ImportError("SIRF is not available")
        self._obj = sirf_obj

    @classmethod
    def read_from_file(cls, filepath: str) -> "SirfAcquisitionData":
        """Read acquisition data from file."""
        return cls(AcquisitionData(filepath))

    @classmethod
    def create_empty(cls) -> "SirfAcquisitionData":
        """Create empty acquisition data."""
        return cls(AcquisitionData())

    def write(self, filepath: str) -> None:
        """Write to file."""
        self._obj.write(filepath)

    def fill(self, data: Union[np.ndarray, float, "SirfAcquisitionData"]) -> None:
        """Fill with data."""
        # Unwrap if data is a wrapped object
        if isinstance(data, SirfAcquisitionData):
            self._obj.fill(data._obj)
        else:
            self._obj.fill(data)

    def clone(self) -> "SirfAcquisitionData":
        """Clone the acquisition data."""
        return SirfAcquisitionData(self._obj.clone())

    def get_uniform_copy(self, value: float) -> "SirfAcquisitionData":
        """Create uniform copy."""
        return SirfAcquisitionData(self._obj.get_uniform_copy(value))

    def dimensions(self) -> Tuple:
        """Get dimensions."""
        return self._obj.dimensions()

    def as_array(self) -> np.ndarray:
        """Convert to numpy array."""
        # Try asarray first (newer SIRF), fall back to as_array
        if hasattr(self._obj, "asarray"):
            return self._obj.asarray()
        else:
            return self._obj.as_array()

    def sum(self) -> float:
        """Sum all values."""
        return float(self._obj.sum())

    def max(self) -> float:
        """Return the maximum bin value."""
        if hasattr(self._obj, "max"):
            return float(self._obj.max())
        return float(np.max(self.as_array()))

    def get_info(self) -> str:
        """Get metadata info."""
        return self._obj.get_info()

    def get_energy_window_bounds(self) -> Tuple[float, float]:
        """Get the energy window bounds for the acquisition.

        Returns:
            Tuple[float, float]: (lower_threshold, upper_threshold) in keV

        Raises:
            NotImplementedError: SIRF backend does not currently support accessing energy window bounds
        """
        raise NotImplementedError(
            "SIRF backend does not currently support accessing energy window bounds. "
            "Use STIR backend for this functionality."
        )

    def create_uniform_image(self, value: float = 0.0) -> SirfImageData:
        """Create compatible uniform image."""
        img = self._obj.create_uniform_image(value)
        return SirfImageData(img)

    @property
    def native_object(self):
        """Get underlying SIRF object."""
        return self._obj

    def __repr__(self) -> str:
        return f"SirfAcquisitionData(dims={self.dimensions()})"

    # Arithmetic operations - delegate to native SIRF object
    def __add__(self, other):
        if isinstance(other, SirfAcquisitionData):
            return SirfAcquisitionData(self._obj + other._obj)
        else:
            return SirfAcquisitionData(self._obj + other)

    def __sub__(self, other):
        if isinstance(other, SirfAcquisitionData):
            return SirfAcquisitionData(self._obj - other._obj)
        else:
            return SirfAcquisitionData(self._obj - other)

    def __mul__(self, other):
        if isinstance(other, SirfAcquisitionData):
            return SirfAcquisitionData(self._obj * other._obj)
        else:
            return SirfAcquisitionData(self._obj * other)

    def __truediv__(self, other):
        if isinstance(other, SirfAcquisitionData):
            return SirfAcquisitionData(self._obj / other._obj)
        else:
            return SirfAcquisitionData(self._obj / other)

    def __radd__(self, other):
        return SirfAcquisitionData(other + self._obj)

    def __rsub__(self, other):
        return SirfAcquisitionData(other - self._obj)

    def __rmul__(self, other):
        return SirfAcquisitionData(other * self._obj)

    def __rtruediv__(self, other):
        return SirfAcquisitionData(other / self._obj)

    def __neg__(self):
        return SirfAcquisitionData(-self._obj)

    def sapyb(
        self, a: float, y: "SirfAcquisitionData", b: float
    ) -> "SirfAcquisitionData":
        """Compute self * a + y * b (element-wise), modifying self in place.

        Args:
            a: Scalar multiplier for self
            y: Other acquisition data
            b: Scalar multiplier for y

        Returns:
            SirfAcquisitionData: self after modification (for chaining)
        """
        y_obj = y._obj if isinstance(y, SirfAcquisitionData) else y
        self._obj.sapyb(a, y_obj, b)
        return self

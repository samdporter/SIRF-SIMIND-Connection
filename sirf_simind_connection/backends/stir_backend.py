"""
STIR Python backend implementation.

Provides wrappers around stir.FloatVoxelsOnCartesianGrid and stir.ProjData
that conform to the backend interface.
"""

from typing import Tuple, Union

import numpy as np


try:
    import stir
    import stirextra

    STIR_AVAILABLE = True
except ImportError:
    STIR_AVAILABLE = False

from .base import AcquisitionDataInterface, ImageDataInterface


class StirImageData(ImageDataInterface):
    """Wrapper for stir.FloatVoxelsOnCartesianGrid."""

    def __init__(self, stir_obj):
        """Initialize wrapper.

        Args:
            stir_obj: STIR FloatVoxelsOnCartesianGrid object
        """
        if not STIR_AVAILABLE:
            raise ImportError("STIR Python is not available")
        self._obj = stir_obj

    @classmethod
    def read_from_file(cls, filepath: str) -> "StirImageData":
        """Read image from file."""
        obj = stir.FloatVoxelsOnCartesianGrid.read_from_file(filepath)
        return cls(obj)

    @classmethod
    def create_empty(
        cls, exam_info=None, index_range=None, origin=None
    ) -> "StirImageData":
        """Create empty image.

        Args:
            exam_info: ExamInfo object (optional)
            index_range: IndexRange object (optional)
            origin: CartesianCoordinate3D for origin (optional)

        Returns:
            StirImageData: Empty image
        """
        # This is complex - STIR requires geometry info to create images
        # For now, raise NotImplementedError
        raise NotImplementedError(
            "Creating empty STIR images requires geometry information. "
            "Use read_from_file or create from template."
        )

    def write(self, filepath: str) -> None:
        """Write to file.

        Note: STIR uses write_to_file instead of write.
        """
        self._obj.write_to_file(filepath)

    def fill(self, data: Union[np.ndarray, float]) -> None:
        """Fill with data."""
        if isinstance(data, (int, float)) or not isinstance(data, np.ndarray):
            self._obj.fill(float(data))
        else:
            self._obj.fill(data.flat)

    def clone(self) -> "StirImageData":
        """Clone the image."""
        # STIR doesn't have a direct clone method
        # We need to create a new object and copy data
        cloned = self._obj.get_empty_copy()
        cloned.fill(self.as_array().flat)
        return StirImageData(cloned)

    def get_uniform_copy(self, value: float) -> "StirImageData":
        """Create uniform copy."""
        cloned = self.clone()
        cloned._obj.fill(value)
        return cloned

    def dimensions(self) -> Tuple[int, int, int]:
        """Get dimensions as (z, y, x)."""
        # Extract from numpy array shape
        arr = stirextra.to_numpy(self._obj)
        # STIR arrays are typically (z, y, x)
        return arr.shape

    def voxel_sizes(self) -> Tuple[float, float, float]:
        """Get voxel sizes as (z, y, x) in mm."""
        # Access STIR's grid spacing
        grid_spacing = self._obj.get_grid_spacing()
        # grid_spacing is a CartesianCoordinate3D with [z, y, x] ordering
        return (grid_spacing[1], grid_spacing[2], grid_spacing[3])

    def as_array(self) -> np.ndarray:
        """Convert to numpy array using stirextra."""
        return stirextra.to_numpy(self._obj)

    def sum(self) -> float:
        """Sum all values."""
        return float(self._obj.sum())

    def max(self) -> float:
        """Return the maximum voxel value."""
        return float(np.max(self.as_array()))

    def maximum(self, value: float) -> None:
        """Apply element-wise maximum.

        Note: STIR may not have this method directly.
        """
        # STIR doesn't have a direct maximum method so we need to extract array
        # and fill a clone
        arr = self.as_array()
        arr = np.maximum(arr, value)
        self.fill(arr)

    def minimum(self, value: float) -> None:
        """Apply element-wise minimum.

        Note: STIR may not have this method directly.
        """
        arr = self.as_array()
        arr = np.minimum(arr, value)
        self.fill(arr)

    @property
    def native_object(self):
        """Get underlying STIR object."""
        return self._obj

    def __repr__(self) -> str:
        return f"StirImageData(dims={self.dimensions()})"

    # Arithmetic operations - delegate to underlying STIR object
    def __add__(self, other):
        if isinstance(other, StirImageData):
            return StirImageData(self._obj + other._obj)
        else:
            return StirImageData(self._obj + float(other))

    def __sub__(self, other):
        if isinstance(other, StirImageData):
            return StirImageData(self._obj - other._obj)
        else:
            return StirImageData(self._obj - float(other))

    def __mul__(self, other):
        if isinstance(other, StirImageData):
            return StirImageData(self._obj * other._obj)
        else:
            return StirImageData(self._obj * float(other))

    def __truediv__(self, other):
        if isinstance(other, StirImageData):
            return StirImageData(self._obj / other._obj)
        else:
            return StirImageData(self._obj / float(other))

    def __radd__(self, other):
        return StirImageData(float(other) + self._obj)

    def __rsub__(self, other):
        return StirImageData(float(other) - self._obj)

    def __rmul__(self, other):
        return StirImageData(float(other) * self._obj)

    def __rtruediv__(self, other):
        return StirImageData(float(other) / self._obj)

    def __neg__(self):
        return StirImageData(-self._obj)

    def sapyb(self, a: float, y: "StirImageData", b: float) -> "StirImageData":
        """Compute self * a + y * b (element-wise), modifying self in place.

        Args:
            a: Scalar multiplier for self
            y: Other image
            b: Scalar multiplier for y

        Returns:
            StirImageData: self after modification (for chaining)
        """
        y_obj = y._obj if isinstance(y, StirImageData) else y
        self._obj.sapyb(a, y_obj, b)
        return self


class StirAcquisitionData(AcquisitionDataInterface):
    """Wrapper for stir.ProjDataInMemory.

    Always uses ProjDataInMemory to ensure arithmetic operations are available.
    """

    def __init__(self, stir_obj):
        """Initialize wrapper.

        Args:
            stir_obj: STIR ProjData or ProjDataInMemory object
        """
        if not STIR_AVAILABLE:
            raise ImportError("STIR Python is not available")

        # Always convert to ProjDataInMemory for arithmetic support
        if isinstance(stir_obj, stir.ProjDataInMemory):
            self._obj = stir_obj
        else:
            self._obj = stir.ProjDataInMemory(stir_obj)

    @classmethod
    def read_from_file(cls, filepath: str) -> "StirAcquisitionData":
        """Read projection data from file and convert to ProjDataInMemory."""
        obj = stir.ProjData.read_from_file(filepath)
        # Constructor will convert to ProjDataInMemory
        return cls(obj)

    @classmethod
    def create_empty(cls, exam_info=None, proj_data_info=None) -> "StirAcquisitionData":
        """Create empty projection data.

        Args:
            exam_info: ExamInfo object (required)
            proj_data_info: ProjDataInfo object (required)

        Returns:
            StirAcquisitionData: Empty projection data
        """
        if exam_info is None or proj_data_info is None:
            raise ValueError(
                "Creating empty STIR ProjData requires exam_info and proj_data_info"
            )
        obj = stir.ProjDataInMemory(exam_info, proj_data_info)
        return cls(obj)

    def write(self, filepath: str) -> None:
        """Write to file.

        Note: STIR uses write_to_file instead of write.
        """
        self._obj.write_to_file(filepath)

    def fill(self, data: Union[np.ndarray, float]) -> None:
        """Fill with data."""
        if isinstance(data, (int, float)):
            self._obj.fill(float(data))
        elif isinstance(data, np.ndarray):
            self._obj.fill(data.flat)
        else:
            try:
                self._obj.fill(data)
            except Exception as e:
                raise ValueError(
                    "Filling STIR ProjData with arbitrary data is not supported"
                ) from e

    def clone(self) -> "StirAcquisitionData":
        """Clone the projection data."""
        # ProjDataInMemory copy constructor creates a deep copy
        cloned = stir.ProjDataInMemory(self._obj)
        return StirAcquisitionData(cloned)

    def get_uniform_copy(self, value: float) -> "StirAcquisitionData":
        """Create uniform copy."""
        cloned = self.clone()
        cloned._obj.fill(value)
        return cloned

    def dimensions(self) -> Tuple:
        """Get dimensions."""
        # Extract from numpy array
        arr = stirextra.to_numpy(self._obj)
        return arr.shape

    def as_array(self) -> np.ndarray:
        """Convert to numpy array using stirextra."""
        return stirextra.to_numpy(self._obj)

    def sum(self) -> float:
        """Sum all values."""
        return float(self._obj.sum())

    def max(self) -> float:
        """Return the maximum bin value."""
        return float(np.max(self.as_array()))

    def maximum(self, value: float) -> None:
        """Apply element-wise maximum."""
        arr = self.as_array()
        arr = np.maximum(arr, value)
        self.fill(arr)

    def minimum(self, value: float) -> None:
        """Apply element-wise minimum.

        Note: STIR may not have this method directly.
        """
        arr = self.as_array()
        arr = np.minimum(arr, value)
        self.fill(arr)

    def get_info(self) -> str:
        """Get metadata info."""
        # STIR doesn't have a direct get_info() string method
        # Construct info from proj_data_info
        info = self._obj.get_proj_data_info()
        return str(info)

    @property
    def native_object(self):
        """Get underlying STIR object."""
        return self._obj

    def __repr__(self) -> str:
        return f"StirAcquisitionData(dims={self.dimensions()})"

    # Arithmetic operations - delegate to underlying ProjDataInMemory object
    def __add__(self, other):
        if isinstance(other, StirAcquisitionData):
            return StirAcquisitionData(self._obj + other._obj)
        else:
            return StirAcquisitionData(self._obj + float(other))

    def __sub__(self, other):
        if isinstance(other, StirAcquisitionData):
            return StirAcquisitionData(self._obj - other._obj)
        else:
            return StirAcquisitionData(self._obj - float(other))

    def __mul__(self, other):
        if isinstance(other, StirAcquisitionData):
            return StirAcquisitionData(self._obj * other._obj)
        else:
            return StirAcquisitionData(self._obj * float(other))

    def __truediv__(self, other):
        if isinstance(other, StirAcquisitionData):
            return StirAcquisitionData(self._obj / other._obj)
        else:
            return StirAcquisitionData(self._obj / float(other))

    def __radd__(self, other):
        return StirAcquisitionData(float(other) + self._obj)

    def __rsub__(self, other):
        return StirAcquisitionData(float(other) - self._obj)

    def __rmul__(self, other):
        return StirAcquisitionData(float(other) * self._obj)

    def __rtruediv__(self, other):
        return StirAcquisitionData(float(other) / self._obj)

    def __neg__(self):
        return StirAcquisitionData(-self._obj)

    def sapyb(
        self, a: float, y: "StirAcquisitionData", b: float
    ) -> "StirAcquisitionData":
        """Compute self * a + y * b (element-wise), modifying self in place.

        Args:
            a: Scalar multiplier for self
            y: Other acquisition data
            b: Scalar multiplier for y

        Returns:
            StirAcquisitionData: self after modification (for chaining)
        """
        y_obj = y._obj if isinstance(y, StirAcquisitionData) else y
        self._obj.sapyb(a, y_obj, b)
        return self

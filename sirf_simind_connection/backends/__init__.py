"""
Backend abstraction layer for SIRF and STIR Python.

This module provides automatic backend detection and factory functions
for creating image and acquisition data objects that work with both
SIRF and STIR Python interfaces.

Usage:
    # Automatic detection
    from sirf_simind_connection.backends import get_backend, create_image_data

    print(f"Using backend: {get_backend()}")
    img = create_image_data("phantom.hv")

    # Manual selection
    from sirf_simind_connection.backends import set_backend
    set_backend("stir")  # Force STIR Python
"""

import importlib
import logging
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from .base import AcquisitionDataInterface, ImageDataInterface


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    try:
        from sirf.STIR import AcquisitionData, ImageData
    except ImportError:  # pragma: no cover - for type checkers without SIRF
        AcquisitionData = Any  # type: ignore[assignment]
        ImageData = Any  # type: ignore[assignment]
    try:
        import stir

        ProjData = stir.ProjData  # type: ignore[attr-defined]
        FloatVoxelsOnCartesianGrid = stir.FloatVoxelsOnCartesianGrid  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover - for type checkers without STIR
        ProjData = Any  # type: ignore[assignment]
        FloatVoxelsOnCartesianGrid = Any  # type: ignore[assignment]

# Global backend state
_backend: Optional[str] = None
_backend_initialized = False


def _import_sirf_types() -> tuple[Optional[type], Optional[type]]:
    """Import SIRF image/acquisition types lazily."""
    try:
        from sirf.STIR import AcquisitionData as SirfAcquisitionData  # type: ignore
        from sirf.STIR import ImageData as SirfImageData  # type: ignore
    except ImportError:
        return None, None
    return SirfImageData, SirfAcquisitionData


def _import_stir_types() -> tuple[Optional[type], Optional[type]]:
    """Import STIR image/acquisition types lazily."""
    try:
        from stir import FloatVoxelsOnCartesianGrid as StirImageData  # type: ignore
        from stir import ProjData as StirProjData  # type: ignore
    except ImportError:
        return None, None

    return StirImageData, StirProjData


def detect_image_backend(obj: Any) -> Optional[str]:
    """Return 'sirf' or 'stir' if obj is a native image type."""
    if obj is None:
        return None

    sirf_image, _ = _import_sirf_types()
    if sirf_image is not None and isinstance(obj, sirf_image):
        return "sirf"

    stir_image, _ = _import_stir_types()
    if stir_image is not None and isinstance(obj, stir_image):
        return "stir"

    return None


def detect_acquisition_backend(obj: Any) -> Optional[str]:
    """Return 'sirf' or 'stir' if obj is a native acquisition type."""
    if obj is None:
        return None

    _, sirf_acq = _import_sirf_types()
    if sirf_acq is not None and isinstance(obj, sirf_acq):
        return "sirf"

    _, stir_proj = _import_stir_types()
    if stir_proj is not None and isinstance(obj, stir_proj):
        return "stir"

    return None


def detect_backend_from_interface(
    obj: Union[ImageDataInterface, AcquisitionDataInterface, Any],
) -> Optional[str]:
    """Detect backend by unwrapping known interface objects."""
    native = getattr(obj, "native_object", None)
    if native is None:
        return None

    backend = detect_image_backend(native)
    if backend:
        return backend

    return detect_acquisition_backend(native)


def _activate_backend(preferred: Optional[str]) -> Optional[str]:
    """Force-activate a specific backend if requested."""
    if preferred is None:
        return None
    if preferred not in ("sirf", "stir"):
        raise ValueError(
            f"Invalid backend hint: {preferred}. Expected 'sirf' or 'stir'."
        )
    set_backend(preferred)
    return preferred


def get_backend() -> str:
    """Get the current backend or auto-detect.

    Returns:
        str: Either "sirf" or "stir"

    Raises:
        ImportError: If neither SIRF nor STIR Python is available
    """
    global _backend, _backend_initialized

    if _backend is not None and _backend_initialized:
        return _backend

    # Auto-detect
    try:
        importlib.import_module("sirf.STIR")

        _backend = "sirf"
        logger.info("Auto-detected SIRF backend")
    except ImportError:
        try:
            importlib.import_module("stir")
            importlib.import_module("stirextra")

            _backend = "stir"
            logger.info("Auto-detected STIR Python backend")
        except ImportError:
            raise ImportError(
                "Neither SIRF nor STIR Python found. "
                "Please install one of: "
                "\n  - SIRF (sirf.STIR)"
                "\n  - STIR Python (stir + stirextra)"
            )

    _backend_initialized = True
    return _backend


def set_backend(backend: Literal["sirf", "stir"]) -> None:
    """Manually set the backend.

    Args:
        backend: Either "sirf" or "stir"

    Raises:
        ValueError: If backend is not valid
        ImportError: If requested backend is not available
    """
    global _backend, _backend_initialized

    if backend not in ["sirf", "stir"]:
        raise ValueError(f"Invalid backend: {backend}. Must be 'sirf' or 'stir'")

    # Verify the backend is available
    if backend == "sirf":
        try:
            importlib.import_module("sirf.STIR")
        except ImportError:
            raise ImportError("SIRF is not available")
    elif backend == "stir":
        try:
            importlib.import_module("stir")
            importlib.import_module("stirextra")
        except ImportError:
            raise ImportError("STIR Python is not available")

    _backend = backend
    _backend_initialized = True
    logger.info(f"Backend manually set to: {backend}")


def reset_backend() -> None:
    """Reset backend to allow re-detection.

    Useful for testing or switching backends dynamically.
    """
    global _backend, _backend_initialized
    _backend = None
    _backend_initialized = False


def create_image_data(
    filepath_or_object: Optional[
        Union[str, ImageDataInterface, "ImageData", "FloatVoxelsOnCartesianGrid"]
    ] = None,
) -> ImageDataInterface:
    """Create an image data object using the current backend.

    Args:
        filepath_or_object: Can be:
            - str: Path to image file to load
            - ImageDataInterface: Already wrapped object (returned as-is)
            - Native SIRF/STIR object: Will be wrapped
            - None: Creates empty object (SIRF only)

    Returns:
        ImageDataInterface: Wrapped image object

    Raises:
        NotImplementedError: If trying to create empty image with STIR backend
        TypeError: If input type is not supported
    """
    # Already wrapped?
    if isinstance(filepath_or_object, ImageDataInterface):
        return filepath_or_object

    backend_hint = detect_image_backend(filepath_or_object)
    if backend_hint is None:
        backend_hint = detect_backend_from_interface(filepath_or_object)

    backend = _activate_backend(backend_hint) or get_backend()

    if backend == "sirf":
        from .sirf_backend import SirfImageData

        try:
            from sirf.STIR import ImageData
        except ImportError:
            ImageData = type(None)

        if filepath_or_object is None:
            return SirfImageData.create_empty()
        elif isinstance(filepath_or_object, str):
            return SirfImageData.read_from_file(filepath_or_object)
        elif ImageData is not type(None) and isinstance(filepath_or_object, ImageData):
            # Wrap native SIRF object
            return SirfImageData(filepath_or_object)
        else:
            raise TypeError(
                f"Cannot create image from type {type(filepath_or_object)}. "
                f"Expected str, ImageDataInterface, or ImageData."
            )

    elif backend == "stir":
        from .stir_backend import StirImageData

        try:
            import stir

            FloatVoxelsOnCartesianGrid = stir.FloatVoxelsOnCartesianGrid
        except ImportError:
            FloatVoxelsOnCartesianGrid = type(None)

        if filepath_or_object is None:
            raise NotImplementedError(
                "Creating empty STIR images requires geometry. "
                "Please provide a filepath or use a template."
            )
        elif isinstance(filepath_or_object, str):
            return StirImageData.read_from_file(filepath_or_object)
        elif FloatVoxelsOnCartesianGrid is not type(None) and isinstance(
            filepath_or_object, FloatVoxelsOnCartesianGrid
        ):
            # Wrap native STIR object
            return StirImageData(filepath_or_object)
        else:
            raise TypeError(
                f"Cannot create image from type {type(filepath_or_object)}. "
                f"Expected str, ImageDataInterface, or FloatVoxelsOnCartesianGrid."
            )


def create_acquisition_data(
    filepath_or_object: Optional[
        Union[str, AcquisitionDataInterface, "AcquisitionData", "ProjData"]
    ] = None,
) -> AcquisitionDataInterface:
    """Create an acquisition data object using the current backend.

    Args:
        filepath_or_object: Can be:
            - str: Path to acquisition file to load
            - AcquisitionDataInterface: Already wrapped object (returned as-is)
            - Native SIRF/STIR object: Will be wrapped
            - None: Creates empty object (SIRF only)

    Returns:
        AcquisitionDataInterface: Wrapped acquisition object

    Raises:
        NotImplementedError: If trying to create empty acquisition with STIR backend
        TypeError: If input type is not supported
    """
    # Already wrapped?
    if isinstance(filepath_or_object, AcquisitionDataInterface):
        return filepath_or_object

    backend_hint = detect_acquisition_backend(filepath_or_object)
    if backend_hint is None:
        backend_hint = detect_backend_from_interface(filepath_or_object)

    backend = _activate_backend(backend_hint) or get_backend()

    if backend == "sirf":
        from .sirf_backend import SirfAcquisitionData

        try:
            from sirf.STIR import AcquisitionData
        except ImportError:
            AcquisitionData = type(None)

        if filepath_or_object is None:
            return SirfAcquisitionData.create_empty()
        elif isinstance(filepath_or_object, str):
            return SirfAcquisitionData.read_from_file(filepath_or_object)
        elif AcquisitionData is not type(None) and isinstance(
            filepath_or_object, AcquisitionData
        ):
            # Wrap native SIRF object
            return SirfAcquisitionData(filepath_or_object)
        else:
            raise TypeError(
                f"Cannot create acquisition from type {type(filepath_or_object)}. "
                f"Expected str, AcquisitionDataInterface, or AcquisitionData."
            )

    elif backend == "stir":
        from .stir_backend import StirAcquisitionData

        try:
            import stir

            ProjData = stir.ProjData
        except ImportError:
            ProjData = type(None)

        if filepath_or_object is None:
            raise NotImplementedError(
                "Creating empty STIR ProjData requires exam_info and proj_data_info. "
                "Please provide a filepath or use "
                "StirAcquisitionData.create_empty() directly."
            )
        elif isinstance(filepath_or_object, str):
            return StirAcquisitionData.read_from_file(filepath_or_object)
        elif ProjData is not type(None) and isinstance(filepath_or_object, ProjData):
            # Wrap native STIR object
            return StirAcquisitionData(filepath_or_object)
        else:
            raise TypeError(
                f"Cannot create acquisition from type {type(filepath_or_object)}. "
                f"Expected str, AcquisitionDataInterface, or ProjData."
            )


def load_image_data(
    filepath: str, backend: Optional[Literal["sirf", "stir"]] = None
) -> ImageDataInterface:
    """Load image data from file for the requested backend.

    Args:
        filepath: Path to the image header file to load.
        backend: Optional backend override. If omitted, use auto-detected backend.

    Returns:
        ImageDataInterface: Wrapped backend-specific image object.

    Raises:
        ValueError: If backend is explicitly provided but invalid.
    """
    backend_to_use = backend or get_backend()

    if backend_to_use == "sirf":
        from .sirf_backend import SirfImageData

        return SirfImageData.read_from_file(filepath)
    if backend_to_use == "stir":
        from .stir_backend import StirImageData

        return StirImageData.read_from_file(filepath)

    raise ValueError("Invalid backend. Expected 'sirf' or 'stir'.")


def load_acquisition_data(
    filepath: str, backend: Optional[Literal["sirf", "stir"]] = None
) -> AcquisitionDataInterface:
    """Load acquisition data from file for the requested backend.

    Args:
        filepath: Path to the acquisition header file to load.
        backend: Optional backend override. If omitted, use auto-detected backend.

    Returns:
        AcquisitionDataInterface: Wrapped backend-specific acquisition object.

    Raises:
        ValueError: If backend is explicitly provided but invalid.
    """
    backend_to_use = backend or get_backend()

    if backend_to_use == "sirf":
        from .sirf_backend import SirfAcquisitionData

        return SirfAcquisitionData.read_from_file(filepath)
    if backend_to_use == "stir":
        from .stir_backend import StirAcquisitionData

        return StirAcquisitionData.read_from_file(filepath)

    raise ValueError("Invalid backend. Expected 'sirf' or 'stir'.")


def is_sirf_backend() -> bool:
    """Check if SIRF backend is active.

    Returns:
        bool: True if SIRF is the active backend
    """
    return get_backend() == "sirf"


def is_stir_backend() -> bool:
    """Check if STIR Python backend is active.

    Returns:
        bool: True if STIR is the active backend
    """
    return get_backend() == "stir"


def unwrap(obj: Union[ImageDataInterface, AcquisitionDataInterface]):
    """Get the native object from a wrapped backend object.

    Args:
        obj: Wrapped image or acquisition data

    Returns:
        The underlying SIRF or STIR object
    """
    if hasattr(obj, "native_object"):
        return obj.native_object
    return obj


__all__ = [
    "get_backend",
    "set_backend",
    "reset_backend",
    "detect_image_backend",
    "detect_acquisition_backend",
    "detect_backend_from_interface",
    "create_image_data",
    "create_acquisition_data",
    "load_image_data",
    "load_acquisition_data",
    "is_sirf_backend",
    "is_stir_backend",
    "unwrap",
    "ImageDataInterface",
    "AcquisitionDataInterface",
]

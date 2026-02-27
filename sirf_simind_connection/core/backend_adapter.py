"""
Backend input adapter for consistent wrapping and backend enforcement.

This module provides the BackendInputAdapter class that consolidates all
backend detection, wrapping, and consistency enforcement logic that was
previously duplicated across set_source, set_mu_map, and set_template_sinogram.
"""

from typing import Any, Callable, Optional, Type, Union

from sirf_simind_connection.utils.backend_access import BACKEND_AVAILABLE, BACKENDS
from sirf_simind_connection.utils.sirf_stir_utils import register_and_enforce_backend


# Unpack needed interfaces
ensure_image_interface = BACKENDS.wrappers.ensure_image_interface
ensure_acquisition_interface = BACKENDS.wrappers.ensure_acquisition_interface
detect_image_backend = BACKENDS.detection.detect_image_backend
detect_acquisition_backend = BACKENDS.detection.detect_acquisition_backend
detect_backend_from_interface = BACKENDS.detection.detect_backend_from_interface
ImageDataInterface = BACKENDS.types.ImageDataInterface
AcquisitionDataInterface = BACKENDS.types.AcquisitionDataInterface


class BackendInputAdapter:
    """Handle backend detection and wrapping for simulator inputs.

    This adapter eliminates duplicate backend handling logic across set_source,
    set_mu_map, and set_template_sinogram by providing a single place to:
    1. Detect the backend from input objects
    2. Wrap inputs with the appropriate interface
    3. Enforce backend consistency (no mixing SIRF and STIR)

    Example:
        >>> adapter = BackendInputAdapter()
        >>> source = adapter.wrap_image('source.hv')  # First input sets backend
        >>> mu_map = adapter.wrap_image('mu_map.hv')  # Must match backend
        >>> template = adapter.wrap_acquisition('template.hs')  # Must match backend
    """

    def __init__(self):
        """Initialize the adapter with no backend preference."""
        self.preferred_backend: Optional[str] = None

    def wrap_image(
        self, image: Union[str, ImageDataInterface, Any]
    ) -> ImageDataInterface:
        """Detect backend, enforce consistency, and wrap image input.

        Args:
            image: Image input (path, interface, or native object)

        Returns:
            Wrapped ImageDataInterface

        Raises:
            ImportError: If backends are not available
            ValueError: If image backend conflicts with preferred_backend
        """
        return self._wrap_value(
            value=image,
            detector=detect_image_backend,
            ensure_fn=ensure_image_interface,
            interface_type=ImageDataInterface,
            missing_backend_message=(
                "SIRF/STIR backend wrappers are not available to load image data"
            ),
        )

    def wrap_acquisition(
        self, acquisition: Union[str, AcquisitionDataInterface, Any]
    ) -> AcquisitionDataInterface:
        """Detect backend, enforce consistency, and wrap acquisition input.

        Args:
            acquisition: Acquisition input (path, interface, or native object)

        Returns:
            Wrapped AcquisitionDataInterface

        Raises:
            ImportError: If backends are not available
            ValueError: If acquisition backend conflicts with preferred_backend
        """
        return self._wrap_value(
            value=acquisition,
            detector=detect_acquisition_backend,
            ensure_fn=ensure_acquisition_interface,
            interface_type=AcquisitionDataInterface,
            missing_backend_message=(
                "SIRF/STIR backend wrappers are not available to load acquisition data"
            ),
        )

    def get_preferred_backend(self) -> Optional[str]:
        """Get the currently preferred backend ('sirf', 'stir', or None).

        Returns:
            Preferred backend name or None if not yet determined
        """
        return self.preferred_backend

    def enforce_backend(self, backend_hint: Optional[str]) -> Optional[str]:
        """Force a backend preference (e.g., when callers request native outputs)."""
        if backend_hint is None:
            return self.preferred_backend

        normalized = backend_hint.lower()
        self.preferred_backend = register_and_enforce_backend(
            normalized, self.preferred_backend
        )
        return self.preferred_backend

    def _wrap_value(
        self,
        value: Union[str, Any],
        detector: Optional[Callable[[Any], Optional[str]]],
        ensure_fn: Optional[Callable[[Any, Optional[str]], Any]],
        interface_type: Type[Any],
        missing_backend_message: str,
    ) -> Any:
        """Shared helper for wrapping images/acquisitions with backend safety."""
        if not BACKEND_AVAILABLE or ensure_fn is None:
            raise ImportError(missing_backend_message)

        # Detect backend from non-string inputs before wrapping
        if not isinstance(value, str):
            backend = detector(value) if detector else None
            if backend is None and isinstance(value, interface_type):
                backend = detect_backend_from_interface(value)
            self.preferred_backend = register_and_enforce_backend(
                backend, self.preferred_backend
            )

        wrapped = ensure_fn(value, preferred_backend=self.preferred_backend)

        backend = (
            detect_backend_from_interface(wrapped)
            if detect_backend_from_interface
            else None
        )
        self.preferred_backend = register_and_enforce_backend(
            backend, self.preferred_backend
        )
        return wrapped


__all__ = ["BackendInputAdapter"]

"""
Helpers for working with SIRF/STIR native objects alongside the backend interfaces.

These utilities hide the wrapping/unwrapping boilerplate so callers can pass the
objects they already have, while the simulator continues to interact with the
uniform interface layer.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from sirf_simind_connection.backends import (
    AcquisitionDataInterface,
    ImageDataInterface,
    create_acquisition_data,
    create_image_data,
    detect_acquisition_backend,
    detect_backend_from_interface,
    detect_image_backend,
    get_backend,
    set_backend,
    unwrap,
)

image_like = Union[str, ImageDataInterface, Any]
acquisition_like = Union[str, AcquisitionDataInterface, Any]


def _validate_backend_name(name: str) -> str:
    if name not in ("sirf", "stir"):
        raise ValueError(f"Backend must be 'sirf' or 'stir', got {name!r}")
    return name


def _ensure_backend(preferred: Optional[str]) -> None:
    """Ensure a particular backend is active if explicitly requested."""
    if preferred is None:
        return

    preferred = _validate_backend_name(preferred)
    if get_backend() != preferred:
        set_backend(preferred)


def ensure_image_interface(
    value: image_like, preferred_backend: Optional[str] = None
) -> ImageDataInterface:
    """Return an ImageDataInterface for the provided value."""
    if isinstance(value, ImageDataInterface):
        if preferred_backend:
            backend = detect_backend_from_interface(value)
            preferred_backend = _validate_backend_name(preferred_backend)
            if backend and backend != preferred_backend:
                raise ValueError(
                    f"Image is backed by {backend}, expected {preferred_backend}"
                )
        return value

    if preferred_backend:
        _ensure_backend(preferred_backend)
    return create_image_data(value)


def ensure_acquisition_interface(
    value: acquisition_like, preferred_backend: Optional[str] = None
) -> AcquisitionDataInterface:
    """Return an AcquisitionDataInterface for the provided value."""
    if isinstance(value, AcquisitionDataInterface):
        if preferred_backend:
            backend = detect_backend_from_interface(value)
            preferred_backend = _validate_backend_name(preferred_backend)
            if backend and backend != preferred_backend:
                raise ValueError(
                    f"Acquisition data is backed by {backend}, "
                    f"expected {preferred_backend}"
                )
        return value

    if preferred_backend:
        _ensure_backend(preferred_backend)
    return create_acquisition_data(value)


def to_native_image(
    value: image_like,
    preferred_backend: Optional[str] = None,
    ensure_interface: bool = True,
) -> Any:
    """
    Retrieve a native SIRF/STIR image from any supported input.

    Args:
        value: Path, interface, or native object.
        preferred_backend: Optional backend to enforce. When provided and value
            is a path, the backend will be switched before loading.
        ensure_interface: When True, non-interface values are wrapped so that
            the simulator's interface guarantees remain intact before unwrapping.
    """
    if ensure_interface or not isinstance(value, ImageDataInterface):
        wrapper = ensure_image_interface(value, preferred_backend)
    else:
        wrapper = value

    native = unwrap(wrapper)
    if preferred_backend:
        expected = _validate_backend_name(preferred_backend)
        actual = detect_image_backend(native)
        if actual and actual != expected:
            raise ValueError(
                f"Image native backend is {actual}, expected {expected}"
            )
    return native


def to_native_acquisition(
    value: acquisition_like,
    preferred_backend: Optional[str] = None,
    ensure_interface: bool = True,
) -> Any:
    """Retrieve a native acquisition object from any supported input."""
    if ensure_interface or not isinstance(value, AcquisitionDataInterface):
        wrapper = ensure_acquisition_interface(value, preferred_backend)
    else:
        wrapper = value

    native = unwrap(wrapper)
    if preferred_backend:
        expected = _validate_backend_name(preferred_backend)
        actual = detect_acquisition_backend(native)
        if actual and actual != expected:
            raise ValueError(
                f"Acquisition native backend is {actual}, expected {expected}"
            )
    return native


def register_and_enforce_backend(
    detected_backend: Optional[str],
    current_backend: Optional[str]
) -> Optional[str]:
    """Register and enforce backend consistency across simulator inputs.

    This helper manages backend hints and ensures that all inputs to the
    simulator use the same backend (either SIRF or STIR). Once a backend
    is detected from the first input, subsequent inputs must match.

    Args:
        detected_backend: Backend detected from current input ('sirf', 'stir', or None)
        current_backend: Currently registered backend preference (or None)

    Returns:
        Updated backend preference (either current_backend or detected_backend)

    Raises:
        ValueError: If detected_backend conflicts with current_backend

    Example:
        >>> backend = None
        >>> backend = register_and_enforce_backend('sirf', backend)  # Returns 'sirf'
        >>> backend = register_and_enforce_backend('sirf', backend)  # OK, matches
        >>> backend = register_and_enforce_backend('stir', backend)  # Raises ValueError
    """
    if detected_backend:
        detected_backend = detected_backend.lower()

        # Validate backend name
        if detected_backend not in ('sirf', 'stir'):
            raise ValueError(f"Backend must be 'sirf' or 'stir', got {detected_backend!r}")

        # Check for conflicts
        if current_backend and current_backend != detected_backend:
            raise ValueError(
                f"Backend mismatch: simulator already configured for "
                f"{current_backend.upper()} backend but received "
                f"{detected_backend.upper()} data."
            )

        # Register the backend globally
        current_backend = detected_backend
        current = get_backend()
        if current != detected_backend:
            set_backend(detected_backend)

    return current_backend


__all__ = [
    "ensure_image_interface",
    "ensure_acquisition_interface",
    "to_native_image",
    "to_native_acquisition",
    "register_and_enforce_backend",
]

"""
Centralized backend import and access helpers.

This module provides a single source of truth for backend imports, eliminating
the need for duplicate try/except import blocks scattered throughout the codebase.

All backend-related imports (factory functions, interfaces, detection utilities,
and wrappers) are centralized here, with proper fallback handling when backends
are unavailable.
"""

from types import SimpleNamespace
from typing import Any, Dict, Optional


def get_backend_interfaces() -> tuple[bool, Dict[str, Any]]:
    """Get all backend interfaces and utilities with fallback handling.

    This is the single source of truth for backend imports. When backends are
    unavailable, it returns stub values (None or type(None)) so calling code
    can check BACKEND_AVAILABLE and handle the missing dependencies gracefully.

    Returns:
        Tuple of (BACKEND_AVAILABLE, interfaces_dict) where interfaces_dict contains:
        - 'factories': Backend factory functions (create_acquisition_data, create_image_data)
        - 'types': Interface type hints (AcquisitionDataInterface, ImageDataInterface)
        - 'wrappers': Wrapping utilities (ensure_acquisition_interface, ensure_image_interface, etc.)
        - 'detection': Backend detection functions (detect_*_backend, get/set_backend)

    Example:
        >>> BACKEND_AVAILABLE, backends = get_backend_interfaces()
        >>> if BACKEND_AVAILABLE:
        >>>     create_acq = backends['factories']['create_acquisition_data']
        >>>     data = create_acq('template.hs')
    """
    try:
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
        )
        from sirf_simind_connection.utils.sirf_stir_utils import (
            ensure_acquisition_interface,
            ensure_image_interface,
            to_native_acquisition,
            to_native_image,
        )

        return True, {
            'factories': {
                'create_acquisition_data': create_acquisition_data,
                'create_image_data': create_image_data,
            },
            'types': {
                'AcquisitionDataInterface': AcquisitionDataInterface,
                'ImageDataInterface': ImageDataInterface,
            },
            'wrappers': {
                'ensure_acquisition_interface': ensure_acquisition_interface,
                'ensure_image_interface': ensure_image_interface,
                'to_native_acquisition': to_native_acquisition,
                'to_native_image': to_native_image,
            },
            'detection': {
                'detect_acquisition_backend': detect_acquisition_backend,
                'detect_image_backend': detect_image_backend,
                'detect_backend_from_interface': detect_backend_from_interface,
                'get_backend': get_backend,
                'set_backend': set_backend,
            }
        }
    except ImportError:
        # Return fallback stubs when backends are unavailable
        return False, {
            'factories': {
                'create_acquisition_data': None,
                'create_image_data': None,
            },
            'types': {
                'AcquisitionDataInterface': type(None),
                'ImageDataInterface': type(None),
            },
            'wrappers': {
                'ensure_acquisition_interface': None,
                'ensure_image_interface': None,
                'to_native_acquisition': None,
                'to_native_image': None,
            },
            'detection': {
                'detect_acquisition_backend': None,
                'detect_image_backend': None,
                'detect_backend_from_interface': None,
                'get_backend': None,
                'set_backend': None,
            }
        }


def _to_namespace(mapping: Dict[str, Any]) -> SimpleNamespace:
    """Convert a dictionary to a SimpleNamespace (handles None gracefully)."""
    return SimpleNamespace(**(mapping or {}))


BACKEND_AVAILABLE, _interfaces = get_backend_interfaces()
BACKENDS = SimpleNamespace(
    factories=_to_namespace(_interfaces.get('factories', {})),
    types=_to_namespace(_interfaces.get('types', {})),
    wrappers=_to_namespace(_interfaces.get('wrappers', {})),
    detection=_to_namespace(_interfaces.get('detection', {})),
)


__all__ = ['get_backend_interfaces', 'BACKEND_AVAILABLE', 'BACKENDS']

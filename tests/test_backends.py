"""
Tests for the backend abstraction layer.

These tests verify that the backend system correctly detects and switches
between SIRF and STIR Python backends.
"""

import pytest


def test_backend_detection():
    """Test that backend auto-detection works."""
    from sirf_simind_connection.backends import get_backend, reset_backend

    # Reset to force re-detection
    reset_backend()

    try:
        backend = get_backend()
    except ImportError as exc:
        pytest.skip(f"Backend unavailable: {exc}")

    # Should detect one of the backends
    assert backend in ["sirf", "stir"], f"Unknown backend: {backend}"


@pytest.mark.requires_sirf
def test_backend_manual_selection_sirf():
    """Test manual backend selection for SIRF."""
    pytest.importorskip("sirf.STIR")

    from sirf_simind_connection.backends import get_backend, reset_backend, set_backend

    reset_backend()
    set_backend("sirf")
    assert get_backend() == "sirf"


@pytest.mark.requires_stir
def test_backend_manual_selection_stir():
    """Test manual backend selection for STIR Python."""
    pytest.importorskip("stir")
    pytest.importorskip("stirextra")

    from sirf_simind_connection.backends import get_backend, reset_backend, set_backend

    reset_backend()
    set_backend("stir")
    assert get_backend() == "stir"


def test_invalid_backend():
    """Test that invalid backend raises error."""
    from sirf_simind_connection.backends import set_backend

    with pytest.raises(ValueError, match="Invalid backend"):
        set_backend("invalid")


def test_backend_utility_functions():
    """Test is_sirf_backend and is_stir_backend functions."""
    from sirf_simind_connection.backends import (
        get_backend,
        is_sirf_backend,
        is_stir_backend,
        reset_backend,
    )

    reset_backend()
    try:
        backend = get_backend()
    except ImportError as exc:
        pytest.skip(f"Backend unavailable: {exc}")

    if backend == "sirf":
        assert is_sirf_backend() is True
        assert is_stir_backend() is False
    elif backend == "stir":
        assert is_sirf_backend() is False
        assert is_stir_backend() is True


def test_get_array_with_backend():
    """Test that get_array works with current backend."""
    # This test just ensures get_array is importable and doesn't crash
    # Actual data testing would require test fixtures
    import numpy as np

    from sirf_simind_connection.utils import get_array

    # Create a mock object with as_array method
    class MockImage:
        def as_array(self):
            return np.array([1, 2, 3])

    obj = MockImage()
    arr = get_array(obj)
    assert isinstance(arr, np.ndarray)
    assert arr.tolist() == [1, 2, 3]


@pytest.mark.requires_sirf
def test_sirf_backend_wrappers():
    """Test SIRF backend wrapper classes."""
    pytest.importorskip("sirf.STIR")

    from sirf_simind_connection.backends import reset_backend, set_backend
    from sirf_simind_connection.backends.sirf_backend import (
        SirfAcquisitionData,
        SirfImageData,
    )

    reset_backend()
    set_backend("sirf")

    # Test that classes are importable and have expected methods
    assert hasattr(SirfImageData, "read_from_file")
    assert hasattr(SirfImageData, "as_array")
    assert hasattr(SirfAcquisitionData, "read_from_file")
    assert hasattr(SirfAcquisitionData, "as_array")


@pytest.mark.requires_stir
def test_stir_backend_wrappers():
    """Test STIR Python backend wrapper classes."""
    pytest.importorskip("stir")
    pytest.importorskip("stirextra")

    from sirf_simind_connection.backends import reset_backend, set_backend
    from sirf_simind_connection.backends.stir_backend import (
        StirAcquisitionData,
        StirImageData,
    )

    reset_backend()
    set_backend("stir")

    # Test that classes are importable and have expected methods
    assert hasattr(StirImageData, "read_from_file")
    assert hasattr(StirImageData, "as_array")
    assert hasattr(StirAcquisitionData, "read_from_file")
    assert hasattr(StirAcquisitionData, "as_array")


def test_factory_functions():
    """Test factory functions are available."""
    from sirf_simind_connection.backends import (
        create_acquisition_data,
        create_image_data,
        load_acquisition_data,
        load_image_data,
    )

    # Just test that functions are importable
    assert callable(create_image_data)
    assert callable(create_acquisition_data)
    assert callable(load_image_data)
    assert callable(load_acquisition_data)


def test_load_functions_invalid_backend():
    """Ensure load helpers validate backend argument."""
    from sirf_simind_connection.backends import (
        load_acquisition_data,
        load_image_data,
    )

    with pytest.raises(ValueError, match="Invalid backend"):
        load_image_data("phantom.hv", backend="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Invalid backend"):
        load_acquisition_data("scan.hs", backend="invalid")  # type: ignore[arg-type]


def test_unwrap_function():
    """Test unwrap utility function."""
    from sirf_simind_connection.backends import unwrap

    # Test with mock object
    class MockWrapper:
        @property
        def native_object(self):
            return "native"

    wrapper = MockWrapper()
    assert unwrap(wrapper) == "native"

    # Test with non-wrapper (should return as-is)
    obj = "regular_object"
    assert unwrap(obj) == "regular_object"


@pytest.mark.unit
def test_detect_image_backend_with_mock(monkeypatch):
    """detect_image_backend should recognise mocked SIRF/STIR types."""
    import sirf_simind_connection.backends as backends

    class FakeSirfImage:
        pass

    class FakeStirImage:
        pass

    monkeypatch.setattr(backends, "_import_sirf_types", lambda: (FakeSirfImage, None))
    monkeypatch.setattr(backends, "_import_stir_types", lambda: (FakeStirImage, None))

    assert backends.detect_image_backend(FakeSirfImage()) == "sirf"
    assert backends.detect_image_backend(FakeStirImage()) == "stir"
    assert backends.detect_image_backend(object()) is None


@pytest.mark.unit
def test_detect_acquisition_backend_with_mock(monkeypatch):
    """detect_acquisition_backend should recognise mocked native classes."""
    import sirf_simind_connection.backends as backends

    class FakeSirfAcq:
        pass

    class FakeStirProj:
        pass

    monkeypatch.setattr(backends, "_import_sirf_types", lambda: (None, FakeSirfAcq))
    monkeypatch.setattr(backends, "_import_stir_types", lambda: (None, FakeStirProj))

    assert backends.detect_acquisition_backend(FakeSirfAcq()) == "sirf"
    assert backends.detect_acquisition_backend(FakeStirProj()) == "stir"
    assert backends.detect_acquisition_backend(object()) is None


@pytest.mark.unit
def test_detect_backend_from_interface(monkeypatch):
    """detect_backend_from_interface should unwrap and reuse detection helpers."""
    import sirf_simind_connection.backends as backends

    class FakeSirfAcq:
        pass

    class MockWrapper:
        def __init__(self, native):
            self._native = native

        @property
        def native_object(self):
            return self._native

    monkeypatch.setattr(backends, "_import_sirf_types", lambda: (None, FakeSirfAcq))
    monkeypatch.setattr(backends, "_import_stir_types", lambda: (None, None))

    wrapper = MockWrapper(FakeSirfAcq())
    assert backends.detect_backend_from_interface(wrapper) == "sirf"


def test_arithmetic_operations():
    """Test that arithmetic operations work correctly."""

    from sirf_simind_connection.backends import get_backend, reset_backend

    reset_backend()
    try:
        backend = get_backend()
    except ImportError as exc:
        pytest.skip(f"Backend unavailable: {exc}")

    if backend == "sirf":
        pytest.importorskip("sirf.STIR")
        # Create mock SIRF ImageData with arithmetic operations

        from sirf_simind_connection.backends.sirf_backend import SirfImageData

        # We can't easily create real ImageData without files, so skip if imports work
        assert hasattr(SirfImageData, "__add__")
        assert hasattr(SirfImageData, "__sub__")
        assert hasattr(SirfImageData, "__mul__")
        assert hasattr(SirfImageData, "__truediv__")

    elif backend == "stir":
        pytest.importorskip("stir")
        pytest.importorskip("stirextra")
        from sirf_simind_connection.backends.stir_backend import StirImageData

        # Same for STIR
        assert hasattr(StirImageData, "__add__")
        assert hasattr(StirImageData, "__sub__")
        assert hasattr(StirImageData, "__mul__")
        assert hasattr(StirImageData, "__truediv__")

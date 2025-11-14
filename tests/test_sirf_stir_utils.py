import pytest

from sirf_simind_connection.utils import sirf_stir_utils as utils


@pytest.mark.unit
def test_ensure_image_interface_passthrough(monkeypatch):
    """Interface instances should be returned unchanged."""

    class DummyInterface:
        pass

    dummy = DummyInterface()

    monkeypatch.setattr(utils, "ImageDataInterface", DummyInterface)
    monkeypatch.setattr(utils, "detect_backend_from_interface", lambda _: "sirf")

    result = utils.ensure_image_interface(dummy)
    assert result is dummy


@pytest.mark.unit
def test_ensure_image_interface_with_backend_hint(monkeypatch):
    """ensure_image_interface should activate backend before wrapping."""
    calls = {"backend": None, "value": None}

    monkeypatch.setattr(utils, "create_image_data", lambda value: f"wrapped:{value}")
    monkeypatch.setattr(
        utils, "_ensure_backend", lambda backend: calls.update(backend=backend)
    )

    result = utils.ensure_image_interface("phantom.hv", preferred_backend="sirf")

    assert result == "wrapped:phantom.hv"
    assert calls["backend"] == "sirf"


@pytest.mark.unit
def test_to_native_acquisition_prefers_unwrap(monkeypatch):
    """to_native_acquisition should unwrap interfaces when preferred backend matches."""

    class DummyInterface:
        pass

    wrapper = DummyInterface()
    native = object()

    monkeypatch.setattr(utils, "AcquisitionDataInterface", DummyInterface)
    monkeypatch.setattr(utils, "unwrap", lambda value: native)
    monkeypatch.setattr(utils, "detect_acquisition_backend", lambda _: "sirf")

    result = utils.to_native_acquisition(
        wrapper, preferred_backend="sirf", ensure_interface=False
    )

    assert result is native


import types

import pytest

import sirf_simind_connection.core.backend_adapter as adapter_mod
from sirf_simind_connection.core.backend_adapter import BackendInputAdapter


class DummyInterface:
    def __init__(self, value, backend):
        self.value = value
        self.backend = backend


def _install_fake_backend(monkeypatch):
    def fake_register(detected, current):
        detected = detected or current
        if detected and current and detected != current:
            raise ValueError("Backend mismatch")
        return detected

    def fake_ensure(value, preferred_backend=None):
        backend = preferred_backend or getattr(value, "backend_hint", "sirf")
        return DummyInterface(value, backend)

    def fake_detector(value):
        return getattr(value, "backend_hint", None)

    monkeypatch.setattr(adapter_mod, "BACKEND_AVAILABLE", True)
    monkeypatch.setattr(adapter_mod, "register_and_enforce_backend", fake_register)
    monkeypatch.setattr(adapter_mod, "ensure_image_interface", fake_ensure)
    monkeypatch.setattr(adapter_mod, "ensure_acquisition_interface", fake_ensure)
    monkeypatch.setattr(adapter_mod, "detect_image_backend", fake_detector)
    monkeypatch.setattr(adapter_mod, "detect_acquisition_backend", fake_detector)
    monkeypatch.setattr(
        adapter_mod, "detect_backend_from_interface", lambda obj: obj.backend
    )
    monkeypatch.setattr(adapter_mod, "ImageDataInterface", DummyInterface)
    monkeypatch.setattr(adapter_mod, "AcquisitionDataInterface", DummyInterface)


@pytest.mark.unit
def test_backend_adapter_prefers_first_backend(monkeypatch):
    """First wrapped object sets backend preference."""
    _install_fake_backend(monkeypatch)
    adapter = BackendInputAdapter()
    first = types.SimpleNamespace(backend_hint="sirf")

    wrapped = adapter.wrap_image(first)
    assert wrapped.backend == "sirf"
    assert adapter.get_preferred_backend() == "sirf"

    # Subsequent acquisitions inherit same backend
    acq = types.SimpleNamespace(backend_hint="sirf")
    wrapped_acq = adapter.wrap_acquisition(acq)
    assert wrapped_acq.backend == "sirf"


@pytest.mark.unit
def test_backend_adapter_raises_on_mixed_backends(monkeypatch):
    """Mixing SIRF/STIR inputs should raise ValueError."""
    _install_fake_backend(monkeypatch)
    adapter = BackendInputAdapter()
    adapter.wrap_image(types.SimpleNamespace(backend_hint="sirf"))

    with pytest.raises(ValueError, match="Backend mismatch"):
        adapter.wrap_image(types.SimpleNamespace(backend_hint="stir"))

import numpy as np
import pytest

from sirf_simind_connection.builders.image_builder import STIRSPECTImageDataBuilder


class DummyWrappedImage:
    """Wrapper-like stand-in exposing a native_object attribute."""

    def __init__(self, header_path):
        self.header_path = header_path
        self._native = object()

    @property
    def native_object(self):
        return self._native


@pytest.mark.unit
def test_image_builder_explicit_backend_restores_global_backend(monkeypatch, tmp_path):
    from sirf_simind_connection.builders import image_builder as builder_mod

    created = []
    set_backend_calls = []

    def _factory(header_path):
        obj = DummyWrappedImage(header_path)
        created.append(obj)
        return obj

    monkeypatch.setattr(builder_mod.BACKENDS.factories, "create_image_data", _factory)
    monkeypatch.setattr(builder_mod.BACKENDS.detection, "get_backend", lambda: "sirf")
    monkeypatch.setattr(
        builder_mod.BACKENDS.detection,
        "set_backend",
        lambda backend: set_backend_calls.append(backend),
    )

    builder = STIRSPECTImageDataBuilder(
        header_overrides={
            "!matrix size [1]": "4",
            "!matrix size [2]": "3",
            "!matrix size [3]": "2",
        },
        backend="stir",
    )
    builder.set_pixel_array(np.ones((2, 3, 4), dtype=np.float32))
    output = builder.build(output_path=tmp_path / "img")

    assert set_backend_calls == ["stir", "sirf"]
    assert output is created[0].native_object


@pytest.mark.unit
def test_image_builder_autodetect_backend_when_not_specified(monkeypatch, tmp_path):
    from sirf_simind_connection.builders import image_builder as builder_mod

    created = []
    set_backend_calls = []

    def _factory(header_path):
        obj = DummyWrappedImage(header_path)
        created.append(obj)
        return obj

    monkeypatch.setattr(builder_mod.BACKENDS.factories, "create_image_data", _factory)
    monkeypatch.setattr(
        builder_mod.BACKENDS.detection,
        "set_backend",
        lambda backend: set_backend_calls.append(backend),
    )

    builder = STIRSPECTImageDataBuilder(
        header_overrides={
            "!matrix size [1]": "4",
            "!matrix size [2]": "3",
            "!matrix size [3]": "2",
        }
    )
    builder.set_pixel_array(np.zeros((2, 3, 4), dtype=np.float32))
    output = builder.build(output_path=tmp_path / "img")

    assert set_backend_calls == []
    assert output is created[0].native_object


@pytest.mark.unit
def test_image_builder_rejects_invalid_backend():
    with pytest.raises(ValueError, match="backend must be one of"):
        STIRSPECTImageDataBuilder(backend="invalid")  # type: ignore[arg-type]

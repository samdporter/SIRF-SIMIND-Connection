import numpy as np
import pytest

from sirf_simind_connection.utils import stir_utils


pytestmark = pytest.mark.unit


def test_create_stir_image_forwards_explicit_backend(monkeypatch):
    import sirf_simind_connection.builders as builders

    captured = {}

    class FakeImageBuilder:
        def __init__(self, header_overrides=None, backend=None):
            captured["backend"] = backend
            self.header = dict(header_overrides or {})

        def update_header(self, updates):
            self.header.update(updates)
            captured["header"] = dict(self.header)

        def set_pixel_array(self, array):
            captured["shape"] = tuple(array.shape)

        def build(self):
            return {"kind": "image", "backend": captured["backend"]}

    monkeypatch.setattr(builders, "STIRSPECTImageDataBuilder", FakeImageBuilder)

    output = stir_utils.create_stir_image(
        matrix_dim=[2, 3, 4],
        voxel_size=[1.0, 2.0, 3.0],
        backend="sirf",
    )

    assert output["kind"] == "image"
    assert output["backend"] == "sirf"
    assert captured["shape"] == (2, 3, 4)


def test_create_stir_acqdata_forwards_explicit_backend(monkeypatch):
    import sirf_simind_connection.builders as builders

    captured = {}

    class FakeAcqBuilder:
        def __init__(self, header_overrides=None, backend=None):
            captured["backend"] = backend
            self.header = dict(header_overrides or {})
            self.pixel_array = None

        def update_header(self, updates):
            self.header.update(updates)
            captured["header"] = dict(self.header)

        def build(self):
            captured["pixel_shape"] = tuple(np.asarray(self.pixel_array).shape)
            return {"kind": "acquisition", "backend": captured["backend"]}

    monkeypatch.setattr(builders, "STIRSPECTAcquisitionDataBuilder", FakeAcqBuilder)

    output = stir_utils.create_stir_acqdata(
        proj_matrix=[3, 4],
        num_projections=5,
        pixel_size=[2.0, 2.0],
        backend="stir",
    )

    assert output["kind"] == "acquisition"
    assert output["backend"] == "stir"
    assert captured["pixel_shape"] == (1, 3, 5, 4)


def test_stir_utils_backend_choice_rejects_invalid_value():
    with pytest.raises(ValueError, match="Invalid backend"):
        stir_utils.create_stir_image([2, 2, 2], [1.0, 1.0, 1.0], backend="invalid")

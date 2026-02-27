from pathlib import Path

import numpy as np
import pytest

import sirf_simind_connection.connectors.pytomography_adaptor as pytomo_mod
from sirf_simind_connection.configs import get
from sirf_simind_connection.connectors.python_connector import ProjectionResult
from sirf_simind_connection.connectors.pytomography_adaptor import (
    PyTomographySimindAdaptor,
)


torch = pytest.importorskip("torch")


pytestmark = pytest.mark.requires_pytomography


@pytest.mark.unit
def test_pytomography_adaptor_preserves_projection_shape(tmp_path: Path, monkeypatch):
    connector = PyTomographySimindAdaptor(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
        voxel_size_mm=4.0,
    )

    source = torch.zeros((6, 8, 8), dtype=torch.float32)
    source[2:5, 2:6, 2:6] = 1.0
    mu_map = torch.zeros_like(source)
    mu_map[source > 0] = 0.15

    connector.set_source(source)
    connector.set_mu_map(mu_map)
    connector.set_energy_windows([126], [154], [0])

    projection = np.arange(1 * 8 * 5 * 8, dtype=np.float32).reshape(1, 8, 5, 8)
    fake_result = ProjectionResult(
        projection=projection,
        header_path=tmp_path / "case01_tot_w1.hs",
        data_path=tmp_path / "case01_tot_w1.a00",
        metadata={"key": "tot_w1"},
    )

    monkeypatch.setattr(
        connector.python_connector,
        "run",
        lambda runtime_operator=None: {"tot_w1": fake_result},
    )

    outputs = connector.run()

    assert "tot_w1" in outputs
    assert tuple(outputs["tot_w1"].shape) == (1, 8, 5, 8)
    assert torch.equal(outputs["tot_w1"], torch.from_numpy(projection))
    assert (
        connector.get_output_header_path("tot_w1") == fake_result.header_path.resolve()
    )


@pytest.mark.unit
def test_pytomography_adaptor_system_matrix_helpers(tmp_path: Path, monkeypatch):
    connector = PyTomographySimindAdaptor(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    h00_path = tmp_path / "case01_tot_w1.h00"
    h00_path.write_text("!INTERFILE :=\n")
    connector._output_header_paths = {"tot_w1": h00_path}

    class DummySimind:
        @staticmethod
        def get_metadata(path):
            assert path == str(h00_path)
            return ("obj_meta", "proj_meta")

        @staticmethod
        def get_psfmeta_from_header(path):
            assert path == str(h00_path)
            return "psf_meta"

    class DummyPSFTransform:
        def __init__(self, psf_meta):
            self.psf_meta = psf_meta

    class DummySystemMatrix:
        def __init__(
            self, obj2obj_transforms, proj2proj_transforms, object_meta, proj_meta
        ):
            self.obj2obj_transforms = obj2obj_transforms
            self.proj2proj_transforms = proj2proj_transforms
            self.object_meta = object_meta
            self.proj_meta = proj_meta

    monkeypatch.setattr(pytomo_mod, "pytomo_simind", DummySimind)
    monkeypatch.setattr(pytomo_mod, "SPECTPSFTransform", DummyPSFTransform)
    monkeypatch.setattr(pytomo_mod, "SPECTSystemMatrix", DummySystemMatrix)

    object_meta, proj_meta = connector.get_pytomography_metadata("tot_w1")
    assert object_meta == "obj_meta"
    assert proj_meta == "proj_meta"

    system_matrix = connector.build_system_matrix("tot_w1", use_psf=True)
    assert isinstance(system_matrix, DummySystemMatrix)
    assert system_matrix.object_meta == "obj_meta"
    assert system_matrix.proj_meta == "proj_meta"
    assert system_matrix.proj2proj_transforms == []
    assert len(system_matrix.obj2obj_transforms) == 1
    assert system_matrix.obj2obj_transforms[0].psf_meta == "psf_meta"


@pytest.mark.unit
def test_pytomography_adaptor_uses_mu_map_in_pytomography_axes(
    tmp_path: Path, monkeypatch
):
    connector = PyTomographySimindAdaptor(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    source = torch.zeros((2, 3, 4), dtype=torch.float32)
    mu_map = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    connector.set_source(source)
    connector.set_mu_map(mu_map)

    h00_path = tmp_path / "case01_tot_w1.h00"
    h00_path.write_text("!INTERFILE :=\n")
    connector._output_header_paths = {"tot_w1": h00_path}

    class DummySimind:
        @staticmethod
        def get_metadata(path):
            assert path == str(h00_path)
            return ("obj_meta", "proj_meta")

    class DummyAttenuationTransform:
        def __init__(self, attenuation_map):
            self.attenuation_map = attenuation_map

    class DummySystemMatrix:
        def __init__(
            self, obj2obj_transforms, proj2proj_transforms, object_meta, proj_meta
        ):
            self.obj2obj_transforms = obj2obj_transforms
            self.proj2proj_transforms = proj2proj_transforms
            self.object_meta = object_meta
            self.proj_meta = proj_meta

    monkeypatch.setattr(pytomo_mod, "pytomo_simind", DummySimind)
    monkeypatch.setattr(pytomo_mod, "SPECTSystemMatrix", DummySystemMatrix)
    monkeypatch.setattr(
        pytomo_mod, "SPECTAttenuationTransform", DummyAttenuationTransform
    )
    monkeypatch.setattr(pytomo_mod, "SPECTPSFTransform", None)

    system_matrix = connector.build_system_matrix(
        "tot_w1", use_psf=False, use_attenuation=True
    )

    assert len(system_matrix.obj2obj_transforms) == 1
    attenuation_map = system_matrix.obj2obj_transforms[0].attenuation_map
    assert tuple(attenuation_map.shape) == tuple(mu_map.shape)
    assert torch.equal(attenuation_map, mu_map.contiguous())


@pytest.mark.unit
def test_pytomography_axis_helpers_roundtrip_through_simind_order() -> None:
    xyz = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    zyx = PyTomographySimindAdaptor.to_simind_image_axes(xyz)
    restored = PyTomographySimindAdaptor.from_simind_image_axes(zyx)

    assert tuple(zyx.shape) == (4, 3, 2)
    assert tuple(restored.shape) == (2, 3, 4)
    assert torch.equal(restored, xyz)

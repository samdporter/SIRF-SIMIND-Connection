from pathlib import Path

import numpy as np
import pytest

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
def test_pytomography_adaptor_avoids_wrapping_pytomography_methods(tmp_path: Path):
    connector = PyTomographySimindAdaptor(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    assert not hasattr(connector, "build_system_matrix")
    assert not hasattr(connector, "get_pytomography_metadata")


@pytest.mark.unit
def test_pytomography_axis_helpers_roundtrip_through_simind_order() -> None:
    xyz = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    zyx = PyTomographySimindAdaptor.to_simind_image_axes(xyz)
    restored = PyTomographySimindAdaptor.from_simind_image_axes(zyx)

    assert tuple(zyx.shape) == (4, 3, 2)
    assert tuple(restored.shape) == (2, 3, 4)
    assert torch.equal(restored, xyz)

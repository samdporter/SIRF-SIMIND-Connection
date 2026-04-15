from pathlib import Path

import numpy as np
import pytest

from py_smc.configs import get
from py_smc.connectors.python_connector import (
    ProjectionResult,
    RuntimeOperator,
)
from py_smc.connectors.pytomography_adaptor import (
    PyTomographySimindAdaptor,
)
from py_smc.core.types import ScoringRoutine


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
def test_pytomography_adaptor_forwards_connector_wiring_and_axis_order(
    tmp_path: Path, monkeypatch
):
    connector = PyTomographySimindAdaptor(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
        voxel_size_mm=3.5,
        scoring_routine=ScoringRoutine.PENETRATE,
    )

    source_xyz = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    mu_xyz = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) / 10.0
    connector.set_source(source_xyz)
    connector.set_mu_map(mu_xyz)
    connector.set_energy_windows([75.0, 120.0], [85.0, 140.0], [0, 1])

    captured: dict[str, object] = {}

    def fake_configure_voxel_phantom(
        source,
        mu_map,
        voxel_size_mm,
        scoring_routine,
    ):
        captured["source"] = np.asarray(source)
        captured["mu_map"] = np.asarray(mu_map)
        captured["voxel_size_mm"] = voxel_size_mm
        captured["scoring_routine"] = scoring_routine
        return (tmp_path / "case01_src.smi", tmp_path / "case01_dns.dmi")

    def fake_set_energy_windows(lower_bounds, upper_bounds, scatter_orders):
        captured["windows"] = (
            list(lower_bounds),
            list(upper_bounds),
            list(scatter_orders),
        )

    projection = np.arange(1 * 8 * 5 * 8, dtype=np.float32).reshape(1, 8, 5, 8)
    raw_outputs = {
        "tot_w1": ProjectionResult(
            projection=projection,
            header_path=tmp_path / "case01_tot_w1.hs",
            data_path=tmp_path / "case01_tot_w1.a00",
            metadata={"component": "tot_w1"},
        ),
        "sca_w1": ProjectionResult(
            projection=projection + 1.0,
            header_path=tmp_path / "case01_sca_w1.hs",
            data_path=tmp_path / "case01_sca_w1.a00",
            metadata={"component": "sca_w1"},
        ),
        "pri_w1": ProjectionResult(
            projection=projection + 2.0,
            header_path=tmp_path / "case01_pri_w1.hs",
            data_path=tmp_path / "case01_pri_w1.a00",
            metadata={"component": "pri_w1"},
        ),
        "air_w1": ProjectionResult(
            projection=projection + 3.0,
            header_path=tmp_path / "case01_air_w1.hs",
            data_path=tmp_path / "case01_air_w1.a00",
            metadata={"component": "air_w1"},
        ),
    }

    def fake_run(runtime_operator=None):
        captured["runtime_operator"] = runtime_operator
        return raw_outputs

    monkeypatch.setattr(
        connector.python_connector,
        "configure_voxel_phantom",
        fake_configure_voxel_phantom,
    )
    monkeypatch.setattr(
        connector.python_connector,
        "set_energy_windows",
        fake_set_energy_windows,
    )
    monkeypatch.setattr(
        connector.python_connector,
        "run",
        fake_run,
    )

    runtime_operator = RuntimeOperator(switches={"RR": 12345})
    outputs = connector.run(runtime_operator=runtime_operator)

    expected_source_zyx = source_xyz.permute(2, 1, 0).numpy()
    expected_mu_zyx = mu_xyz.permute(2, 1, 0).numpy()
    assert np.array_equal(captured["source"], expected_source_zyx)
    assert np.array_equal(captured["mu_map"], expected_mu_zyx)
    assert captured["voxel_size_mm"] == pytest.approx(3.5)
    assert captured["scoring_routine"] == ScoringRoutine.PENETRATE
    assert captured["windows"] == ([75.0, 120.0], [85.0, 140.0], [0, 1])
    assert captured["runtime_operator"] is runtime_operator

    assert torch.equal(connector.get_total_output(window=1), outputs["tot_w1"])
    assert torch.equal(connector.get_scatter_output(window=1), outputs["sca_w1"])
    assert torch.equal(connector.get_primary_output(window=1), outputs["pri_w1"])
    assert torch.equal(connector.get_air_output(window=1), outputs["air_w1"])


@pytest.mark.unit
def test_pytomography_axis_helpers_roundtrip_through_simind_order() -> None:
    xyz = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    zyx = PyTomographySimindAdaptor.to_simind_image_axes(xyz)
    restored = PyTomographySimindAdaptor.from_simind_image_axes(zyx)

    assert tuple(zyx.shape) == (4, 3, 2)
    assert tuple(restored.shape) == (2, 3, 4)
    assert torch.equal(restored, xyz)

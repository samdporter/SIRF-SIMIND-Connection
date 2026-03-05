from pathlib import Path

import numpy as np
import pytest

from sirf_simind_connection.configs import get
from sirf_simind_connection.connectors import RuntimeOperator, SimindPythonConnector
from sirf_simind_connection.core.types import ScoringRoutine


@pytest.mark.unit
def test_python_connector_requires_run_before_get_outputs(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    with pytest.raises(RuntimeError, match="Run the connector first"):
        connector.get_outputs()


@pytest.mark.unit
def test_python_connector_accepts_quantization_scale_parameter(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
        quantization_scale=0.05,
    )
    assert connector.quantization_scale == pytest.approx(0.05)


@pytest.mark.unit
def test_python_connector_rejects_non_positive_quantization_scale(tmp_path: Path):
    with pytest.raises(ValueError, match="quantization_scale must be > 0"):
        SimindPythonConnector(
            config_source=get("AnyScan.yaml"),
            output_dir=tmp_path,
            output_prefix="case01",
            quantization_scale=0.0,
        )


@pytest.mark.unit
def test_python_connector_run_returns_numpy_outputs(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    captured: dict[str, object] = {}

    def fake_run_simulation(
        output_prefix: str,
        orbit_file=None,
        runtime_switches=None,
    ) -> None:
        captured["output_prefix"] = output_prefix
        captured["orbit_file"] = orbit_file
        captured["runtime_switches"] = dict(runtime_switches or {})

        projection = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        data_path = tmp_path / f"{output_prefix}_tot_w1.a00"
        header_path = tmp_path / f"{output_prefix}_tot_w1.hs"

        projection.tofile(data_path)
        header_path.write_text(
            "\n".join(
                [
                    "!INTERFILE :=",
                    "!number format := float",
                    "!number of bytes per pixel := 4",
                    "imagedata byte order := LITTLEENDIAN",
                    "!matrix size [1] := 4",
                    "!matrix size [2] := 3",
                    "!matrix size [3] := 2",
                    f"!name of data file := {data_path.name}",
                    "!END OF INTERFILE :=",
                ]
            )
        )

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    outputs = connector.run(RuntimeOperator(switches={"NN": 2, "RR": 12345}))

    assert captured["output_prefix"] == "case01"
    assert captured["runtime_switches"] == {"NN": 2, "RR": 12345}

    assert "tot_w1" in outputs
    result = outputs["tot_w1"]
    assert result.header_path == (tmp_path / "case01_tot_w1.hs").resolve()
    assert result.data_path == (tmp_path / "case01_tot_w1.a00").resolve()
    assert result.projection.shape == (2, 3, 4)
    expected_sum = float(np.arange(24, dtype=np.float32).sum())
    assert float(result.projection.sum()) == expected_sum


@pytest.mark.unit
def test_python_connector_skips_malformed_interfile_outputs(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    def fake_run_simulation(
        output_prefix: str,
        orbit_file=None,
        runtime_switches=None,
    ) -> None:
        good_projection = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        good_data_path = tmp_path / f"{output_prefix}_tot_w1.a00"
        good_header_path = tmp_path / f"{output_prefix}_tot_w1.hs"
        good_projection.tofile(good_data_path)
        good_header_path.write_text(
            "\n".join(
                [
                    "!INTERFILE :=",
                    "!number format := float",
                    "!number of bytes per pixel := 4",
                    "imagedata byte order := LITTLEENDIAN",
                    "!matrix size [1] := 4",
                    "!matrix size [2] := 3",
                    "!matrix size [3] := 2",
                    f"!name of data file := {good_data_path.name}",
                    "!END OF INTERFILE :=",
                ]
            )
        )

        bad_projection = np.arange(16, dtype=np.float32).reshape(4, 4)
        bad_data_path = tmp_path / f"{output_prefix}_air_w1.a00"
        bad_header_path = tmp_path / f"{output_prefix}_air_w1.hs"
        bad_projection.tofile(bad_data_path)
        bad_header_path.write_text(
            "\n".join(
                [
                    "!INTERFILE :=",
                    "!number format := float",
                    "!number of bytes per pixel := 4",
                    "imagedata byte order := LITTLEENDIAN",
                    "!matrix size [1] := 4",
                    "!matrix size [2] := 4",
                    "!matrix size [3] := 4",
                    f"!name of data file := {bad_data_path.name}",
                    "!END OF INTERFILE :=",
                ]
            )
        )

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    outputs = connector.run(RuntimeOperator(switches={"NN": 1}))

    assert "tot_w1" in outputs
    assert "air_w1" not in outputs


@pytest.mark.unit
def test_python_connector_penetrate_uses_bxx_component_headers(
    tmp_path: Path, monkeypatch
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    connector.add_config_value(84, 4)

    def fake_run_simulation(
        output_prefix: str,
        orbit_file=None,
        runtime_switches=None,
    ) -> None:
        (tmp_path / f"{output_prefix}.h00").write_text(
            "\n".join(["!INTERFILE :=", "!END OF INTERFILE :="])
        )

        projection = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        data_path = tmp_path / f"{output_prefix}.b01"
        header_path = tmp_path / f"{output_prefix}_component_01.hs"
        projection.tofile(data_path)
        header_path.write_text(
            "\n".join(
                [
                    "!INTERFILE :=",
                    "!number format := float",
                    "!number of bytes per pixel := 4",
                    "imagedata byte order := LITTLEENDIAN",
                    "!matrix size [1] := 4",
                    "!matrix size [2] := 3",
                    "!matrix size [3] := 2",
                    f"!name of data file := {data_path.name}",
                    "!END OF INTERFILE :=",
                ]
            )
        )

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    monkeypatch.setattr(
        connector.converter,
        "find_penetrate_h00_file",
        lambda output_prefix, output_dir: str(tmp_path / f"{output_prefix}.h00"),
    )
    monkeypatch.setattr(
        connector.converter,
        "create_penetrate_headers_from_template",
        lambda h00_file, output_prefix, output_dir: {},
    )

    outputs = connector.run(RuntimeOperator(switches={"NN": 1}))

    assert "all_interactions" in outputs
    result = outputs["all_interactions"]
    assert result.data_path == (tmp_path / "case01.b01").resolve()
    assert result.projection.shape == (2, 3, 4)


@pytest.mark.unit
def test_python_connector_configure_voxel_phantom_writes_input_files(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    source = np.zeros((4, 5, 6), dtype=np.float32)
    source[1:3, 2:4, 2:5] = 1.0
    mu_map = np.full_like(source, 0.15, dtype=np.float32)

    source_path, density_path = connector.configure_voxel_phantom(
        source=source,
        mu_map=mu_map,
        voxel_size_mm=4.0,
        scoring_routine=1,
    )

    assert source_path.exists()
    assert density_path.exists()
    assert source_path.name == "case01_src.smi"
    assert density_path.name == "case01_dns.dmi"
    assert connector.runtime_switches.switches["PX"] == pytest.approx(0.4)

    source_u16 = np.fromfile(source_path, dtype=np.uint16)
    density_u16 = np.fromfile(density_path, dtype=np.uint16)
    assert source_u16.size == source.size
    assert density_u16.size == mu_map.size
    assert source_u16.max() > 0


@pytest.mark.unit
def test_python_connector_configure_voxel_phantom_rejects_shape_mismatch(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    source = np.zeros((4, 5, 6), dtype=np.float32)
    mu_map = np.zeros((4, 5, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="identical shapes"):
        connector.configure_voxel_phantom(source=source, mu_map=mu_map)


@pytest.mark.unit
def test_python_connector_configure_voxel_phantom_rejects_non_3d_inputs(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    source_2d = np.zeros((4, 5), dtype=np.float32)
    mu_2d = np.zeros((4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="must both be 3D arrays"):
        connector.configure_voxel_phantom(source=source_2d, mu_map=mu_2d)

    source_4d = np.zeros((2, 3, 4, 5), dtype=np.float32)
    mu_4d = np.zeros((2, 3, 4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="must both be 3D arrays"):
        connector.configure_voxel_phantom(source=source_4d, mu_map=mu_4d)


@pytest.mark.unit
def test_python_connector_configure_voxel_phantom_rejects_non_positive_voxel_size(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    source = np.zeros((4, 5, 6), dtype=np.float32)
    mu_map = np.zeros_like(source)

    with pytest.raises(ValueError, match="voxel_size_mm must be > 0"):
        connector.configure_voxel_phantom(
            source=source,
            mu_map=mu_map,
            voxel_size_mm=0.0,
        )

    with pytest.raises(ValueError, match="voxel_size_mm must be > 0"):
        connector.configure_voxel_phantom(
            source=source,
            mu_map=mu_map,
            voxel_size_mm=-4.0,
        )


@pytest.mark.unit
def test_python_connector_configure_voxel_phantom_zeroes_density_when_attenuation_off(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    connector.get_config().set_flag(11, False)

    source = np.zeros((4, 5, 6), dtype=np.float32)
    source[1:3, 1:4, 1:5] = 1.0
    mu_map = np.full_like(source, 0.25, dtype=np.float32)

    _, density_path = connector.configure_voxel_phantom(source=source, mu_map=mu_map)
    density_u16 = np.fromfile(density_path, dtype=np.uint16)
    assert density_u16.size == mu_map.size
    assert np.all(density_u16 == 0)


@pytest.mark.unit
def test_python_connector_configure_voxel_phantom_accepts_scoring_routine_enum(
    tmp_path: Path,
):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )
    source = np.zeros((4, 5, 6), dtype=np.float32)
    mu_map = np.zeros_like(source)

    connector.configure_voxel_phantom(
        source=source,
        mu_map=mu_map,
        scoring_routine=ScoringRoutine.PENETRATE,
    )

    assert int(connector.get_config().get_value(84)) == ScoringRoutine.PENETRATE.value


@pytest.mark.unit
def test_python_connector_set_energy_windows_writes_window_file(tmp_path: Path):
    connector = SimindPythonConnector(
        config_source=get("AnyScan.yaml"),
        output_dir=tmp_path,
        output_prefix="case01",
    )

    connector.set_energy_windows([126.0], [154.0], [0])
    window_file = tmp_path / "case01.win"
    assert window_file.exists()

    lines = [line.strip() for line in window_file.read_text().splitlines() if line]
    assert lines[0] == "126.0,154.0,0"

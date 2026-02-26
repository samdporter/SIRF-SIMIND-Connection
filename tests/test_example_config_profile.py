import pytest

from sirf_simind_connection.configs import get
from sirf_simind_connection.core.config import SimulationConfig


pytestmark = pytest.mark.unit


def _value(cfg: SimulationConfig, key: str) -> float:
    return float(cfg.get_value(key))


def test_example_yaml_uses_smaller_projection_space_than_anyscan():
    anyscan = SimulationConfig(get("AnyScan.yaml"))
    example = SimulationConfig(get("Example.yaml"))

    assert _value(example, "spect_no_projections") < _value(
        anyscan, "spect_no_projections"
    )
    assert _value(example, "matrix_size_image_i") < _value(
        anyscan, "matrix_size_image_i"
    )
    assert _value(example, "matrix_size_image_j") < _value(
        anyscan, "matrix_size_image_j"
    )
    assert _value(example, "number_density_images") < _value(
        anyscan, "number_density_images"
    )
    assert _value(example, "matrix_size_density_map_i") < _value(
        anyscan, "matrix_size_density_map_i"
    )
    assert _value(example, "matrix_size_source_map_i") < _value(
        anyscan, "matrix_size_source_map_i"
    )


def test_example_yaml_enables_spect_flag():
    example = SimulationConfig(get("Example.yaml"))
    assert example.get_flag("simulate_spect_study")

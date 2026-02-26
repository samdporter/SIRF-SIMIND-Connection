from pathlib import Path

import numpy as np
import pytest

from sirf_simind_connection.core.file_managers import DataFileManager


class DummyImage:
    def __init__(self, array: np.ndarray):
        self._array = array

    def as_array(self) -> np.ndarray:
        return self._array


@pytest.mark.unit
def test_prepare_source_file_applies_quantization_scaling(tmp_path: Path):
    source = DummyImage(np.array([[[0.0, 0.5], [1.0, 0.25]]], dtype=np.float32))
    manager = DataFileManager(tmp_path, quantization_scale=0.05)

    manager.prepare_source_file(source, "case01")
    smi_path = tmp_path / "case01_src.smi"
    values = np.fromfile(smi_path, dtype=np.uint16)

    assert values.max() == 25
    assert values.min() == 0


@pytest.mark.unit
def test_data_file_manager_rejects_non_positive_quantization_scale(tmp_path: Path):
    with pytest.raises(ValueError, match="quantization_scale must be > 0"):
        DataFileManager(tmp_path, quantization_scale=0.0)

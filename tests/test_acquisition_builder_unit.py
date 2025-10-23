import sys
import types


if "pydicom" not in sys.modules:
    sys.modules["pydicom"] = types.ModuleType("pydicom")

import numpy as np
import pytest

from sirf_simind_connection.builders.acquisition_builder import (
    STIRSPECTAcquisitionDataBuilder,
)


class DummyAcquisitionData:
    """Minimal stand-in for backend acquisition data used by the builder."""

    def __init__(self, header_path):
        self.header_path = header_path
        self.fill_data = None
        self.write_calls = []

    def clone(self):
        return self

    def fill(self, data):
        self.fill_data = np.asarray(data)
        return self

    def write(self, path):
        self.write_calls.append(path)

    def get_uniform_copy(self, value):
        if self.fill_data is None:
            arr = np.asarray(value, dtype=float)
        else:
            arr = np.full_like(self.fill_data, value, dtype=float)
        copy = DummyAcquisitionData(self.header_path)
        copy.fill(arr)
        return copy


@pytest.fixture
def fake_create_acquisition(monkeypatch):
    """Patch create_acquisition_data to avoid SIRF/STIR dependency."""
    created_objects = []

    def _factory(header_path):
        obj = DummyAcquisitionData(header_path)
        created_objects.append(obj)
        return obj

    monkeypatch.setattr(
        "sirf_simind_connection.builders.acquisition_builder.create_acquisition_data",
        _factory,
    )
    return created_objects


@pytest.mark.unit
def test_build_writes_header_and_data(tmp_path, fake_create_acquisition):
    builder = STIRSPECTAcquisitionDataBuilder()

    output_prefix = tmp_path / "acq"
    acq = builder.build(output_path=str(output_prefix))

    header_path = output_prefix.with_suffix(".hs")
    raw_path = output_prefix.with_suffix(".s")

    # Files should exist and contain the terminating key
    assert header_path.exists()
    assert raw_path.exists()
    header_text = header_path.read_text()
    assert "!END OF INTERFILE :=" in header_text

    # Builder should return the stub object and write via it
    assert acq is fake_create_acquisition[0]
    assert acq.write_calls == [str(header_path)]
    assert acq.fill_data is not None
    assert acq.fill_data.shape[0] == 1  # segments dimension


@pytest.mark.unit
def test_build_multi_energy_splits_windows(tmp_path, fake_create_acquisition):
    builder = STIRSPECTAcquisitionDataBuilder()

    # Use small synthetic geometry to keep files light
    builder.header["!matrix size [1]"] = "4"
    builder.header["!matrix size [2]"] = "2"
    builder.header["!number of projections"] = "4"

    # Provide two windows and matching pixel data (4 projections total)
    builder.energy_windows = [
        {"lower": 110.0, "upper": 130.0},
        {"lower": 130.0, "upper": 150.0},
    ]
    builder.pixel_array = np.arange(1 * 4 * 4 * 2, dtype=np.float32).reshape(1, 4, 4, 2)

    outputs = builder.build_multi_energy(output_path_base=str(tmp_path / "multi"))

    assert builder.header["!number of projections"] == "2"
    assert len(outputs) == 2
    # Each stub should see half the projections
    for idx, stub in enumerate(outputs, start=1):
        assert stub.fill_data.shape == (1, 4, 2, 2)
        assert stub.write_calls == [str(tmp_path / f"multi_ew{idx}.hs")]

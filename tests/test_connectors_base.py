import inspect

import pytest

from sirf_simind_connection.connectors import (
    BaseConnector,
    NativeBackendConnector,
    SimindPythonConnector,
    SirfSimindAdaptor,
    StirSimindAdaptor,
)


@pytest.mark.unit
def test_connector_inheritance_hierarchy():
    assert issubclass(SimindPythonConnector, BaseConnector)
    assert issubclass(SirfSimindAdaptor, NativeBackendConnector)
    assert issubclass(StirSimindAdaptor, NativeBackendConnector)


@pytest.mark.unit
def test_native_backend_connector_forwards_quantization_scale(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class DummySimulator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "sirf_simind_connection.connectors.base.set_backend",
        lambda backend: None,
    )
    monkeypatch.setattr(
        "sirf_simind_connection.core.simulator.SimindSimulator",
        DummySimulator,
    )

    NativeBackendConnector(
        config_source="dummy.yaml",
        output_dir=str(tmp_path),
        backend="stir",
        quantization_scale=0.05,
    )

    assert captured["quantization_scale"] == pytest.approx(0.05)


@pytest.mark.unit
def test_adaptors_expose_quantization_scale_parameter():
    sirf_params = inspect.signature(SirfSimindAdaptor.__init__).parameters
    stir_params = inspect.signature(StirSimindAdaptor.__init__).parameters
    assert "quantization_scale" in sirf_params
    assert "quantization_scale" in stir_params

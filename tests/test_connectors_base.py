import inspect

import pytest

from py_smc.connectors import (
    BaseConnector,
    PyTomographySimindAdaptor,
    SimindPythonConnector,
    SirfSimindAdaptor,
    StirSimindAdaptor,
)


@pytest.mark.unit
def test_connector_inheritance_hierarchy():
    assert issubclass(SimindPythonConnector, BaseConnector)
    assert issubclass(SirfSimindAdaptor, BaseConnector)
    assert issubclass(StirSimindAdaptor, BaseConnector)
    assert issubclass(PyTomographySimindAdaptor, BaseConnector)


@pytest.mark.unit
def test_native_adaptors_expose_quantization_scale_parameter():
    sirf_params = inspect.signature(SirfSimindAdaptor.__init__).parameters
    stir_params = inspect.signature(StirSimindAdaptor.__init__).parameters
    assert "quantization_scale" in sirf_params
    assert "quantization_scale" in stir_params

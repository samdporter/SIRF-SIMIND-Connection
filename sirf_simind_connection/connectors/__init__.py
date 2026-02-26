"""
Connector/adaptor APIs layered over the SIMIND execution core.
"""

from .base import BaseConnector, NativeBackendConnector
from .python_connector import (
    NumpyConnector,
    ProjectionResult,
    RuntimeOperator,
    SimindPythonConnector,
)
from .pytomography_adaptor import PyTomographySimindAdaptor
from .sirf_adaptor import SirfSimindAdaptor
from .stir_adaptor import StirSimindAdaptor


__all__ = [
    "BaseConnector",
    "NativeBackendConnector",
    "NumpyConnector",
    "PyTomographySimindAdaptor",
    "ProjectionResult",
    "RuntimeOperator",
    "SirfSimindAdaptor",
    "SimindPythonConnector",
    "StirSimindAdaptor",
]

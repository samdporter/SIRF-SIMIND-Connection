.. _introduction:

Introduction
============

SIRF-SIMIND-Connection is a connector-first Python toolkit for SIMIND SPECT workflows.

The repository has two explicit goals:

1. Connect SIMIND to Python with a small, direct API.
2. Adapt SIMIND outputs to well-used Python reconstruction packages.

In practice this means:

- ``SimindPythonConnector`` runs SIMIND and returns NumPy-first outputs.
- ``StirSimindAdaptor`` bridges SIMIND outputs into STIR-native objects.
- ``SirfSimindAdaptor`` bridges SIMIND outputs into SIRF-native objects.
- ``PyTomographySimindAdaptor`` bridges SIMIND outputs into torch/PyTomography workflows.

Reconstruction algorithms are intentionally left to the target packages
(STIR, SIRF, PyTomography). The adaptor layer handles data conversion and
I/O boundaries, not reconstruction-method wrappers.

For axis and geometry conventions across these ecosystems, see :doc:`geometry`.

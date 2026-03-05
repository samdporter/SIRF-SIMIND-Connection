.. _introduction:

Introduction
============

py-smc is a Python toolkit for SIMIND SPECT workflows.

Disclaimer
----------

This project is independent and is **not affiliated with, endorsed by, or
maintained by** the SIMIND project or Lund University.

For users, the package provides two core capabilities:

1. Run SIMIND from Python through a small, direct API.
2. Adapt SIMIND outputs to well-used Python reconstruction packages.

In practice this means:

- ``SimindPythonConnector`` runs SIMIND and returns NumPy-first outputs.
- ``StirSimindAdaptor`` bridges SIMIND outputs into STIR-native objects.
- ``SirfSimindAdaptor`` bridges SIMIND outputs into SIRF-native objects.
- ``PyTomographySimindAdaptor`` bridges SIMIND outputs into torch/PyTomography workflows.

After generating projections with this package, run reconstruction directly in
STIR, SIRF, or PyTomography using each package's native reconstruction tools.

For axis and geometry conventions across these ecosystems, see :doc:`geometry`.

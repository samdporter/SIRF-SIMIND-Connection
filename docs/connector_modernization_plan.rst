:orphan:

Connector/Adaptor Modernization Plan
====================================

Target
------

Restructure the project into:

1. A pure Python SIMIND connector that works from SIMIND config files and runtime switches, then returns NumPy projection outputs plus their Interfile header/data paths.
2. Toolkit-specific adaptors for STIR, SIRF, and PyTomography that preserve native input/output object types.
3. Dedicated container images for STIR, SIRF, and PyTomography example workflows.

Execution Plan
--------------

Phase 1: Pure Python connector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Add ``SimindPythonConnector`` (no hard dependency on SIRF/STIR).
- Add ``RuntimeOperator`` for runtime switch/orbit-file control.
- Add direct Interfile-to-NumPy loader for projection outputs.
- Add unit tests for parsing and connector execution flow.

Phase 2: Framework adaptors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Add ``StirSimindAdaptor`` and ``SirfSimindAdaptor`` as native-object facades.
- Add ``PyTomographySimindAdaptor`` with explicit adapter hooks to avoid hard dependency coupling.
- Treat adaptor APIs as the primary native-backend public surface.

Phase 3: Containerized examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Add dedicated Dockerfiles for STIR, SIRF, and PyTomography workflows.
- Add ``docker/compose.yaml`` to simplify launching each environment.
- Provide a short container usage guide in ``docker/README.md``.

Current Status
--------------

- Phase 1: completed
- Phase 2: completed
- Phase 3: completed

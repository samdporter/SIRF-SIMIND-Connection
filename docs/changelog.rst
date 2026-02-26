.. _changelog:

Changelog
=========

All notable changes to this project are documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_, and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Unreleased
----------

[0.4.0] - 2026-02-26
--------------------

**Breaking Changes**
~~~~~~~~~~~~~~~~~~~~

**Example suite renumbering and adaptor split**
  The old monolithic OSEM examples were replaced with backend-specific adaptor
  examples:

  - ``07A_stir_adaptor_osem.py``
  - ``07B_sirf_adaptor_osem.py``
  - ``07C_pytomography_adaptor_osem.py``
  - ``08A/08B/08C`` DICOM-driven adaptor examples

  Legacy numbering and naming references should be updated to the new files.

**Connector/adaptor naming normalization**
  Public docs and examples now consistently use:

  - ``SimindPythonConnector`` for backend-agnostic NumPy workflows
  - ``StirSimindAdaptor`` / ``SirfSimindAdaptor`` / ``PyTomographySimindAdaptor``
    for backend-native workflows

Added
~~~~~

- Geometry documentation for SIMIND/STIR/SIRF/PyTomography axis conventions
  (``docs/geometry.rst``).
- Container scripts for split validation and examples:
  ``scripts/run_container_validation.sh`` and
  ``scripts/run_container_examples.sh``.
- Backend-isolation and connector/adaptor unit tests (markers + import gating).
- Example profile config (``configs/Example.yaml``) and Python-connector helper
  flow for lightweight deterministic runs.

Changed
~~~~~~~

- Core examples 01-06 now use the pure Python connector path.
- OSEM examples now use dedicated adaptor paths and backend-specific outputs.
- Documentation updated for new example numbering, runtime flags, and testing
  entry points.
- Versioning aligned across package metadata and docs.

Fixed
~~~~~

- PENETRATE component header parsing via ``.bXX`` outputs in the Python
  connector.
- SIMIND optional-availability handling in container scripts (fail-fast option
  plus skip-by-default behavior when binary is absent).
- Quantization scaling docs and naming consistency (``quantization_scale``).

[0.3.0] - 2025-01-23
--------------------

**Breaking Changes**
~~~~~~~~~~~~~~~~~~~~

**Backend-Agnostic Public API**
  All public methods now return wrapped backend-agnostic objects instead of native SIRF/STIR objects:

  - ``SimindSimulator.get_total_output()`` → Returns ``AcquisitionDataInterface``
  - ``SimindSimulator.get_scatter_output()`` → Returns ``AcquisitionDataInterface``
  - ``SimindSimulator.get_primary_output()`` → Returns ``AcquisitionDataInterface``
  - ``SimindSimulator.get_penetrate_output()`` → Returns ``AcquisitionDataInterface``
  - ``SimindToStirConverter.convert(return_object=True)`` → Returns ``AcquisitionDataInterface``
  - ``convert_simind_to_stir()`` → Returns ``AcquisitionDataInterface``

  **Migration**: Use ``.native_object`` property to access underlying SIRF/STIR objects if needed for backend-specific operations. Wrapper objects support common operations (``.write()``, ``.sum()``, ``.as_array()``, etc.).

**Backend-Agnostic extract_attributes_from_stir()**
  Function now only accepts filepath (string), not acquisition data objects:

  - **Before**: ``extract_attributes_from_stir(acq_obj)`` or ``extract_attributes_from_stir(filepath)``
  - **After**: ``extract_attributes_from_stir(filepath)`` only
  - **Migration**: If you have an acquisition object, write it to a file first: ``acq_obj.write(temp_path); extract_attributes_from_stir(temp_path)``

Changed
~~~~~~~

**Enhanced Factory Functions**
  Factory functions now accept multiple input types:

  - String filepath → Loads and wraps
  - Already wrapped object → Returns as-is (idempotent)
  - Native SIRF/STIR object → Wraps it
  - ``None`` → Creates empty object (SIRF only)

**Internal Template Storage**
  ``SimindSimulator`` now stores both template filepath (backend-agnostic) and wrapped object for efficient operations.

Fixed
~~~~~

**Critical: Coordinator Update Timing Fix**
  Fixed off-by-one error in coordinator update scheduling:

  - **Issue**: Coordinator updates were occurring at iteration N, but callbacks (UpdateEtaCallback, ArmijoTriggerCallback) ran AFTER the step was taken, causing Armijo line search to run one iteration late with a mismatched objective function
  - **Fix**: Modified ``SimindCoordinator.should_update()`` and ``StirPsfCoordinator.should_update()`` to trigger updates ONE iteration early (at ``correction_update_interval - 1`` instead of ``correction_update_interval``)
  - **Impact**: Ensures proper callback ordering for convergent optimization
  - **Result**: Eliminates divergence caused by taking steps with wrong objective, ensures convexity is maintained

Added
~~~~~

**Backend Abstraction Layer**
  New backend system supporting both SIRF and STIR Python:

  - Factory functions: ``create_image_data()``, ``create_acquisition_data()``
  - Utility functions: ``get_backend()``, ``set_backend()``, ``is_sirf_backend()``
  - Wrapper classes: ``SirfImageData``, ``SirfAcquisitionData``, ``StirImageData``, ``StirAcquisitionData``
  - See :doc:`backends` for complete documentation

**Utility Functions for STIR Compatibility**
  - ``to_projdata_in_memory()``: Convert ProjData to ProjDataInMemory for arithmetic operations (STIR Python compatibility)
  - Fixes arithmetic operations (``+``, ``-``, ``*``) on STIR ProjData objects
  - Works transparently with both SIRF and STIR backends

**Example Updates**
  - Updated examples 01-06 to support both SIRF and STIR Python backends
  - Added ``--backend`` command-line option to force specific backend
  - Fixed arithmetic operations in example 03 (TEW correction)

**ShiftedKullbackLeibler Support**
  - Added conditional support for SETR's ``ShiftedKullbackLeibler`` in data partitioning
  - New ``use_shifted_kl`` parameter in ``partition_data_with_cil_objectives()`` function
  - Automatic fallback to standard ``KullbackLeibler`` when SETR is not available

**ResidualCorrectedKullbackLeibler Jordan Split**
  - Residual-aware KL now decomposes residuals into positive/negative parts
  - Keeps the objective convex and allows arbitrary residual signs

Changed
~~~~~~~

**Documentation Organization**
  - Moved BACKENDS.md to docs/backends.rst (reStructuredText format)
  - Updated docs/changelog.rst with comprehensive change history
  - Added backends page to documentation index

Version 0.2.1
-------------

New Features
~~~~~~~~~~~~

**Schneider2000 Density Conversion**
  Advanced 44-segment piecewise model for HU-to-density conversion:

  - ``hu_to_density_schneider()``: Interpolated conversion using all 44 tissue segments
  - ``hu_to_density_schneider_piecewise()``: Exact piecewise conversion matching lookup table
  - ``get_schneider_tissue_info()``: Lookup tissue information for specific HU values
  - ``compare_density_methods()``: Compare bilinear vs Schneider methods
  - Enhanced Accuracy: ~0.17-0.19 g/cm³ improved accuracy over bilinear model
  - Comprehensive Tissue Support: Covers air, lung variations, soft tissues, bones, and metal implants
  - New Example: ``06_schneider_density_conversion.py`` demonstrates advanced density conversion

**Coordinator Architecture**
  Efficient subset reconstruction with shared Monte Carlo corrections:

  - ``SimindCoordinator``: Manages single SIMIND simulation for all subsets
  - ``SimindSubsetProjector``: Projector for individual subsets
  - CIL partitioner utilities for subset-based reconstruction
  - 12× faster for 12-subset reconstruction
  - See coordinator-related API/docs sections for implementation details

**SimindProjector AcquisitionModel Interface**
  Drop-in replacement for SIRF AcquisitionModel with Monte Carlo corrections:

  - Complete AcquisitionModel API compatibility
  - Three correction modes (residual only, additive only, both)
  - Automatic iteration tracking
  - Intelligent scaling strategy
  - CIL framework compatibility

Improvements
~~~~~~~~~~~~

- Extended documentation with density conversion methods comparison
- Comprehensive test suite for Schneider functionality (16 new tests)
- Enhanced attenuation conversion utilities with clinical-grade accuracy
- MPI parallel execution support for SIMIND simulations
- STIR CUDARelativeDifferencePrior integration

Fixed
~~~~~

**Critical: Non-Circular Orbit Support**
  Multiple fixes for proper orbit file handling:

  - Issue 1: Missing orbit file creation (AcquisitionData.get_info() doesn't include orbit/radii)
  - Issue 2: Missing unit conversion (radii in mm instead of cm)
  - Issue 3: File naming collision (input orbit file overwritten)
  - Issue 4: Incorrect command argument order
  - Issue 5: Full path instead of filename

**Critical: Subset View Index Out of Range**
  - Fixed coordinator receiving subset acquisition models but attempting to extract using full-data indices
  - Coordinator now requires full-data acquisition models passed to ``__init__()``

**Module Import Issues**
  - Fixed RecursionError in lazy imports (``converters``, ``utils``, ``builders`` modules)
  - Wrapped SIRF imports in try/except blocks for better fallback behavior

Version 0.2.0
-------------

Breaking Changes
~~~~~~~~~~~~~~~~

- Modified config file loading mechanism in SimulationConfig class
- Updated API for configuration initialization

New Features
~~~~~~~~~~~~

- Comprehensive test suite with unit and integration tests
- Enhanced documentation with ReadTheDocs support
- GitHub Actions CI/CD pipeline
- Auto-generated API documentation
- Professional documentation structure

Improvements
~~~~~~~~~~~~

- Better test coverage for all components
- Improved code quality with automated checks
- Streamlined README for better user experience

Version 0.1.1
-------------

- Bug fixes and minor improvements from initial release

Version 0.1.0
-------------

- Initial release with core functionalities and examples

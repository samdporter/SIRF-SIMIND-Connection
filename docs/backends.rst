Backend Abstraction Layer
=========================

For geometry/axis conventions across SIMIND, STIR/SIRF, and PyTomography,
see :doc:`geometry`.

This module provides a unified interface for working with both **SIRF** and **STIR Python** libraries, allowing users to choose their preferred backend for image reconstruction and simulation tasks.

Overview
--------

The backend system automatically detects which library is available and uses it transparently. Users can also manually select a backend if both libraries are installed.

Supported Backends
~~~~~~~~~~~~~~~~~~

- **SIRF** (``sirf.STIR``): Full-featured SIRF interface with all reconstruction capabilities
- **STIR Python** (``stir`` + ``stirextra``): Direct STIR Python bindings

Quick Start
-----------

Automatic Backend Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sirf_simind_connection import SimindSimulator
    from sirf_simind_connection.backends import get_backend

    # Automatically uses SIRF if available, otherwise STIR Python
    print(f"Using backend: {get_backend()}")

    # Use the simulator normally - backend is handled internally
    simulator = SimindSimulator(config, output_dir="output")

Manual Backend Selection
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sirf_simind_connection.backends import set_backend

    # Force STIR Python backend
    set_backend("stir")

    # Now all operations will use STIR Python
    from sirf_simind_connection import SimindSimulator
    simulator = SimindSimulator(config, output_dir="output")

API Reference
-------------

Factory Functions
~~~~~~~~~~~~~~~~~

``get_backend() -> str``
^^^^^^^^^^^^^^^^^^^^^^^^

Returns the current backend ("sirf" or "stir"). Auto-detects if not set.

``set_backend(backend: str) -> None``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Manually set the backend. Options: "sirf" or "stir".

``create_image_data(filepath: str = None) -> ImageDataInterface``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create an image data object using the current backend.

.. code-block:: python

    from sirf_simind_connection.backends import create_image_data

    # Load from file
    img = create_image_data("phantom.hv")

    # Access data
    arr = img.as_array()  # Works with both backends
    img.write("output.hv")

``create_acquisition_data(filepath: str = None) -> AcquisitionDataInterface``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create an acquisition data object using the current backend.

.. code-block:: python

    from sirf_simind_connection.backends import create_acquisition_data

    # Load from file
    acq = create_acquisition_data("projections.hs")

    # Access data
    arr = acq.as_array()  # Works with both backends
    acq.write("output.hs")

``unwrap(obj) -> native_object``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get the underlying native SIRF or STIR object from a wrapped object.

.. code-block:: python

    from sirf_simind_connection.backends import unwrap

    wrapped_img = create_image_data("phantom.hv")
    native_img = unwrap(wrapped_img)  # Returns ImageData or FloatVoxelsOnCartesianGrid

Utility Functions
~~~~~~~~~~~~~~~~~

``is_sirf_backend() -> bool``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Check if SIRF backend is active.

``is_stir_backend() -> bool``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Check if STIR Python backend is active.

``reset_backend() -> None``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reset backend selection to allow re-detection.

Feature Compatibility
---------------------

.. list-table::
   :header-rows: 1
   :widths: 40 20 30

   * - Feature
     - SIRF Required
     - STIR Python Supported
   * - Basic simulation (examples 01-06)
     - No
     - ✅ Yes
   * - SCATTWIN scoring
     - No
     - ✅ Yes
   * - PENETRATE scoring
     - No
     - ✅ Yes
   * - File I/O (Interfile format)
     - No
     - ✅ Yes
   * - Array conversion
     - No
     - ✅ Yes
   * - **CIL integration**
     - ✅ Yes
     - ❌ No
   * - **Coordinator architecture**
     - ✅ Yes
     - ❌ No
   * - **SimindProjector**
     - ✅ Yes
     - ❌ No
   * - **OSEM via STIR adaptor (example 07A)**
     - ❌ No
     - ✅ Yes
   * - **OSEM via SIRF adaptor (example 07B)**
     - ✅ Yes
     - ❌ No
   * - **OSEM via PyTomography adaptor (example 07C)**
     - ❌ No
     - ❌ No (requires PyTomography)

Key Differences Between Backends
---------------------------------

Array Conversion
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Both backends support:
    from sirf_simind_connection.utils import get_array
    arr = get_array(image_obj)  # Works with both!

    # Internally:
    # - SIRF uses: obj.asarray() or obj.as_array()
    # - STIR uses: stirextra.to_numpy(obj)

File Writing
~~~~~~~~~~~~

.. code-block:: python

    # Unified interface:
    img.write("output.hv")

    # Internally:
    # - SIRF uses: obj.write(filepath)
    # - STIR uses: obj.write_to_file(filepath)

Object Construction
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Unified interface:
    img = create_image_data("input.hv")
    acq = create_acquisition_data("input.hs")

    # Internally:
    # - SIRF: ImageData(filepath), AcquisitionData(filepath)
    # - STIR: FloatVoxelsOnCartesianGrid.read_from_file(filepath),
    #         ProjData.read_from_file(filepath)

Implementation Notes
--------------------

STIR Python Limitations
~~~~~~~~~~~~~~~~~~~~~~~~

Some operations are not directly supported with STIR Python backend:

1. **Creating empty objects**: STIR requires geometry information

   .. code-block:: python

       # This works with SIRF but raises NotImplementedError with STIR:
       img = create_image_data()  # No filepath

2. **Filling with arrays**: STIR's ProjData requires segment-by-segment operations

   .. code-block:: python

       # This works with SIRF but raises NotImplementedError with STIR:
       acq.fill(numpy_array)  # Works with scalars only for STIR

3. **Element-wise operations**: Not all SIRF methods are available in STIR

   .. code-block:: python

       # SIRF only:
       img.maximum(0)  # Clip negative values

Interfile Format Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both backends read and write the same Interfile format, so files are **fully interchangeable**:

.. code-block:: python

    # Create with SIRF
    set_backend("sirf")
    img = create_image_data()
    img.fill(1.0)
    img.write("test.hv")

    # Read with STIR
    set_backend("stir")
    img2 = create_image_data("test.hv")
    arr = img2.as_array()  # Works!

Examples
--------

Running Examples with Different Backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Core Python-connector examples (backend-agnostic NumPy path)
    python examples/01_basic_simulation.py

    # STIR-native adaptor + STIR OSEM
    python examples/07A_stir_adaptor_osem.py

    # SIRF-native adaptor + SIRF OSEM
    python examples/07B_sirf_adaptor_osem.py

    # PyTomography-native adaptor + PyTomography OSEM
    python examples/07C_pytomography_adaptor_osem.py

Testing
-------

The backend system includes comprehensive tests:

.. code-block:: bash

    # Test backend auto-detection
    pytest tests/test_backends.py::test_auto_detection

    # Test array conversion with both backends
    pytest tests/test_backends.py::test_array_conversion

    # Test file I/O compatibility
    pytest tests/test_backends.py::test_file_io_compatibility

Troubleshooting
---------------

ImportError: Neither SIRF nor STIR Python found
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install one of the supported backends:

.. code-block:: bash

    # Option 1: Install SIRF (recommended for full features)
    # Follow instructions at: https://github.com/SyneRBI/SIRF

    # Option 2: Install STIR Python
    # Follow instructions at: https://github.com/UCL/STIR

Backend not switching
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Force reset if backend seems stuck
    from sirf_simind_connection.backends import reset_backend
    reset_backend()
    set_backend("stir")  # Now it will switch

Feature not available with STIR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check the compatibility table above. Some features require SIRF:

- CIL integration
- Coordinator architecture
- SimindProjector/coordinator workflows

STIR-native OSEM is supported via ``examples/07A_stir_adaptor_osem.py``.
For SIRF-only features, use SIRF backend:

.. code-block:: python

    set_backend("sirf")

Architecture
------------

The backend system uses an adapter pattern::

    User Code
        ↓
    Backend Factory (auto-detect or manual)
        ↓
    ┌─────────────────┬─────────────────┐
    │  SIRF Backend   │  STIR Backend   │
    │  (wrappers)     │  (wrappers)     │
    └─────────────────┴─────────────────┘
        ↓                     ↓
    sirf.STIR           stir + stirextra

All wrappers implement ``ImageDataInterface`` and ``AcquisitionDataInterface`` for consistent API.

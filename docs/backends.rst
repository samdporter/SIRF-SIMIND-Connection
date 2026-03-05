Backend and Adaptor Dependencies
================================

Use ``SimindPythonConnector`` for direct SIMIND execution from Python. Use an
adaptor when you want outputs returned as native STIR/SIRF/PyTomography types.

Dependency Matrix
-----------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Component
     - Required dependency
   * - ``SimindPythonConnector``
     - SIMIND only (no SIRF/STIR/PyTomography requirement)
   * - ``StirSimindAdaptor``
     - STIR Python (``stir``)
   * - ``SirfSimindAdaptor``
     - SIRF (``sirf.STIR``)
   * - ``PyTomographySimindAdaptor``
     - PyTomography + torch

The adaptors are responsible for converting input/output object types across
package boundaries. Reconstruction-system objects (for example, a PyTomography
system matrix) should be created directly in the target package.

Backend Abstraction Module
--------------------------

The ``sirf_simind_connection.backends`` module provides optional helper APIs for
working with SIRF/STIR image and acquisition wrapper objects.

Quick example:

.. code-block:: python

    from sirf_simind_connection.backends import get_backend, set_backend

    backend = get_backend()  # auto-detect: "sirf" or "stir"
    print(f"Using backend: {backend}")

    # Optional explicit override
    set_backend("stir")

Main helper functions include:

- ``get_backend()``
- ``set_backend(backend)``
- ``reset_backend()``
- ``create_image_data(...)``
- ``create_acquisition_data(...)``
- ``unwrap(...)``

These helpers are independent from the connector/adaptor execution API.

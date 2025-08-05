.. _api:

API Documentation
=================

Core Modules
------------

.. automodule:: sirf_simind_connection
   :members:
   :undoc-members:
   :show-inheritance:

Simulator and Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sirf_simind_connection.core.simulator
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sirf_simind_connection.core.config
   :members:
   :undoc-members:
   :show-inheritance:

Projector
~~~~~~~~~

.. automodule:: sirf_simind_connection.core.projector
   :members:
   :undoc-members:
   :show-inheritance:

Converters
----------

Attenuation and Density Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sirf_simind_connection.converters.attenuation
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions:

* :func:`~sirf_simind_connection.converters.attenuation.hu_to_density` - Traditional bilinear HU-to-density conversion
* :func:`~sirf_simind_connection.converters.attenuation.hu_to_density_schneider` - Advanced Schneider2000 interpolated conversion
* :func:`~sirf_simind_connection.converters.attenuation.hu_to_density_schneider_piecewise` - Exact Schneider2000 piecewise conversion
* :func:`~sirf_simind_connection.converters.attenuation.get_schneider_tissue_info` - Tissue information lookup
* :func:`~sirf_simind_connection.converters.attenuation.compare_density_methods` - Method comparison utility

DICOM and SIMIND Converters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sirf_simind_connection.converters.dicom_to_stir
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sirf_simind_connection.converters.simind_to_stir
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: sirf_simind_connection.utils.stir_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sirf_simind_connection.utils.simind_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sirf_simind_connection.utils.io_utils
   :members:
   :undoc-members:
   :show-inheritance:

Explore the different modules and their functionalities. This section is auto-generated using Sphinx autodoc.

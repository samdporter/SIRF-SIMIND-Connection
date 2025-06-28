SIRF-SIMIND-Connection Documentation
===============================

Welcome to the documentation for SIRF-SIMIND-Connection, a Python wrapper that provides
seamless integration between SIRF (Synergistic Image Reconstruction Framework) and
SIMIND Monte Carlo simulator for SPECT imaging applications.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide
   api_reference
   examples
   contributing

Overview
--------

SIRF-SIMIND-Connection enables researchers and developers to:

- Run Monte Carlo SPECT simulations using familiar SIRF data types
- Convert SPECT DICOM files to STIR format for processing
- Simulate scatter and attenuation effects accurately
- Implement advanced scatter correction techniques
- Create custom scanner configurations

Key Features
------------

**Monte Carlo Simulation**
   Leverage SIMIND's accurate photon transport modeling with SIRF's reconstruction framework

**Multi-Energy Window Support**
   Simulate and process multiple energy windows for scatter correction techniques like TEW and DEW

**Flexible Configuration**
   Use YAML files for easy configuration management and version control

**DICOM Support**
   Convert clinical SPECT DICOM data to STIR format automatically

**Pre-configured Templates**
   Start quickly with templates for common SPECT scanners

Installation
------------

See the :doc:`installation` guide for detailed instructions.

Quick Start
-----------

.. code-block:: python

   from sirf_simind_spect import SimindSimulator
   from sirf.STIR import ImageData

   # Load data
   source = ImageData('phantom.hv')
   mu_map = ImageData('attenuation.hv')

   # Create simulator
   sim = SimindSimulator('config.smc', 'output/', source=source, mu_map=mu_map)
   
   # Set energy window
   sim.set_windows([126], [154], [5])
   
   # Run simulation
   sim.run_simulation()
   
   # Get results
   sinogram = sim.get_output_total()

Getting Help
------------

- **Issues**: Report bugs or request features on `GitHub Issues <https://github.com/yourusername/SIRF-SIMIND-Connection/issues>`_
- **Discussions**: Join conversations on `GitHub Discussions <https://github.com/yourusername/SIRF-SIMIND-Connection/discussions>`_
- **Email**: Contact the maintainers

License
-------

This project is licensed under the Apache License 2.0.

Citation
--------

If you use this software in your research, please cite:

.. code-block:: bibtex

   @software{sirf_simind_spect,
     author = {Porter, Sam and Varzakis, Efstathios},
     title = {SIRF-SIMIND-Connection: A Python wrapper for SPECT Monte Carlo simulations},
     year = {2024},
     url = {https://github.com/yourusername/SIRF-SIMIND-Connection}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
.. _examples:

Examples
========

See the following examples for comprehensive use:

1. **Basic Simulation** - ``examples/01_basic_simulation.py``
   
   Learn how to set up and run simple SPECT simulations with phantom data.

2. **DICOM Conversion** - ``examples/02_dicom_conversion.py``
   
   Convert DICOM files to STIR format for use in simulations.

3. **Multi-Energy Windows** - ``examples/03_multi_window.py``
   
   TEW (Triple Energy Window) scatter correction method demonstration.

4. **Custom Configurations** - ``examples/04_custom_config.py``
   
   Create and use custom YAML configuration files for specialized setups.

5. **Scoring Routine Comparison** - ``examples/05_scattwin_vs_penetrate_comparison.py``
   
   Compare SCATTWIN and PENETRATE Monte Carlo scoring routines.

6. **Schneider Density Conversion** - ``examples/06_schneider_density_conversion.py``
   
   **NEW**: Demonstrate advanced HU-to-density conversion using the Schneider2000 model.
   
   Features:
   
   - Compare bilinear vs Schneider conversion methods
   - Visualize density differences across tissue types  
   - Create simulated CT phantom with multiple tissue types
   - Demonstrate tissue information lookup functionality

Running Examples
----------------

Each example can be run individually:

.. code-block:: bash

    python examples/01_basic_simulation.py
    python examples/06_schneider_density_conversion.py

Or run all examples sequentially:

.. code-block:: bash

    cd scripts/
    python run_all_examples.py  # Run all examples sequentially

Output Files
------------

Each example creates output in its own directory under ``output/``:

- ``output/basic_simulation/`` - Basic simulation results
- ``output/schneider_density/`` - Density conversion plots and data
- ``output/multi_window/`` - TEW correction results
- ``output/custom_configs/`` - Configuration examples
- ``output/routine_comparison/`` - Scoring routine comparison


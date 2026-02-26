.. _examples:

Examples
========

For cross-backend axis and orientation details, see :doc:`geometry`.

See the following examples for comprehensive use:

1. **Basic Simulation** - ``examples/01_basic_simulation.py``
   
   Run a basic SIMIND simulation with NumPy voxel input and NumPy projection output.
   Uses Tc-99m settings (``FI=tc99m``) with an ``ma-lehr`` collimator and a
   126-154 keV photopeak window.

2. **Runtime-Switch Comparison** - ``examples/02_runtime_switch_comparison.py``
   
   Compare projection outputs under different SIMIND runtime switch settings.

3. **Multi-Energy Windows** - ``examples/03_multi_window.py``
   
   TEW (Triple Energy Window) scatter-estimation demonstration using NumPy projections.
   Uses Lu-177 settings (``FI=lu177``) with TEW windows 166-187, 187-229, 229-250 keV.

4. **Custom Configurations** - ``examples/04_custom_config.py``
   
   Create and use custom YAML configuration files for specialized setups.

5. **Scoring Routine Comparison** - ``examples/05_scattwin_vs_penetrate_comparison.py``
   
   Compare SCATTWIN and PENETRATE Monte Carlo scoring routines using the Python Connector.

6. **Schneider Density Conversion** - ``examples/06_schneider_density_conversion.py``

   Demonstrate advanced HU-to-density conversion using the Schneider2000 model.

   Features:

   - Compare bilinear vs Schneider conversion methods
   - Visualize density differences across tissue types
   - Create simulated CT phantom with multiple tissue types
   - Demonstrate tissue information lookup functionality

7A. **STIR Adaptor OSEM** - ``examples/07A_stir_adaptor_osem.py``

   Simulate with the STIR adaptor and run STIR OSEM reconstruction in one Python workflow.
   Uses ``configs/Example.yaml`` plus Y-90 runtime switches (``FI`` and ``CC``).

7B. **SIRF Adaptor OSEM** - ``examples/07B_sirf_adaptor_osem.py``

   Simulate with the SIRF adaptor and run SIRF OSEM reconstruction in one Python workflow.
   Uses ``configs/Example.yaml`` plus Y-90 runtime switches (``FI`` and ``CC``).

7C. **PyTomography Adaptor OSEM** - ``examples/07C_pytomography_adaptor_osem.py``

   Simulate with the PyTomography adaptor and run PyTomography OSEM in one Python workflow.
   Uses ``configs/Example.yaml`` plus Y-90 runtime switches (``FI`` and ``CC``).

Running Examples
----------------

Each example can be run individually:

.. code-block:: bash

    python examples/01_basic_simulation.py
    python examples/02_runtime_switch_comparison.py
    python examples/03_multi_window.py
    python examples/04_custom_config.py
    python examples/05_scattwin_vs_penetrate_comparison.py
    python examples/06_schneider_density_conversion.py
    python examples/07A_stir_adaptor_osem.py
    python examples/07B_sirf_adaptor_osem.py
    python examples/07C_pytomography_adaptor_osem.py

Or run all examples sequentially:

.. code-block:: bash

    cd scripts/
    python run_all_examples.py  # Run all examples sequentially

Or run one backend-specific adaptor example inside each isolated Docker image:

.. code-block:: bash

    bash scripts/run_container_examples.sh

Output Files
------------

Each example creates output in its own directory under ``output/``:

- ``output/basic_simulation/`` - Basic simulation results
- ``output/runtime_switch_comparison/`` - Runtime-switch comparison results
- ``output/schneider_density/`` - Density conversion plots and data
- ``output/multi_window/`` - TEW correction results
- ``output/custom_configs/`` - Configuration examples
- ``output/routine_comparison/`` - Scoring routine comparison
- ``output/stir_adaptor_osem/`` - STIR adaptor simulation + STIR OSEM reconstruction
- ``output/sirf_adaptor_osem/`` - SIRF adaptor simulation + SIRF OSEM reconstruction
- ``output/pytomography_adaptor_osem/`` - PyTomography adaptor simulation + PyTomography OSEM reconstruction

Each OSEM example also writes a summary plot with:
- input source slice
- one projection view
- reconstructed image slice

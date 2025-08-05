.. _usage:

Usage Guide
============

Getting Started
----------------

A quick example to get started with SIRF-SIMIND-Connection:

.. code-block:: python

    from sirf_simind_connection import SimindSimulator, SimulationConfig
    from sirf_simind_connection.configs import get
    from sirf_simind_connection.utils.stir_utils import create_simple_phantom, create_attenuation_map

    # Create phantom and attenuation map
    phantom = create_simple_phantom()
    mu_map = create_attenuation_map(phantom)

    # Load pre-configured scanner settings
    config = SimulationConfig(get("AnyScan.yaml"))
    simulator = SimindSimulator(config, output_dir='output')

    # Set inputs and run
    simulator.set_source(phantom)
    simulator.set_mu_map(mu_map)
    simulator.set_energy_windows([126], [154], [0])  # Tc-99m ± 10%
    simulator.run_simulation()

    result = simulator.get_total_output(window=1)
    print("Simulation completed successfully.")

Density Conversion
------------------

The package provides advanced Hounsfield Unit (HU) to density conversion methods:

Traditional Bilinear Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sirf_simind_connection.converters.attenuation import hu_to_density
    import numpy as np

    # Simple 3-point model (air, water, bone)
    hu_image = np.array([[-1000, 0, 500], [800, 1200, 2000]])
    density_bilinear = hu_to_density(hu_image)

Advanced Schneider2000 Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sirf_simind_connection.converters.attenuation import (
        hu_to_density_schneider,
        hu_to_density_schneider_piecewise,
        get_schneider_tissue_info,
        compare_density_methods
    )

    # Advanced 44-segment piecewise model
    density_schneider = hu_to_density_schneider(hu_image)
    density_piecewise = hu_to_density_schneider_piecewise(hu_image)

    # Lookup tissue information
    tissue_info = get_schneider_tissue_info(50)  # HU = 50
    print(f"Tissue: {tissue_info['name']}")
    print(f"Density: {tissue_info['density_g_cm3']:.3f} g/cm³")

    # Compare methods
    comparison = compare_density_methods(hu_image)
    print(f"Mean difference: {comparison['mean_diff_interp']:.3f} g/cm³")

**Key Advantages of Schneider Model:**

- **44 tissue segments** vs 3 points (bilinear)
- **Clinically validated densities** from Schneider et al. 2000
- **Better accuracy** especially for lung and bone regions
- **Metal implant support** - handles dental materials and implants
- **~0.17-0.19 g/cm³ improved accuracy** over bilinear model

Detailed Use Cases
--------------------

1. **Basic Simulation** - Learn how to set up and run simple simulations.
2. **Advanced Configuration** - Using custom YAML configurations.
3. **Density Conversion** - Choose between bilinear and Schneider models for HU-to-density conversion.
4. **Extensive Output Analysis** - Understand the output from SCATTWIN vs PENETRATE routines.

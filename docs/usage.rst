.. _usage:

Usage Guide
===========

Quick Start
-----------

Use ``SimindPythonConnector`` when you want direct Python control of SIMIND
inputs/outputs without any reconstruction-package dependency.

.. code-block:: python

    import numpy as np
    from sirf_simind_connection import SimindPythonConnector
    from sirf_simind_connection.configs import get

    source = np.zeros((32, 32, 32), dtype=np.float32)  # z, y, x
    source[12:20, 12:20, 12:20] = 1.0
    mu_map = np.zeros_like(source)
    mu_map[source > 0] = 0.15

    connector = SimindPythonConnector(
        config_source=get("Example.yaml"),
        output_dir="output/basic",
        output_prefix="case01",
        quantization_scale=0.05,
    )

    connector.configure_voxel_phantom(
        source=source,
        mu_map=mu_map,
        voxel_size_mm=4.0,
    )
    connector.set_energy_windows([126], [154], [0])
    connector.add_runtime_switch("FI", "tc99m")
    connector.add_runtime_switch("CC", "ma-lehr")
    connector.add_runtime_switch("NN", 1)
    connector.add_runtime_switch("RR", 12345)

    outputs = connector.run()
    total = outputs["tot_w1"].projection
    print(total.shape)

Map Input Types
---------------

``configure_voxel_phantom()`` supports three ``mu_map`` input conventions via
``mu_map_type``:

- ``"attenuation"``: linear attenuation coefficients (cm^-1), converted to
  density before SIMIND input writing.
- ``"density"``: density map in g/cm^3, passed through directly.
- ``"hu"``: CT HU map, converted to density with the Schneider model.

.. code-block:: python

    connector.configure_voxel_phantom(
        source=source,
        mu_map=mu_map,
        voxel_size_mm=4.0,
        mu_map_type="attenuation",  # or "density" / "hu"
    )

Adaptor Workflows
-----------------

Use adaptors when you want connector-managed SIMIND execution plus native
objects for a target reconstruction package.

STIR adaptor
~~~~~~~~~~~~

.. code-block:: python

    from sirf_simind_connection import StirSimindAdaptor
    from sirf_simind_connection.configs import get

    adaptor = StirSimindAdaptor(
        config_source=get("Example.yaml"),
        output_dir="output/stir_adaptor",
        output_prefix="stir_case01",
        mu_map_type="attenuation",  # or "density" / "hu"
    )
    adaptor.set_source(stir_source)
    adaptor.set_mu_map(stir_mu_map)
    adaptor.set_energy_windows([75], [225], [0])
    adaptor.add_runtime_switch("FI", "y90_tissue")
    adaptor.add_runtime_switch("CC", "ma-megp")
    adaptor.add_runtime_switch("RR", 12345)

    outputs = adaptor.run()
    stir_total = outputs["tot_w1"]

SIRF adaptor
~~~~~~~~~~~~

.. code-block:: python

    from sirf_simind_connection import SirfSimindAdaptor
    from sirf_simind_connection.configs import get

    adaptor = SirfSimindAdaptor(
        config_source=get("Example.yaml"),
        output_dir="output/sirf_adaptor",
        output_prefix="sirf_case01",
        mu_map_type="attenuation",  # or "density" / "hu"
    )
    adaptor.set_source(sirf_source)
    adaptor.set_mu_map(sirf_mu_map)
    adaptor.set_energy_windows([75], [225], [0])
    adaptor.add_runtime_switch("FI", "y90_tissue")
    adaptor.add_runtime_switch("CC", "ma-megp")
    adaptor.add_runtime_switch("RR", 12345)

    outputs = adaptor.run()
    sirf_total = outputs["tot_w1"]

PyTomography adaptor
~~~~~~~~~~~~~~~~~~~~

The adaptor returns PyTomography-compatible tensors and output headers.
Build the system matrix directly with PyTomography APIs.

.. code-block:: python

    import torch
    from pytomography.io.SPECT import simind as pytomo_simind
    from pytomography.projectors.SPECT import SPECTSystemMatrix

    from sirf_simind_connection import PyTomographySimindAdaptor
    from sirf_simind_connection.configs import get

    adaptor = PyTomographySimindAdaptor(
        config_source=get("Example.yaml"),
        output_dir="output/pytomo_adaptor",
        output_prefix="pytomo_case01",
        mu_map_type="attenuation",  # or "density" / "hu"
    )
    adaptor.set_source(source_tensor_xyz)
    adaptor.set_mu_map(mu_tensor_xyz)
    adaptor.set_energy_windows([75], [225], [0])
    adaptor.add_runtime_switch("FI", "y90_tissue")
    adaptor.add_runtime_switch("CC", "ma-megp")
    adaptor.add_runtime_switch("RR", 12345)
    adaptor.run()

    projections = adaptor.get_total_output(window=1).to(dtype=torch.float32)
    header = adaptor.get_output_header_path("tot_w1")
    object_meta, proj_meta = pytomo_simind.get_metadata(str(header))

    system_matrix = SPECTSystemMatrix(
        obj2obj_transforms=[],
        proj2proj_transforms=[],
        object_meta=object_meta,
        proj_meta=proj_meta,
    )

Density Conversion
------------------

The package includes HU-to-density conversion utilities, including the
Schneider2000 model.

.. code-block:: python

    import numpy as np
    from sirf_simind_connection.converters.attenuation import hu_to_density_schneider

    hu_image = np.array([[-1000, 0, 500], [800, 1200, 2000]])
    density_map = hu_to_density_schneider(hu_image)

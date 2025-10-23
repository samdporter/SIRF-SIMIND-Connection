.. _coordinator_architecture:

SimindCoordinator Architecture
===============================

Overview
--------

The coordinator-based architecture enables efficient SIMIND Monte Carlo corrections in subset-based iterative reconstruction algorithms. Key improvements:

1. **One MPI-parallelized SIMIND simulation** for all subsets (instead of N separate simulations)
2. **CIL-compatible objectives** using ``KullbackLeibler`` with ``OperatorCompositionFunction``
3. **Automatic residual scaling** to match SIRF's subset scaling
4. **Global iteration tracking** across all subsets for consistent update intervals

Architecture Components
-----------------------

1. SimindCoordinator
~~~~~~~~~~~~~~~~~~~~

**Module**: :mod:`sirf_simind_connection.core.coordinator`

**Purpose**: Shared coordinator managing SIMIND simulations across all subset projectors.

**Key Features**:

- **Algorithm iteration tracking**: Uses CIL algorithm's ``iteration`` counter to track progress
- **Periodic full simulation**: Runs SIMIND (all views) when ``algorithm.iteration - last_update_iteration >= correction_update_interval``
- **Result caching**: Stores b01, b02, scale_factor, and cache_version
- **Subset distribution**: Extracts subset-specific views and distributes corrections

**Three Correction Modes**:

- **Mode A** (residual_only): ``residual = b01_geometric - linear_STIR``
- **Mode B** (additive_only): ``additive_new = (b01 - b02)_penetrate``
- **Mode C** (both): ``residual = b01_penetrate - full_STIR_projection``

Example::

    from sirf_simind_connection.core.coordinator import SimindCoordinator

    coordinator = SimindCoordinator(
        simind_simulator=simind_sim,
        num_subsets=12,
        correction_update_interval=60,  # Update every 60 subiterations (5 epochs)
        residual_correction=True,
        update_additive=True,  # Mode C: both corrections
        linear_acquisition_model=linear_am,
    )

2. SimindSubsetProjector
~~~~~~~~~~~~~~~~~~~~~~~~~

**Module**: :mod:`sirf_simind_connection.core.projector`

**Purpose**: Projector for a single subset that coordinates with the shared coordinator.

**Key Features**:

- **No simulation overhead**: References shared coordinator instead of running own simulations
- **Automatic update triggering**: Calls ``coordinator.should_update()`` on each ``forward()`` to check if simulation is needed
- **Cache-based updates**: Coordinator tracks ``cache_version`` to signal new results
- **CIL-compatible**: Implements ``direct()`` and ``adjoint()`` for CIL framework

**Workflow**:

1. ``forward(image)`` called by optimizer
2. Check ``coordinator.should_update()`` using algorithm's iteration counter
3. If update interval reached, coordinator runs full SIMIND simulation
4. Apply corrections via KL function's ``eta`` parameter (updated by ``UpdateEtaCallback``)
5. Return STIR forward projection (fast, LINEAR model with eta handling additive)

Example::

    from sirf_simind_connection.core.projector import SimindSubsetProjector

    # For each subset i:
    projector_i = SimindSubsetProjector(
        stir_projector=stir_acq_model_i,  # From SIRF partitioner
        coordinator=coordinator,           # Shared across all subsets
        subset_indices=[0, 12, 24, ...],   # Views for this subset
    )
    projector_i.set_up(acq_data_subset_i, image_template)

3. CIL Partitioner
~~~~~~~~~~~~~~~~~~

**Module**: :mod:`sirf_simind_connection.utils.cil_partitioner`

**Purpose**: Partition data into CIL-compatible objective functions.

**Key Functions**:

``partition_data_with_cil_objectives()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wraps SIRF's partitioner to create CIL objectives:

1. Uses ``sirf.contrib.partitioner.data_partition()`` to get subset acquisition models
2. Wraps each in ``SimindSubsetProjector`` (if coordinator provided)
3. Creates ``OperatorCompositionFunction(ResidualCorrectedKullbackLeibler(f=data_subset), projector)``

**Important Note**: The projectors are LINEAR acquisition models (no additive term). The additive and residual corrections are handled inside the residual-aware KL function – the callback updates its ``additive`` and ``residual`` parameters after each SIMIND or STIR-PSF simulation.

Residual-Aware vs Classical KL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


``ResidualCorrectedKullbackLeibler`` now uses the Jordan decomposition of the
residual correction. For each bin we split ``r = r^{+} - r^{-}`` with
``r^{+} = \max(r, 0)`` and ``r^{-} = \max(-r, 0)`` and define the effective
prediction and data counts

.. math::

    \mu_{\text{eff}} = (x + b) + r^{+}, \qquad f_{\text{eff}} = f + r^{-}.

The objective becomes the ordinary Poisson KL evaluated at
``(\mu_{\text{eff}}, f_{\text{eff}})``. This guarantees ``\mu_{\text{eff}} > 0``
and ``f_{\text{eff}} \ge 0`` regardless of the sign of the residual, so the
function stays convex and smooth. The gradient retains the familiar form
``1 - f_{\text{eff}} / \mu_{\text{eff}}``.

To handle empty acquisition bins, the effective data ``f_{\text{eff}}`` is
floored to ``counts_floor`` (default ``1\times10^{-8}``) before evaluating the
logarithm. This avoids ``\log(0)`` while leaving the gradient with respect to
the prediction unchanged.

``create_svrg_objective_with_rdp()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Combines SVRG function with SIRF RDP prior::

    total_objective = SumFunction(SVRGFunction(kl_objectives, ...), rdp_prior)

Example Usage
-------------

Basic Setup::

    from sirf_simind_connection.utils.cil_partitioner import (
        partition_data_with_cil_objectives,
        create_svrg_objective_with_rdp,
    )
    from sirf_simind_connection.core.coordinator import SimindCoordinator

    # Create coordinator ONCE
    coordinator = SimindCoordinator(
        simind_simulator=simind_sim,
        num_subsets=12,
        correction_update_interval=60,
        residual_correction=True,
        update_additive=False,
        linear_acquisition_model=linear_am,
    )

    # Partition data with CIL objectives
    kl_objectives, projectors, indices, kl_funcs = partition_data_with_cil_objectives(
        acquisition_data=measured_data,
        additive_data=initial_scatter_estimate,
        multiplicative_factors=normalisation,
        num_subsets=12,
        initial_image=initial_image,
        create_acq_model=lambda: AcquisitionModelUsingRayTracingMatrix(),
        coordinator=coordinator,
        mode="staggered",
    )

    # Create SVRG + RDP objective
    total_obj = create_svrg_objective_with_rdp(
        kl_data_functions=kl_funcs,
        rdp_prior=rdp_prior,
        sampler=sampler,
        initial_image=initial_image,
        num_subsets=12,
    )

Comparison: Old vs New Workflow
--------------------------------

Old Workflow (SIRF objectives)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Each subset would trigger separate SIMIND simulations - very inefficient!

::

    # Create SimindProjector for each subset (N separate simulators!)
    get_am = lambda: SimindProjector(simind_sim, stir_am, ...)

    # Partition using SIRF
    _, _, obj_funs = partitioner.data_partition(..., create_acq_model=get_am)

    # SIRF objectives (incompatible with CIL composition)
    for obj in obj_funs:
        obj.set_prior(rdp_prior)

    # SVRG with SIRF objectives
    f_obj = -SVRGFunction(obj_funs, ...)

New Workflow (CIL objectives + Coordinator)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Benefit**: ONE SIMIND simulation for all subsets, distributed efficiently!

::

    # Create coordinator ONCE
    coordinator = SimindCoordinator(simind_sim, num_subsets, update_interval, ...)

    # Create STIR acquisition model factory (NO SimindProjector wrapper)
    get_am = lambda: AcquisitionModelUsingRayTracingMatrix()

    # Partition using CIL partitioner (automatically wraps in SimindSubsetProjector)
    kl_objs, projs, indices, kl_funcs = partition_data_with_cil_objectives(
        ..., simind_coordinator=coordinator
    )

    # Combine SVRG + RDP
    total_obj = create_svrg_objective_with_rdp(kl_funcs, rdp_prior, ...)

    # Run optimization
    algo = ISTA(initial=initial_image, f=total_obj, g=IndicatorBox(lower=0))
    algo.run(num_iterations)

Scaling Strategy
----------------

SIMIND and STIR Alignment
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: SIMIND and STIR may have different absolute intensities even for same physics.

**Solution**: Scale SIMIND outputs to match STIR linear projection sum:

.. code-block:: python

    scale_factor = linear_proj.sum() / max(b02.sum(), 1e-10)
    b01_scaled = b01.clone()
    b01_scaled.fill(b01.as_array() * scale_factor)

This ensures residuals reflect **modeling differences** not intensity differences.

**Mode-Specific Scaling**:

- **Mode A**: Scale ``b01_geometric`` to match ``linear_proj``
- **Mode B & C**: Scale using ``b02`` (geometric primary) vs ``linear_proj`` comparison

Subset Residual Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Key Principle**: Each subset gets corrections for its own views only (NO scaling by 1/num_subsets).

The coordinator:

1. Runs SIMIND on full image (all views)
2. Computes full residual: ``residual_full = SIMIND - STIR``
3. Extracts subset views: ``residual_subset = residual_full.get_subset(subset_indices)``
4. Returns ``residual_subset`` (NOT ``residual_subset / num_subsets``)

This is correct because:

- Each subset projector only forward/backward projects its own views
- SIRF's subset scaling is already applied in the acquisition models
- Residual corrections should match the scale of subset projections

Iteration Tracking
------------------

Algorithm Iteration Counter
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The coordinator uses the CIL algorithm's ``iteration`` counter (which tracks subiterations) to determine when to run SIMIND simulations.

The coordinator must have a reference to the algorithm::

    coordinator.algorithm = algorithm  # Set before running

Example with 12 subsets, update interval = 24::

    # Epoch 1
    Subset 0 forward() → algorithm.iteration = 1
    Subset 1 forward() → algorithm.iteration = 2
    ...
    Subset 11 forward() → algorithm.iteration = 12

    # Epoch 2
    Subset 0 forward() → algorithm.iteration = 13
    ...
    Subset 11 forward() → algorithm.iteration = 24
    # If last_update = 0, then 24 - 0 = 24 >= 24 → TRIGGER UPDATE!

Update Triggering
~~~~~~~~~~~~~~~~~

Update happens when::

    (algorithm.iteration - last_update_iteration >= correction_update_interval)

**Important**: The coordinator's ``algorithm`` attribute must be set before running reconstruction::

    coordinator.algorithm = algorithm
    algorithm.run(num_iterations)

**Recommended Intervals**:

- **Every epoch**: ``correction_update_interval = num_subsets`` (e.g., 12)
- **Every 2 epochs**: ``correction_update_interval = 2 * num_subsets`` (e.g., 24)
- **Every 5 epochs**: ``correction_update_interval = 5 * num_subsets`` (e.g., 60)

**UpdateEtaCallback**: After each SIMIND simulation, use the ``UpdateEtaCallback`` to update the KL function's ``eta`` parameter with the new cumulative additive term from the coordinator.

Cache Versioning
~~~~~~~~~~~~~~~~

After each SIMIND simulation, ``coordinator.cache_version`` increments. Subset projectors check this to detect new results.

Performance Benefits
--------------------

Efficiency Gains
~~~~~~~~~~~~~~~~

**Without Coordinator**:

- N subsets × M updates = N × M SIMIND simulations
- Example: 12 subsets, 5 updates = 60 SIMIND runs

**With Coordinator**:

- 1 simulation per update = M SIMIND simulations
- Example: 12 subsets, 5 updates = 5 SIMIND runs
- **12× reduction in simulation overhead!**

MPI Parallelization
~~~~~~~~~~~~~~~~~~~

Coordinator runs SIMIND with MPI across all projections::

    mpirun -np 6 simind output output -p /MP:6

This further accelerates each simulation using multiple cores.

Testing
-------

See :ref:`testing` for comprehensive test coverage:

- :file:`tests/test_coordinator.py` - SimindCoordinator tests
- :file:`tests/test_cil_partitioner.py` - CIL partitioner tests

Run tests::

    # All coordinator tests
    pytest tests/test_coordinator.py -v

    # All partitioner tests
    pytest tests/test_cil_partitioner.py -v

    # Skip slow SIMIND integration tests
    pytest tests/test_coordinator.py -v -m "not requires_simind"

Migration Guide
---------------

From SimindProjector to Coordinator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Old approach** (each subset runs own SIMIND)::

    for i in range(num_subsets):
        projector_i = SimindProjector(
            simind_simulator=simind_sim_i,  # Separate simulator!
            stir_projector=stir_am_i,
            correction_update_interval=5,
        )

**New approach** (shared coordinator)::

    # Create coordinator once
    coordinator = SimindCoordinator(
        simind_simulator=simind_sim,  # Single simulator
        num_subsets=num_subsets,
        correction_update_interval=5 * num_subsets,  # Epoch-based
        linear_acquisition_model=linear_am_full,
    )

    # Create subset projectors
    for i in range(num_subsets):
        projector_i = SimindSubsetProjector(
            stir_projector=stir_am_i,
            coordinator=coordinator,  # Shared!
            subset_indices=subset_indices_i,
        )

Further Reading
---------------

- :ref:`api` - Full API documentation
- :ref:`examples` - Example scripts
- :ref:`testing` - Test suite details
- :mod:`sirf_simind_connection.core.coordinator` - Coordinator module
- :mod:`sirf_simind_connection.core.projector` - Projector module
- :mod:`sirf_simind_connection.utils.cil_partitioner` - Partitioner utilities

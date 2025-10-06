# SimindCoordinator Architecture

## Overview

The new coordinator-based architecture enables efficient SIMIND Monte Carlo corrections in subset-based iterative reconstruction algorithms. Key improvements:

1. **One MPI-parallelized SIMIND simulation** for all subsets (instead of N separate simulations)
2. **CIL-compatible objectives** using `KullbackLeibler` with `OperatorCompositionFunction`
3. **Automatic residual scaling** by `1/num_subsets` to match SIRF's subset scaling
4. **Global iteration tracking** across all subsets for consistent update intervals

## Architecture Components

### 1. SimindCoordinator ([sirf_simind_connection/core/coordinator.py](../sirf_simind_connection/core/coordinator.py))

**Purpose**: Shared coordinator managing SIMIND simulations across all subset projectors.

**Key Features**:
- **Global iteration tracking**: Incremented on every `forward()` call from any subset
- **Periodic full simulation**: Runs SIMIND (all views) when `global_subiteration % correction_update_interval == 0`
- **Result caching**: Stores b01, b02, scale_factor, and cache_version
- **Subset distribution**: Extracts subset-specific views and applies `residual / num_subsets` scaling

**Three Correction Modes**:
- **Mode A** (residual_only): `residual = b01_geometric - linear_STIR`
- **Mode B** (additive_only): `additive_new = (b01 - b02)_penetrate`
- **Mode C** (both): `residual = b01_penetrate - full_STIR_projection`

**Example**:
```python
from sirf_simind_connection import SimindSimulator, SimindCoordinator

coordinator = SimindCoordinator(
    simind_simulator=simind_sim,
    num_subsets=12,
    correction_update_interval=60,  # Update every 60 subiterations (5 epochs)
    residual_correction=True,
    update_additive=True,  # Mode C: both corrections
)
```

### 2. SimindSubsetProjector ([sirf_simind_connection/core/projector.py](../sirf_simind_connection/core/projector.py))

**Purpose**: Projector for a single subset that coordinates with the shared coordinator.

**Key Features**:
- **No simulation overhead**: References shared coordinator instead of running own simulations
- **Automatic iteration tracking**: Increments `coordinator.global_subiteration` on each `forward()`
- **Cache-based updates**: Checks `coordinator.cache_version` to detect new results
- **CIL-compatible**: Implements `direct()` and `adjoint()` for CIL framework

**Workflow**:
1. `forward(image)` called by optimizer
2. Increment coordinator's global counter
3. If update interval reached, coordinator runs full SIMIND simulation
4. If new cache available, apply subset-specific residual correction
5. Return STIR forward projection (fast, with updated additive term)

**Example**:
```python
from sirf_simind_connection import SimindSubsetProjector

# For each subset i:
projector_i = SimindSubsetProjector(
    stir_projector=stir_acq_model_i,  # From SIRF partitioner
    coordinator=coordinator,           # Shared across all subsets
    subset_indices=[0, 12, 24, ...],   # Views for this subset
)
projector_i.set_up(acq_data_subset_i, image_template)
```

### 3. CIL Partitioner ([sirf_simind_connection/utils/cil_partitioner.py](../sirf_simind_connection/utils/cil_partitioner.py))

**Purpose**: Partition data into CIL-compatible objective functions.

**Key Functions**:

#### `partition_data_with_cil_objectives()`
Wraps SIRF's partitioner to create CIL objectives:
1. Uses `sirf.contrib.partitioner.data_partition()` to get subset acquisition models
2. Wraps each in `SimindSubsetProjector` (if coordinator provided)
3. Creates `OperatorCompositionFunction(KullbackLeibler(b=data_subset), projector)`

**Important Note**: The KL function's `eta` parameter is NOT used for additive corrections. The additive term is handled by the acquisition model's `forward()` method internally.

#### `create_svrg_objective_with_rdp()`
Combines SVRG function with SIRF RDP prior:
```python
total_objective = SumFunction(SVRGFunction(kl_objectives, ...), rdp_prior)
```

**Example**:
```python
from sirf_simind_connection.utils.cil_partitioner import (
    partition_data_with_cil_objectives,
    create_svrg_objective_with_rdp,
)

# Partition data
kl_objectives, projectors, partition_indices = partition_data_with_cil_objectives(
    acquisition_data=measured_data,
    additive_data=initial_scatter_estimate,
    multiplicative_factors=normalisation,
    num_subsets=12,
    initial_image=initial_image,
    create_acq_model=lambda: get_spect_am(...),
    simind_coordinator=coordinator,  # Optional
    mode="staggered",
)

# Create SVRG + RDP objective
total_obj = create_svrg_objective_with_rdp(
    kl_objectives,
    rdp_prior,
    sampler,
    snapshot_update_interval=24,
    initial_image=initial_image,
    negate=True,  # For maximization with ISTA
)
```

## Integration with compare_psf_models.py

### Old Workflow (SIRF objectives)
```python
# Create SimindProjector for each subset (N separate simulators!)
get_am = lambda: SimindProjector(simind_sim, stir_am, ...)

# Partition using SIRF
_, _, obj_funs = partitioner.data_partition(..., create_acq_model=get_am)

# SIRF objectives (incompatible with CIL composition)
for obj in obj_funs:
    obj.set_prior(rdp_prior)

# SVRG with SIRF objectives
f_obj = -SVRGFunction(obj_funs, ...)
```

**Problem**: Each subset would trigger separate SIMIND simulations, very inefficient!

### New Workflow (CIL objectives + Coordinator)
```python
# Create coordinator ONCE
coordinator = SimindCoordinator(simind_sim, num_subsets, update_interval, ...)

# Create STIR acquisition model factory (NO SimindProjector wrapper)
get_am = lambda: get_spect_am(...)

# Partition using CIL partitioner
kl_objectives, projectors = partition_data_with_cil_objectives(
    ...,
    create_acq_model=get_am,
    simind_coordinator=coordinator,  # Shared coordinator
)

# Create SVRG + RDP objective
total_obj = create_svrg_objective_with_rdp(kl_objectives, rdp_prior, ...)

# ISTA reconstruction
algo = ISTA(initial=..., f=total_obj, g=IndicatorBox(lower=0), ...)
algo.run(num_iterations)
```

**Benefits**:
- One SIMIND simulation for all subsets
- MPI parallelization across cores
- CIL-compatible objectives
- Proper residual scaling by `1/num_subsets`

## Scaling Strategy

### Why scale residuals by `1/num_subsets`?

SIRF's subset-based reconstruction scales forward projections and objectives by `1/num_subsets`. To maintain consistency:

1. **SIMIND simulation**: Run on FULL image (no scaling)
2. **Residual computation**: Compare SIMIND vs STIR (both full projections)
3. **Subset distribution**: Extract subset views, then apply `residual / num_subsets`
4. **Additive update**: Each subset gets `additive_subset + (residual_subset / num_subsets)`

This ensures the total correction across all subsets equals the full residual.

### Scaling Factor Computation

```python
# Mode A (residual only):
scale_factor = linear_proj.sum() / b01.sum()

# Mode B/C (with penetrate):
scale_factor = linear_proj.sum() / b02.sum()  # b02 = geometric primary
```

Ensures SIMIND intensities match STIR intensities before computing residuals.

## Iteration Tracking

### Global Counter
- **Incremented**: On every `forward()` call from any subset
- **Example**: 12 subsets, random sampling
  - Iteration 1: Subset 7 calls `forward()` → `global_subiteration = 1`
  - Iteration 2: Subset 3 calls `forward()` → `global_subiteration = 2`
  - ...
  - Iteration 12: Subset 11 calls `forward()` → `global_subiteration = 12` (1 epoch complete)

### Update Trigger
```python
if (global_subiteration % correction_update_interval) == 0:
    coordinator.run_full_simulation(image)
```

**Example**:
- `correction_update_interval = 60` (5 epochs for 12 subsets)
- Updates occur at subiterations: 60, 120, 180, 240, ...

### Reset Between Beta Values
```python
for beta in beta_values:
    coordinator.reset_iteration_counter()  # Start fresh for each beta
    algo.run(num_iterations)
```

## Testing

### Basic Import Test
```bash
mamba activate sirf-build
python project_psf/test_coordinator_basic.py
```

### Full Workflow Test
```bash
python project_psf/compare_psf_models.py \
    --config config_quick_test.yaml \
    --output_path /tmp/test_coordinator
```

## Migration Guide

### From SimindProjector to Coordinator-based

**Old code**:
```python
projector = SimindProjector(
    simind_simulator=simind_sim,
    stir_projector=stir_am,
    correction_update_interval=60,
    residual_correction=True,
    update_additive=True,
)
```

**New code**:
```python
# 1. Create coordinator (shared)
coordinator = SimindCoordinator(
    simind_simulator=simind_sim,
    num_subsets=12,
    correction_update_interval=60,
    residual_correction=True,
    update_additive=True,
)

# 2. Create subset projectors (one per subset)
projectors = []
for i, (stir_am_i, subset_indices_i) in enumerate(zip(stir_ams, partition_indices)):
    projector_i = SimindSubsetProjector(
        stir_projector=stir_am_i,
        coordinator=coordinator,
        subset_indices=subset_indices_i,
    )
    projectors.append(projector_i)

# Or use partition_data_with_cil_objectives() to do this automatically
```

## Performance Considerations

### SIMIND Simulation Cost
- **Full simulation** (all views, MPI): ~30-60 seconds
- **Per subset** (old approach): ~5-10 seconds × 12 = 60-120 seconds total

**Speedup**: 2-4× faster with coordinator (plus MPI parallelization)

### Recommended Update Intervals
- **Early iterations**: Every 3-5 epochs (more frequent updates)
- **Late iterations**: Every 10-20 epochs (diminishing returns)
- **High-energy isotopes**: More frequent (penetration/scatter more important)

### MPI Configuration
```python
simind_sim.runtime_switches.set_switch("MP", num_cores=6)
```

Set `num_cores` based on available CPU resources.

## Summary

The coordinator-based architecture provides:

✓ **Efficiency**: One SIMIND simulation for all subsets
✓ **Scalability**: MPI parallelization across cores
✓ **Correctness**: Automatic `1/num_subsets` residual scaling
✓ **Flexibility**: CIL-compatible objectives with SIRF RDP prior
✓ **Simplicity**: Clean separation of concerns (coordinator vs projector)

This design enables practical Monte Carlo corrections in iterative reconstruction with acceptable computational cost.

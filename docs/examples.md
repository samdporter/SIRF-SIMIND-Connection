# SIRF-SIMIND-Connection Examples

This directory contains example scripts demonstrating SIMIND Monte Carlo simulations and reconstruction with SIRF.

## Basic Examples (01-06)

The numbered examples (01-06) demonstrate core SIMIND functionality:

- **01_basic_simulation.py**: Simple SIMIND forward simulation
- **02_dicom_conversion.py**: Converting DICOM data for SIMIND
- **03_multi_window.py**: Multiple energy window simulation
- **04_custom_config.py**: Custom SIMIND configuration
- **05_scattwin_vs_penetrate_comparison.py**: Comparing scoring routines
- **06_schneider_density_conversion.py**: HU-to-density conversion

## Advanced Projector Examples (07-08)

### 7. `07_simind_projector_osem.py`

**SIMIND Simulation + OSEM with SimindProjector**

Demonstrates:

- Creating a simple phantom and running SIMIND forward simulation
- Using `SimindProjector` as forward model in OSEM reconstruction
- Comparing standard OSEM vs OSEM with SIMIND corrections
- Configuration: `collimator_routine=0` (geometric only, fast execution)

**Usage:**

```bash
python 07_simind_projector_osem.py
```

**Key Features:**

- Fast execution (10^5 photons, 60 projections, 64×64×64 image)
- Mode A corrections (residual only, no penetration physics)
- Auto-updates SIMIND corrections every 3 subiterations
- Side-by-side comparison plots with line profiles

---

### 8. `08_simind_coordinator_subsets.py`

**SimindCoordinator for Multi-Subset Reconstruction**

Demonstrates:

- Setting up `SimindCoordinator` to manage SIMIND across multiple subsets
- Creating `SimindSubsetProjector` instances for each subset
- Manual OSEM-style iteration loop with coordinator
- Efficient simulation: one SIMIND run distributed to all subsets

**Usage:**

```bash
python 08_simind_coordinator_subsets.py
```

**Key Features:**

- Coordinator manages 6 subset projectors efficiently
- One full SIMIND simulation per epoch (all 60 projections)
- Results distributed to individual subset projectors
- Manual OSEM implementation demonstrating coordinator API
- Detailed usage guidance for choosing projector vs coordinator

---

## Additional Examples

### `example_simind_projector_basic.py`

Legacy example demonstrating basic SimindProjector usage. Similar to example 07 but with slightly different structure.

### `example_simind_coordinator_cil.py`

Advanced example showing SimindCoordinator integration with CIL (Core Imaging Library) for SPDHG reconstruction with Total Variation regularization.

**Requirements:**

```bash
conda install -c ccpi -c conda-forge cil
```

### `example_simind_comparison.py`

Side-by-side comparison of SimindProjector vs SimindCoordinator approaches, demonstrating when to use each method

---

## Quick Start

For a quick overview:

1. **Basic simulation**: Start with `01_basic_simulation.py` to understand SIMIND forward projection
2. **OSEM reconstruction**: Run `07_simind_projector_osem.py` to see SIMIND-corrected reconstruction
3. **Multi-subset**: Try `08_simind_coordinator_subsets.py` to understand coordinator efficiency

## Configuration Details

Examples 07-08 use fast simulation settings optimized for demonstration:

```python
# Fast SIMIND configuration
config.set_value(26, 0.1)   # number_photon_histories (10^5 photons)
config.set_value(29, 60)    # spect_no_projections (60 views)
config.set_value(53, 0)     # collimator_routine = 0 (geometric only)
config.set_value(84, 1)     # scoring_routine = 1 (SCATTWIN - fast)
```

For production use:

- Increase photon histories: `config.set_value(26, 10)` for 10^7 photons
- Use more projections: `config.set_value(29, 120)` for full sampling
- Enable penetration: `config.set_value(53, 1)` for accurate septal penetration
- Use PENETRATE scoring: `config.set_value(84, 4)` for detailed component analysis

## SimindProjector Correction Modes

SimindProjector supports three correction modes (see CLAUDE.md for details):

### Mode A: Residual Correction Only (Example 07)

```python
SimindProjector(
    simind_simulator=simulator,
    stir_projector=stir_am,
    update_additive=False,
    residual_correction=True,
)
```

- **Use case**: Correct resolution modeling differences
- **Speed**: Fastest (no penetration physics needed)
- **SIMIND config**: `collimator_routine=0` (geometric only)

### Mode B: Additive Update Only

```python
SimindProjector(
    simind_simulator=simulator,
    stir_projector=stir_am,
    update_additive=True,
    residual_correction=False,
)
```

- **Use case**: Update scatter estimate with SIMIND
- **Speed**: Moderate (requires penetration physics)
- **SIMIND config**: `collimator_routine=1` (full penetration)

### Mode C: Both Corrections

```python
SimindProjector(
    simind_simulator=simulator,
    stir_projector=stir_am,
    update_additive=True,
    residual_correction=True,
)
```

- **Use case**: Comprehensive corrections (most accurate)
- **Speed**: Slowest (full penetration + residual computation)
- **SIMIND config**: `collimator_routine=1` (full penetration)

## Phantom Geometry

Advanced examples (07-08) use `create_simple_phantom()`:

- **Matrix**: 64×64×64 voxels
- **Voxel size**: 4.42 mm isotropic
- **Geometry**: Cylindrical body with hot sphere insert
- **Activity**: Uniform background + 4:1 hot sphere contrast

## Performance Tips

### For Prototyping (Fast)

Use settings from examples 07-08:

- Photons: 10^5 per projection (~30-60 seconds total)
- Projections: 60 views
- Collimator: geometric only (`routine=0`)
- Scoring: SCATTWIN

### For Research (Accurate)

```python
config.set_value(26, 10)   # 10^7 photons (may take hours)
config.set_value(29, 120)  # 120 projections
config.set_value(53, 1)    # Enable penetration physics
config.set_value(84, 4)    # PENETRATE scoring
```

### Choosing Between Projector and Coordinator

**Use SimindProjector when:**

- ✓ Simple drop-in replacement for SIRF AcquisitionModel
- ✓ Single projector use case (e.g., standard OSEM)
- ✓ Prefer simpler API

**Use SimindCoordinator when:**

- ✓ Subset-based reconstruction (OSEM, SPDHG, SVRG)
- ✓ CIL integration with multiple objectives
- ✓ Want maximum efficiency (one simulation for all subsets)
- ✓ Need advanced correction modes with cumulative additive tracking

### Update Interval Selection

Balance accuracy vs computational cost:

- **Every 1-3 subiterations**: Most accurate, slowest
- **Every 5-10 subiterations**: Good balance
- **Every epoch**: Fastest, still effective for many applications

## Troubleshooting

### SIMIND binary not found

Ensure SIMIND is installed and in your PATH, or specify explicitly:

```python
simulator = SimindSimulator(config, output_dir, simind_binary="/path/to/simind")
```

### CIL not available (for CIL examples)

Install CIL for `example_simind_coordinator_cil.py`:

```bash
conda install -c ccpi -c conda-forge cil
```

### Memory issues

Reduce matrix size or projections:

```python
# Smaller phantom
phantom = create_simple_phantom()  # Already minimal at 64³

# Fewer projections
config.set_value(29, 30)  # 30 instead of 60
```

### Slow execution

For faster testing:

```python
config.set_value(26, 0.05)  # 5×10^4 photons (half of examples)
```

## Further Reading

- **[CLAUDE.md](../CLAUDE.md)**: Comprehensive project documentation and API reference
- **[sirf_simind_connection/core/projector.py](../sirf_simind_connection/core/projector.py)**: SimindProjector full API
- **[sirf_simind_connection/core/coordinator.py](../sirf_simind_connection/core/coordinator.py)**: SimindCoordinator implementation
- **SimindProjector in CLAUDE.md**: Detailed correction mode documentation

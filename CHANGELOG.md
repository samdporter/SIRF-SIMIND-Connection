# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Documentation Consolidation and Example Scripts

- **ReadTheDocs Documentation Updates** (`docs/*.rst`):
  - **New `docs/coordinator_architecture.rst`**: Comprehensive SimindCoordinator architecture guide
    - Overview of coordinator-based architecture and key improvements
    - Detailed component descriptions (SimindCoordinator, SimindSubsetProjector, CIL partitioner)
    - Three correction modes (A, B, C) with examples
    - Comparison of old vs new workflow
    - Scaling strategy and iteration tracking details
    - Performance benefits and MPI parallelization
    - Migration guide from SimindProjector
    - Testing and further reading references
  - **Updated `docs/examples.rst`**: Added examples 07-08
    - Example 07: SimindProjector with OSEM (Monte Carlo corrections)
    - Example 08: SimindCoordinator for multi-subset reconstruction
  - **Updated `docs/api.rst`**: Added coordinator and CIL partitioner modules
    - New section: "Projector and Coordinator" with automodule for coordinator
    - New section: "CIL Partitioner" utilities
  - **Updated `docs/index.rst`**: Added coordinator_architecture to table of contents

- **New Example Scripts**:
  - **`examples/07_simind_projector_osem.py`**: OSEM reconstruction with SimindProjector
    - Demonstrates SimindProjector as drop-in replacement for SIRF AcquisitionModel
    - Mode A (residual correction) with fast SCATTWIN scoring and collimator_routine=0
    - Side-by-side comparison of standard OSEM vs SIMIND-corrected OSEM
    - Generates comparison plots with line profiles
  - **`examples/08_simind_coordinator_subsets.py`**: Multi-subset reconstruction with SimindCoordinator
    - Demonstrates coordinator managing multiple subset projectors
    - Manual OSEM-style iteration loop showing coordinator API
    - Efficient simulation: one SIMIND run distributed to all subsets
    - Explains when to use coordinator vs projector

- **Updated Root README** (`README.md`):
  - Streamlined and concise GitHub landing page
  - Clear navigation to full documentation in `docs/`
  - Quick start examples for basic simulation, OSEM with corrections, and density conversion
  - Examples guide (01-08) with links to detailed documentation
  - Three correction modes overview
  - SimindCoordinator efficiency benefits
  - Testing, contributing, and citation sections

### Changed

- **Documentation Organization**: Consolidated from scattered markdown files to structured `docs/` directory
  - Main API reference: `docs/README.md` (complete documentation hub)
  - Examples guide: `docs/examples.md` (all example documentation)
  - Architecture details: `docs/coordinator_architecture.md` (coordinator design)
  - Root README: Concise overview with links (GitHub landing page)

- **Example Files Cleanup**: Removed duplicate un-numbered example scripts
  - Removed: `examples/example_simind_projector_basic.py` (superseded by `07_simind_projector_osem.py`)
  - Removed: `examples/example_simind_comparison.py` (content integrated into documentation)
  - Kept: `examples/example_simind_coordinator_cil.py` (advanced CIL-SPDHG example, not duplicate)
  - Final structure: 8 numbered examples (01-08) plus one advanced CIL example

- **.gitignore Updates**:
  - Added `CLAUDE.md` to gitignore (AI assistant documentation, local only)

- **New Test Files**:
  - **`tests/test_coordinator.py`** (~400 lines): Comprehensive tests for SimindCoordinator
    - Initialization and mode detection (Modes A, B, C)
    - Iteration tracking and update triggering
    - Cache management and versioning
    - Cumulative additive term tracking
    - Subset indices and residual extraction
    - Edge cases (single subset, large intervals, missing models)
    - Integration tests with actual SIMIND simulations
    - Output directory handling
  - **`tests/test_cil_partitioner.py`** (~350 lines): Tests for CIL partitioner utilities
    - CILAcquisitionModelAdapter wrapper tests
    - partition_data_with_cil_objectives function tests
    - Staggered vs sequential subset index generation
    - Integration with SimindCoordinator
    - SimindSubsetProjector creation and configuration
    - KL data function setup with eta parameter
    - Edge cases (zero additive, missing models)
    - SVRG objective creation with RDP prior (optional)

- **Pytest Configuration Updates** (`pytest.ini`):
  - Added `slow` marker for tests taking >10 seconds

### ReadTheDocs Structure

All documentation now properly structured for Sphinx/ReadTheDocs:

```
docs/
├── index.rst                      # Main documentation index
├── intro.rst                      # Introduction
├── installation.rst               # Installation guide
├── usage.rst                      # Usage guide
├── examples.rst                   # Examples (UPDATED with 07-08)
├── api.rst                        # API reference (UPDATED with coordinator)
├── coordinator_architecture.rst   # Coordinator guide (NEW, converted from .md)
├── testing.rst                    # Testing guide
├── changelog.rst                  # Changelog
└── contributing.rst               # Contributing guide
```

### Fixed

- **Critical: SimindSubsetProjector Double Set_Up Error** (`sirf_simind_connection/core/projector.py`):
  - **Issue**: `RuntimeError: cannot set_up const object` when using `SimindSubsetProjector` with partitioner
  - **Root Cause**: SIRF partitioner sets up acquisition models internally when `initial_image` is provided, then `SimindSubsetProjector.set_up()` tried to set up the same model again
  - **Fix**: Added try-except block in `SimindSubsetProjector.set_up()` to gracefully handle already-setup models
  - **Impact**: Projector now works seamlessly with both direct instantiation and partitioner-based workflows
  - **Files Modified**: `sirf_simind_connection/core/projector.py`

- **Critical: Coordinator Mode A (residual_only) Using Wrong SIMIND Output** (`sirf_simind_connection/core/coordinator.py`):
  - **Issue**: Mode A failed with `'NoneType' object has no attribute 'clone'`
  - **Root Cause**: For `mode_residual_only`, coordinator stored geometric output in `cached_b01` but residual computation used `cached_b02`
  - **Fix**: Corrected output storage - Mode A now stores `GEOM_COLL_PRIMARY_ATT` in `cached_b02` (not `cached_b01`), sets `cached_b01 = None`
  - **Impact**: Mode A (geometric residual correction) now works correctly
  - **Files Modified**: `sirf_simind_connection/core/coordinator.py`

- **Critical: Coordinator Scaling Factor Using Wrong Output** (`sirf_simind_connection/core/coordinator.py`):
  - **Issue**: Mode A scaling used `cached_b01.sum()` when `cached_b01` was `None`
  - **Fix**: Changed scaling for Mode A to use `cached_b02.sum()` (geometric output)
  - **Impact**: Proper scaling for all three correction modes
  - **Files Modified**: `sirf_simind_connection/core/coordinator.py`

- **Critical: Coordinator b01 Scaling in Wrong Location** (`sirf_simind_connection/core/coordinator.py`):
  - **Issue**: Code unconditionally scaled `cached_b01` before checking correction mode, causing `None.clone()` error in Mode A
  - **Fix**: Moved `b01_scaled` computation into Mode B and Mode C branches only (where it's needed)
  - **Impact**: Eliminates unnecessary operations and prevents errors when `cached_b01` is `None`
  - **Files Modified**: `sirf_simind_connection/core/coordinator.py`

- **Critical: Missing output_dir Attribute in Coordinator** (`sirf_simind_connection/core/coordinator.py`):
  - **Issue**: `AttributeError: 'SimindCoordinator' object has no attribute 'output_dir'` when saving intermediate results
  - **Root Cause**: `output_dir` parameter accepted in `__init__()` but never stored as instance attribute
  - **Fix**: Added `self.output_dir = output_dir` in `__init__()`
  - **Impact**: Coordinator can now save intermediate SIMIND outputs when `output_dir` is specified
  - **Files Modified**: `sirf_simind_connection/core/coordinator.py`

- **Critical: SimindSubsetProjector Incorrectly Managing Additive Term** (`sirf_simind_connection/core/projector.py`):
  - **Issue**: `RuntimeError: AcquisitionModel.set_up() call missing` when calling `get_additive_term()` on linear acquisition model
  - **Root Cause**: Fundamental architectural conflict - `SimindSubsetProjector` wrapped LINEAR models (no additive) for CIL compatibility, but `_apply_residual_correction()` tried to get/set additive terms on these models
  - **Fix**: Removed `_apply_residual_correction()` method and its call from `forward()` - corrections now handled entirely via `UpdateEtaCallback` updating KL function's `eta` parameter
  - **Impact**: Clean separation of concerns - projectors stay linear, additive corrections managed via KL eta updates
  - **Architecture**:
    - `SimindSubsetProjector.forward()` triggers coordinator simulations, returns linear projection
    - `UpdateEtaCallback` detects new `cache_version`, extracts subset-specific additive from `coordinator.cumulative_additive`, updates KL `eta`
    - CIL gradient computation uses updated `eta` automatically
  - **Files Modified**: `sirf_simind_connection/core/projector.py`

- **Cleanup: Removed Unused Cache Tracking** (`sirf_simind_connection/core/projector.py`):
  - Removed `last_cache_version` attribute from `SimindSubsetProjector.__init__()` (no longer needed after removing `_apply_residual_correction()`)
  - **Files Modified**: `sirf_simind_connection/core/projector.py`

### Added

#### CIL LinearOperator Compatibility for SimindCoordinator

- **Modified `partition_data_with_cil_objectives()`** (`sirf_simind_connection/utils/cil_partitioner.py`):
  - Now uses `get_linear_acquisition_model()` to extract truly linear acquisition models (no additive term)
  - Creates KL functions with `eta = additive_term + epsilon` for proper CIL LinearOperator compatibility
  - Returns unwrapped `kl_data_functions` list for eta updates after SIMIND simulations
  - Updated return signature: `(kl_objectives, projectors, partition_indices, kl_data_functions)`
  - Documented that projectors are LINEAR (no additive) and eta holds the additive term

- **New `UpdateEtaCallback` Class** (`project_psf/compare_psf_models.py`):
  - Monitors coordinator's `cache_version` to detect new SIMIND simulations
  - Updates eta in all KL data functions after each SIMIND simulation
  - Extracts subset-specific additive terms using `get_subset(subset_indices)`
  - Automatically invoked during SVRG reconstruction when coordinator is present

- **Cumulative Additive Tracking in `SimindCoordinator`**:
  - Added `cumulative_additive` attribute to track total additive term across all simulations
  - Initialized via `initialize_with_additive(initial_additive)` method
  - New `get_full_additive_term()` method returns cumulative additive for eta updates
  - Properly handles all three correction modes with different accumulation strategies

### Fixed

- **Critical: Corrected Residual Computation Modes** (`sirf_simind_connection/core/coordinator.py`):
  - **Mode A (residual_only)**: Now correctly uses `b02_scaled - linear_proj` (geometric SIMIND vs geometric STIR)
    - **Previous (WRONG)**: Used `b01_scaled - linear_proj` (full SIMIND vs geometric STIR)
    - **Corrects**: Projection modeling using geometric SIMIND output
  - **Mode B (additive_only)**: Correctly replaces additive with `b01_scaled - b02_scaled` (scatter estimate)
    - **Behavior**: REPLACES cumulative additive each simulation (not additive)
    - **Corrects**: Additive term (scatter) using full SIMIND physics
  - **Mode C (both)**: Now correctly computes `residual = b01_scaled - old_cumulative - linear_proj`
    - **Result**: `new_cumulative = b01_scaled - linear_proj`
    - **Corrects**: Both projection modeling and additive term simultaneously
  - Updated both `run_full_simulation()` (for cumulative tracking) and `get_subset_residual()` (for subset updates)

- **Critical: Removed Incorrect 1/num_subsets Scaling** (`sirf_simind_connection/core/coordinator.py`):
  - **Issue**: `get_subset_residual()` was incorrectly scaling residuals by `1/num_subsets`
  - **Root Cause**: Each subset should get corrections for its own views only, not a fractional share of global correction
  - **Fix**: Changed from `residual_subset / num_subsets` to `residual_subset` (no scaling)
  - **Impact**: Subset projectors now receive exact residual for their views (extracted via `get_subset(subset_indices)`)
  - Updated docstring to clarify residual is NOT scaled

- **Performance: Replaced `.as_array()` with Fast `get_array()`** (`sirf_simind_connection/core/coordinator.py`):
  - Replaced all `.as_array()` calls with `get_array()` from `setr.utils.sirf`
  - `get_array()` wraps both `.get_array()` (old/slow) and `.as_array()` (fast) methods
  - Improves array access performance in scaling operations
  - Locations updated: `run_full_simulation()` (3 calls) and `get_subset_residual()` (3 calls)

#### Coordinator-Based Architecture for Efficient Subset Reconstruction

- **New `SimindCoordinator` Class** (`sirf_simind_connection/core/coordinator.py`): Shared coordinator managing SIMIND Monte Carlo simulations across multiple subset projectors
  - **Global iteration tracking**: Single counter incremented on every `forward()` call from any subset
  - **Periodic full simulation**: Runs one MPI-parallelized SIMIND simulation (all views) when `global_subiteration % correction_update_interval == 0`
  - **Result caching**: Stores b01, b02, scale_factor, and cache_version to avoid redundant simulations
  - **Subset distribution**: Extracts subset-specific views from full simulation and applies `residual / num_subsets` scaling
  - **Three correction modes**: Residual-only (Mode A), additive-only (Mode B), or both (Mode C)
  - **Automatic SIMIND configuration**: Sets collimator routine based on correction mode at creation time

- **New `SimindSubsetProjector` Class** (`sirf_simind_connection/core/projector.py`): Projector for a single subset that coordinates with shared `SimindCoordinator`
  - **No simulation overhead**: References shared coordinator instead of running own simulations
  - **Automatic iteration tracking**: Increments coordinator's global counter on each `forward()`
  - **Cache-based updates**: Checks `coordinator.cache_version` to detect new simulation results
  - **Subset-specific corrections**: Applies residual corrections already scaled by `1/num_subsets` from coordinator
  - **CIL-compatible**: Implements `direct()` and `adjoint()` for CIL (Core Imaging Library) framework
  - **Efficient workflow**: Only triggers coordinator simulation when update interval reached, then applies cached results

- **New CIL Partitioner** (`sirf_simind_connection/utils/cil_partitioner.py`): Creates CIL-compatible objective functions for subset-based reconstruction
  - **`partition_data_with_cil_objectives()`**: Wraps SIRF's partitioner to create CIL objectives
    - Uses `sirf.contrib.partitioner.data_partition()` to get subset acquisition models
    - Wraps each STIR model in `SimindSubsetProjector` (if coordinator provided)
    - Creates `OperatorCompositionFunction(KullbackLeibler(b=data_subset), projector)`
    - Returns list of CIL-compatible objective functions and projectors
  - **`create_svrg_objective_with_rdp()`**: Combines SVRG function with SIRF RDP prior
    - Uses CIL's `SumFunction` to combine SVRG with the SETR `RelativeDifferencePrior`
    - Handles negation for maximization with ISTA minimizer
    - Proper setup and initialization of all components

- **Updated `compare_psf_models.py`**: Integrated coordinator-based architecture
  - **New `partition_data_once_cil()`**: Replaces SIRF-only partitioner with CIL-compatible version
  - **New `run_svrg_with_prior_cil()`**: Uses CIL objectives with SVRG and SETR RDP prior
  - **Updated `_run_mode_core()`**: Creates coordinator once per mode, partitions with CIL objectives
  - **Automatic coordinator reset**: Resets iteration counter between different beta values
  - **Preconditioner computation**: Computes BSREM preconditioner from subset projector sensitivity
    and combines it with the SETR `RelativeDifferencePrior` Hessian diagonal via a Lehmer mean

- **Comprehensive Documentation** (`docs/coordinator_architecture.md`): Complete guide to coordinator-based architecture
  - Architecture overview and key benefits
  - Detailed component descriptions (coordinator, subset projector, CIL partitioner)
  - Three correction modes explained with examples
  - Scaling strategy and iteration tracking
  - Integration examples with `compare_psf_models.py`
  - Migration guide from `SimindProjector` to coordinator-based approach
  - Performance considerations and recommended update intervals
  - Testing instructions

- **Basic Test Script** (`project_psf/test_coordinator_basic.py`): Validates coordinator module imports and basic setup

### Changed

- **`compare_psf_models.py` Workflow**: Migrated from SIRF objectives to CIL objectives with coordinator
  - **Old**: Each subset created separate `SimindProjector` instance → N independent simulations
  - **New**: Single `SimindCoordinator` shared across all subsets → 1 MPI-parallelized simulation
  - **Performance**: 2-4× faster SIMIND simulations through coordination and MPI parallelization
  - **Correctness**: Automatic `1/num_subsets` residual scaling matches SIRF's subset scaling

### Technical Details

#### Coordinator Workflow
1. Each `SimindSubsetProjector.forward()` increments `coordinator.global_subiteration`
2. When `global_subiteration % correction_update_interval == 0`, coordinator runs full SIMIND simulation (all views)
3. Coordinator caches results (b01, b02, scale_factor) and increments `cache_version`
4. Each subset projector checks if `coordinator.cache_version > last_cache_version`
5. If new cache available, projector calls `coordinator.get_subset_residual()` to get its portion
6. Coordinator extracts subset views and returns `residual_subset / num_subsets`
7. Subset projector updates: `additive_new = additive_current + residual_subset`

#### Scaling Strategy
- **SIMIND simulation**: Run on FULL image (no subset scaling)
- **Residual computation**: Compare SIMIND vs STIR (both full projections)
- **Subset distribution**: Extract subset views, then apply `residual / num_subsets`
- **Ensures**: Total correction across all subsets equals full residual

#### CIL Integration
- **KullbackLeibler**: `b` parameter is measured data subset (NOT used for `eta` additive term)
- **OperatorCompositionFunction**: Composes KL with acquisition model operator
- **Additive corrections**: Handled internally by acquisition model's `forward()` method
- **SIRF RDP prior**: Works directly with CIL's `SumFunction` (no wrapper needed)

### Performance Benefits

- **Efficiency**: One SIMIND simulation for all subsets (vs N separate simulations)
- **Scalability**: MPI parallelization across cores (e.g., 6 cores via `simind.num_mpi_cores=6`)
- **Correctness**: Automatic `1/num_subsets` residual scaling maintains reconstruction accuracy
- **Flexibility**: CIL-compatible objectives enable advanced optimization algorithms

### Usage Example

```python
from sirf_simind_connection import SimindSimulator, SimindCoordinator
from sirf_simind_connection.utils.cil_partitioner import partition_data_with_cil_objectives

# 1. Create SIMIND simulator
simind_sim = create_simind_simulator(config, spect_data, output_dir)

# 2. Create coordinator (shared across all subsets)
coordinator = SimindCoordinator(
    simind_simulator=simind_sim,
    num_subsets=12,
    correction_update_interval=60,  # Update every 5 epochs (12 subsets * 5)
    residual_correction=True,
    update_additive=True,  # Mode C: both corrections
)

# 3. Partition data with CIL objectives
kl_objectives, projectors, _ = partition_data_with_cil_objectives(
    acquisition_data=measured_data,
    additive_data=initial_scatter,
    multiplicative_factors=normalisation,
    num_subsets=12,
    initial_image=initial_image,
    create_acq_model=lambda: get_spect_am(...),  # STIR model factory
    simind_coordinator=coordinator,
    mode="staggered",
)

# 4. Create SVRG + RDP objective and run ISTA
total_obj = create_svrg_objective_with_rdp(kl_objectives, rdp_prior, sampler, ...)
algo = ISTA(initial=initial_image, f=total_obj, g=IndicatorBox(lower=0), ...)
algo.run(num_iterations)
```

### Backward Compatibility

- Existing `SimindProjector` class unchanged and fully functional
- New coordinator-based architecture is opt-in via new classes
- `compare_psf_models.py` updated but old SIRF-based approach still available in git history

#### PSF Comparison Framework (`project_psf/`)

- **New SPECT PSF Model Comparison Script** (`project_psf/compare_psf_models.py`): Comprehensive framework for comparing 6 different PSF modeling approaches using RDP-regularized SVRG reconstruction
  - **Mode 1**: Fast SPECT (no resolution model) - Baseline
  - **Mode 2**: Accurate SPECT (PSF in projector + image-based Gaussian 6.91mm FWHM)
  - **Mode 3**: Fast + Geometric residual correction (SimindProjector with `residual_correction=True, update_additive=False`)
  - **Mode 4**: Accurate + Geometric residual correction
  - **Mode 5**: Fast + Full residual correction (SimindProjector with `residual_correction=True, update_additive=True`)
  - **Mode 6**: Accurate + Full residual correction

- **YAML Configuration System**: Clean separation of parameters from code
  - `config_default.yaml`: Production settings (all 6 modes, 3 beta values, 50 epochs, 14 subsets)
  - `config_quick_test.yaml`: Fast testing settings (modes 1-2, 1 beta value, 5 epochs, 7 subsets)
  - Command-line override support: `--override svrg.num_epochs=10 --override rdp.beta_values=[0.01,0.1]`
  - Nested parameter access with dot notation

- **Performance Optimizations**: Efficient data reuse strategy
  - Partitions data **once per mode**, then loops over beta values
  - Acquisition models created only once per mode (not per beta)
  - Only RDP prior recreated for each beta value (cheap operation)
  - Dramatically reduces runtime for multiple beta values

- **STIR CUDARelativeDifferencePrior Integration**: Uses STIR's native CUDA RDP prior
  - Proper normalization: `beta / num_subsets` for ISTA/SVRG algorithms
  - Anatomical prior support via `set_kappa(attenuation_map)`
  - Configurable gamma and epsilon parameters

- **CIL/SIRF Integration**: Follows PETRIC competition pattern
  - Uses `sirf.contrib.partitioner` for data partitioning
  - STIR objective functions used directly as CIL functions
  - SVRGFunction with sequential sampler
  - Negative objective for minimization (`-SVRGFunction` turns minimizer into maximizer)

- **MPI Parallel Execution Support**: Accelerates SIMIND simulations (modes 3-6)
  - Configuration: `use_mpi: true` and `num_mpi_cores: 6` in YAML
  - Command-line override: `--override simind.num_mpi_cores=12`
  - Generates proper MPI command: `mpirun -np 6 simind ...`

- **Comprehensive Documentation** (`project_psf/README.md`):
  - Quick start examples for all use cases
  - Configuration file structure and customization
  - Command-line override syntax
  - Performance optimization explanation
  - MPI setup and requirements
  - Troubleshooting guide

- **Example Configurations**: Ready-to-use setups
  - Default: Manchester NEMA data, Discovery670 scanner, Y-90 bremsstrahlung (36.84-126.16 keV)
  - Quick test: Reduced parameters for rapid iteration
  - Custom template: Complete example with all parameters documented

#### MPI Execution Enhancements

- **SimindExecutor MPI Support** (`sirf_simind_connection/core/components.py`):
  - Enhanced `run_simulation()` method to properly handle MPI runtime switch
  - Detects `MP` switch value (number of cores) from runtime switches
  - Constructs correct MPI command: `mpirun -np <cores> simind output output -p /MP:<cores>`
  - Added `check=True` to `subprocess.run()` for better error handling
  - Serial fallback: Uses `simind` directly when MPI not requested

- **Runtime Switch Documentation** (`sirf_simind_connection/core/config.py`):
  - `MP` switch already documented in `RuntimeSwitches.standard_switch_dict`
  - Value represents number of MPI processes to spawn

#### Module System Improvements

- **New `sirf_simind_connection/core/types.py` module**: Separated SIRF-independent types for better modularity
  - Contains enums (`ScoringRoutine`, `PenetrateOutputType`, `RotationDirection`, `ScatterType`)
  - Contains exceptions (`SimindError`, `ValidationError`, `SimulationError`, `OutputError`)
  - Contains constants (`SIMIND_VOXEL_UNIT_CONVERSION`, `MAX_SOURCE`, etc.)
  - Can be imported without SIRF installed, enabling CI builds and documentation generation

- **Added `converters` to main package lazy loader**: `from sirf_simind_connection import converters` now works

#### SimindProjector: AcquisitionModel-Compatible Interface

- **Complete AcquisitionModel Interface**: `SimindProjector` now provides a drop-in replacement for SIRF's `AcquisitionModel` with Monte Carlo-based corrections
  - Added `set_up(acq_templ, img_templ)` method to initialize with acquisition and image templates
  - Added `get_additive_term()` / `set_additive_term()` for managing scatter corrections
  - Added `get_background_term()` / `set_background_term()` for managing background
  - Added `get_constant_term()` to retrieve combined additive + background
  - Added `get_linear_acquisition_model()` to access cached linear model (no additive/background)
  - Added `is_linear()` and `is_affine()` methods to check model properties
  - Added `direct()` and `adjoint()` methods for CIL (Core Imaging Library) framework compatibility
  - Added `range_geometry()` and `domain_geometry()` pass-through methods

- **Three Correction Modes**: Flexible correction strategies for different use cases
  - **Mode A - Residual Only** (`residual_correction=True, update_additive=False`): Corrects resolution modeling using geometric SIMIND (index 53=0) vs linear STIR projection
  - **Mode B - Additive Only** (`update_additive=True, residual_correction=False`): Replaces scatter estimate with SIMIND penetrate outputs (b01-b02)
  - **Mode C - Both** (`update_additive=True, residual_correction=True`): Comprehensive corrections using residual between SIMIND full simulation and STIR projection

- **Automatic Iteration Tracking**: Built-in iteration management
  - `forward()` method auto-increments internal `_iteration_counter`
  - Automatic correction updates triggered every N iterations based on `correction_update_interval`
  - Added `reset_iteration_counter()` method for multi-stage reconstructions
  - Added `_should_update_corrections()` internal method to determine update necessity

- **Intelligent Scaling Strategy**: Automatic intensity normalization between SIMIND and STIR
  - Mode A scales SIMIND geometric output to match linear STIR projection
  - Modes B & C scale using b02 (geometric primary) vs linear STIR comparison
  - Ensures residuals reflect modeling differences rather than intensity differences

- **Internal State Management**: Robust caching and state tracking
  - `_linear_acquisition_model`: Cached linear STIR model for scaling and residuals
  - `_current_additive`: Tracks current additive term
  - `_last_update_iteration`: Records when corrections were last updated
  - `acq_templ` / `img_templ`: Stores geometry templates

- **SimindSimulator Enhancement**: Added `set_collimator_routine(enabled: bool)` method
  - Controls SIMIND index 53 (collimator modeling on/off)
  - `True`: Full penetration modeling (slower, more accurate)
  - `False`: Geometric only (faster, for resolution corrections)

- **Comprehensive Test Suite** (`tests/test_simind_projector.py`): Added 9 new test functions
  - `test_projector_initialization()`: Verify initial state
  - `test_projector_set_up()`: Test template initialization
  - `test_iteration_tracking()`: Validate auto-increment and reset
  - `test_should_update_corrections()`: Logic for update triggering
  - `test_acquisition_model_interface()`: Interface compatibility
  - `test_direct_and_adjoint()`: CIL framework operators
  - `test_additive_and_background_terms()`: Term management
  - `test_correction_modes_integration()`: Integration test for all three modes
  - `test_set_collimator_routine()`: Collimator toggle functionality

- **Documentation** (`CLAUDE.md`): Comprehensive SimindProjector usage guide
  - Detailed explanation of three correction modes with use cases
  - Basic setup examples as AcquisitionModel replacement
  - OSEM reconstruction integration examples
  - Iteration tracking and update control patterns
  - Scaling strategy explanation
  - CIL framework compatibility guide
  - Decision table for choosing correction modes
  - Performance considerations and optimization tips
  - Multi-stage reconstruction workflow example

### Fixed

- **Critical: Non-Circular Orbit Support** (Multiple files):
  - **Issue 1 - Missing Orbit File Creation**: Input orbit file (`.cor`) was not being created because `AcquisitionData.get_info()` doesn't include orbit/radii information
    - **Root Cause**: `extract_attributes_from_stir()` when passed an `AcquisitionData` object used `.get_info()` which omits orbit and radii arrays
    - **Fix**: Modified `extract_attributes_from_stir()` to write `AcquisitionData` to a temporary header file, then read attributes from the file (which includes orbit/radii)
    - **Impact**: Non-circular orbits now properly detected and orbit file created
    - **Files Modified**: `sirf_simind_connection/utils/stir_utils.py`

  - **Issue 2 - Missing Unit Conversion**: Orbit file radii written in mm instead of cm
    - **Root Cause**: `OrbitFileManager.write_orbit_file()` wrote radii values directly without converting from mm (STIR) to cm (SIMIND)
    - **Fix**: Added `radius_cm = radius_mm / SIMIND_VOXEL_UNIT_CONVERSION` conversion when writing orbit file
    - **Impact**: SIMIND now receives correct radius values (e.g., 13.4 cm instead of 134 cm)
    - **Files Modified**: `sirf_simind_connection/core/components.py`

  - **Issue 3 - File Naming Collision**: Input orbit file overwritten by SIMIND output
    - **Root Cause**: Input orbit file named `{prefix}.cor`, same as SIMIND's output filename
    - **Fix**: Changed input orbit file name to `{prefix}_input.cor` to prevent collision
    - **Impact**: Input orbit file preserved after SIMIND execution
    - **Files Modified**: `sirf_simind_connection/core/components.py`

  - **Issue 4 - Incorrect Command Argument Order**: Orbit file placed after `-p` flag in MPI commands
    - **Root Cause**: Orbit file appended after `-p` flag instead of before it (SIMIND requires orbit file as 3rd/4th/5th argument)
    - **Fix**: Reordered command construction to place orbit file before `-p` flag
    - **Impact**: SIMIND now correctly reads orbit file for non-circular orbits
    - **Files Modified**: `sirf_simind_connection/core/components.py`

  - **Issue 5 - Full Path Instead of Filename**: Orbit file passed with full path
    - **Root Cause**: `SimindExecutor.run_simulation()` used `str(orbit_file)` which includes full path
    - **Fix**: Changed to `orbit_file.name` to use only filename (since we `chdir` to output directory)
    - **Impact**: SIMIND command uses `output_input.cor` instead of `/full/path/output_input.cor`
    - **Files Modified**: `sirf_simind_connection/core/components.py`

  - **Tests Added**: Comprehensive regression tests for orbit file handling
    - `test_orbit_file_manager_write_with_unit_conversion()`: Verifies mm→cm conversion
    - `test_orbit_file_manager_naming_no_collision()`: Verifies `_input.cor` suffix prevents overwrite
    - `test_orbit_file_manager_read()`: Verifies cm→mm conversion when reading
    - `test_simind_executor_orbit_file_position()`: Verifies orbit file placed before `-p` flag in command
    - **Files Modified**: `tests/test_components.py`

  - **Deprecated**: `extract_attributes_from_stir_sinogram()` function
    - Now obsolete since `extract_attributes_from_stir()` always writes temp file for `AcquisitionData` objects
    - Marked with deprecation notice explaining missing orbit/radii data issue
    - **Files Modified**: `sirf_simind_connection/utils/stir_utils.py`

- **Critical: Subset View Index Out of Range Error** (`SimindCoordinator` and `SimindSubsetProjector`):
  - **Issue**: `ERROR: ProjDataInfoSubsetByView: views[1]=13 out of range (10)` when using coordinator with subset reconstruction
  - **Root Cause**: Coordinator was receiving subset acquisition models (e.g., 10 views) but attempting to extract subset views using full-data indices (e.g., [1, 13, 25, ...])
  - **Fix**: Coordinator now requires full-data acquisition models passed to `__init__()` instead of receiving them from subset projectors
  - **Changes**:
    - Added `linear_acquisition_model` and `stir_acquisition_model` parameters to `SimindCoordinator.__init__()`
    - `SimindCoordinator.run_full_simulation()` now uses stored full-data models (no longer accepts them as arguments)
    - `SimindSubsetProjector.forward()` calls `coordinator.run_full_simulation(image)` without passing models
    - Removed unused `_linear_acquisition_model` from `SimindSubsetProjector`
    - Added validation in `partition_data_with_cil_objectives()` to ensure coordinator has required full-data models
  - **Migration**: Users must create full-data acquisition models and pass them when creating coordinator:
    ```python
    # Create full-data models
    linear_am = get_am()
    linear_am.set_up(full_acquisition_data, initial_image)

    # For mode_both, also create full STIR model
    stir_am = get_am()
    stir_am.set_up(full_acquisition_data, initial_image)

    # Pass to coordinator
    coordinator = SimindCoordinator(
        simind_simulator=simind_sim,
        num_subsets=num_subsets,
        correction_update_interval=correction_update_interval,
        residual_correction=residual_correction,
        update_additive=update_additive,
        linear_acquisition_model=linear_am,   # NEW: required
        stir_acquisition_model=stir_am,       # NEW: for mode_both
    )
    ```
  - **Files Modified**:
    - `sirf_simind_connection/core/coordinator.py`: Added model storage and validation
    - `sirf_simind_connection/core/projector.py`: Removed model passing from `SimindSubsetProjector`
    - `sirf_simind_connection/utils/cil_partitioner.py`: Added validation for required models
    - `project_psf/compare_psf_models.py`: Updated to create and pass full-data models

- **RecursionError in lazy imports** (`converters`, `utils`, `builders` modules):
  - Changed from `from . import module_name` to `importlib.import_module()` in `__getattr__`
  - Prevents infinite recursion when importing submodules
  - Fixes: `from sirf_simind_connection.converters import attenuation` now works without recursion

- **Unconditional SIRF import in `components.py`**:
  - Wrapped `from sirf.STIR import ...` in try/except block
  - Now provides fallback types when SIRF not available
  - Enables importing `ScoringRoutine` and other enums without SIRF installation

- **`ScoringRoutine` import dependency**:
  - Moved from SIRF-dependent `components.py` to SIRF-independent `types.py`
  - `from sirf_simind_connection.core import ScoringRoutine` now works without SIRF

### Changed

- **SimindProjector Constructor**: Updated parameter naming for clarity
  - Renamed `update_scatter` → `update_additive` to better reflect functionality (replaces entire additive term with SIMIND scatter estimate)
  - Maintains backward compatibility through deprecated internal properties

- **SimindProjector Documentation**: Expanded class docstring
  - Now clearly states it's an AcquisitionModel-compatible interface
  - Documents three correction modes in class-level documentation
  - Clarifies attributes and use cases

- **forward() Method**: Enhanced with automatic correction updates
  - Now auto-increments iteration counter on each call
  - Triggers `_update_corrections()` when update interval reached
  - Maintains backward compatibility (still accepts same parameters)

### Implementation Details

- **Correction Update Logic** (`_update_corrections()` method): Implements mode-specific behavior
  - Mode A: Runs SIMIND without penetration, computes `residual = b01_scaled - linear_proj`, updates `additive += residual`
  - Mode B: Runs SIMIND with penetration, extracts b01 and b02, computes `scatter = b01_scaled - b02_scaled`, updates `additive = scatter`
  - Mode C: Runs SIMIND with penetration, computes `residual = b01_scaled - stir_full_proj`, updates `additive += residual`
  - All modes apply `.maximum(0)` to prevent negative values

- **Scaling Implementation**: Ensures consistency between SIMIND and STIR
  ```python
  scale_factor = linear_proj.sum() / max(b02.sum(), 1e-10)  # Avoid division by zero
  b01_scaled.fill(b01.as_array() * scale_factor)
  ```

- **Linear Acquisition Model**: Created in `set_up()` method
  ```python
  self._linear_acquisition_model = self._stir_projector.get_linear_acquisition_model()
  self._linear_acquisition_model.num_subsets = 1
  ```

### Technical Notes

- All changes maintain backward compatibility with existing code
- Deprecated properties (`_additive_correction`, `_additive_estimate`) retained for compatibility
- SIMIND-STIR conversion uses existing `PenetrateOutputType` enum from `core.components`
- Scaling uses sum-based normalization (more robust than max for noisy data)
- Iteration counter does not auto-reset; user must call `reset_iteration_counter()` explicitly

### Usage Examples

#### PSF Comparison Script

```bash
# Run all 6 modes with default settings
python project_psf/compare_psf_models.py \
    --config project_psf/config_default.yaml \
    --output_path /path/to/output

# Quick test (modes 1-2, 5 epochs)
python project_psf/compare_psf_models.py \
    --config project_psf/config_quick_test.yaml \
    --output_path /tmp/test

# Override parameters
python project_psf/compare_psf_models.py \
    --config project_psf/config_default.yaml \
    --output_path /path/to/output \
    --override svrg.num_epochs=20 \
    --override rdp.beta_values=[0.001,0.01,0.1,1.0] \
    --override simind.num_mpi_cores=12
```

#### SimindProjector with AcquisitionModel

```python
from sirf.STIR import AcquisitionModelUsingRayTracingMatrix
from sirf_simind_connection import SimindProjector, SimindSimulator, SimulationConfig
from sirf_simind_connection.configs import get
from sirf_simind_connection.core.components import ScoringRoutine

# Setup
config = SimulationConfig(get("AnyScan.yaml"))
simind_sim = SimindSimulator(config, 'output', scoring_routine=ScoringRoutine.PENETRATE)
simind_sim.set_source(phantom)
simind_sim.set_mu_map(mu_map)

stir_am = AcquisitionModelUsingRayTracingMatrix()

# Create projector with Mode C (both corrections)
projector = SimindProjector(
    simind_simulator=simind_sim,
    stir_projector=stir_am,
    correction_update_interval=5,
    update_additive=True,
    residual_correction=True
)

projector.set_up(acquisition_template, image_template)

# Use in OSEM reconstruction
from sirf.STIR import OSMAPOSLReconstructor
recon = OSMAPOSLReconstructor()
recon.set_acquisition_model(projector)  # Drop-in replacement!
recon.set_input(measured_data)
recon.set_up(initial_image)
recon.process()
```

## [0.2.2] - Previous Release

(Previous changelog entries would go here)

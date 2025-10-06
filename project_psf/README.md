# SPECT PSF Model Comparison

Compare different SPECT PSF modeling approaches using RDP-regularized SVRG reconstruction.

## 6 Reconstruction Approaches

1. **Fast SPECT (no res)** - Baseline with no resolution modeling
2. **Accurate SPECT (with res)** - PSF in projector + image-based Gaussian (6.91mm FWHM)
3. **Fast + Geometric residual** - (1) + SimindProjector geometric correction only
4. **Accurate + Geometric residual** - (2) + SimindProjector geometric correction only
5. **Fast + Full residual** - (1) + SimindProjector full correction (additive + residual)
6. **Accurate + Full residual** - (2) + SimindProjector full correction (additive + residual)

## Quick Start

```bash
# Run with default configuration
python compare_psf_models.py \
    --config config_default.yaml \
    --output_path /path/to/output

# Quick test with reduced parameters
python compare_psf_models.py \
    --config config_quick_test.yaml \
    --output_path /tmp/test

# Override specific parameters
python compare_psf_models.py \
    --config config_default.yaml \
    --output_path /path/to/output \
    --override svrg.num_epochs=10 \
    --override rdp.beta_values=[0.01,0.1]

# Run only specific modes
python compare_psf_models.py \
    --config config_default.yaml \
    --output_path /path/to/output \
    --override reconstruction.modes=[1,2]
```

## Configuration Files

### config_default.yaml
Production settings with all 6 modes, 3 beta values, 50 epochs, 14 subsets.

### config_quick_test.yaml
Fast testing with modes 1-2 only, 1 beta value, 5 epochs, 7 subsets.

### Custom Configuration

Create your own YAML config file:

```yaml
# Data paths
data:
  data_path: "/path/to/SPECT/data"
  mu_map_filename: "umap_zoomed.hv"
  measured_data_filename: "peak.hs"

# SIMIND configuration
simind:
  config: "Discovery670.yaml"
  energy_lower: 36.84
  energy_upper: 126.16
  use_mpi: true  # Enable MPI parallel execution
  num_mpi_cores: 6  # Number of MPI cores

# Resolution modeling
resolution:
  psf_fwhm: [6.91, 6.91, 6.91]  # mm
  stir_psf_params: [1.22, 0.031]  # [slope, intercept]

# SVRG reconstruction
svrg:
  num_subsets: 14
  num_epochs: 50
  initial_step_size: 1.0
  relaxation_eta: 0.02

# RDP prior
rdp:
  beta_values: [0.01, 0.1, 1.0]
  gamma: 2.0
  epsilon: 0.0001

# SimindProjector
projector:
  correction_update_epochs: 5

# Reconstruction selection
reconstruction:
  modes: [1, 2, 3, 4, 5, 6]

# Output options
output:
  save_intermediate: false
  verbose: false
```

## Command-Line Overrides

Override any config parameter without editing the file:

```bash
# Single override
--override svrg.num_epochs=20

# Multiple overrides
--override svrg.num_epochs=20 \
--override rdp.beta_values=[0.1,1.0,10.0] \
--override reconstruction.modes=[1,2,3]

# Override nested values
--override data.data_path="/new/path" \
--override simind.energy_lower=30.0
```

## Performance Optimization

**Key efficiency feature**: The script partitions data **ONCE per mode**, then loops over beta values. This means:
- Acquisition models created only once per mode
- Data partitioning (expensive) done only once per mode
- Only the RDP prior is recreated for each beta value (cheap)

This dramatically reduces runtime compared to recreating everything for each (mode, beta) combination.

## Output Structure

```
output_path/
├── mode1_nores_beta0.010_final.hv
├── mode1_nores_beta0.100_final.hv
├── mode1_nores_beta1.000_final.hv
├── mode2_withres_beta0.010_final.hv
├── mode2_withres_beta0.100_final.hv
├── mode2_withres_beta1.000_final.hv
├── mode3_fast_geom_beta0.010_final.hv
├── mode3_fast_geom_beta0.100_final.hv
├── mode3_fast_geom_beta1.000_final.hv
├── mode4_accurate_geom_beta0.010_final.hv
├── mode4_accurate_geom_beta0.100_final.hv
├── mode4_accurate_geom_beta1.000_final.hv
├── mode5_fast_full_beta0.010_final.hv
├── mode5_fast_full_beta0.100_final.hv
├── mode5_fast_full_beta1.000_final.hv
├── mode6_accurate_full_beta0.010_final.hv
├── mode6_accurate_full_beta0.100_final.hv
├── mode6_accurate_full_beta1.000_final.hv
├── simind_mode3/  # Shared SIMIND dir for all mode3 beta values
├── simind_mode4/  # Shared SIMIND dir for all mode4 beta values
├── simind_mode5/  # Shared SIMIND dir for all mode5 beta values
└── simind_mode6/  # Shared SIMIND dir for all mode6 beta values
```

## Example: Production Run

```bash
python compare_psf_models.py \
    --config config_default.yaml \
    --output_path /home/sporter/results/psf_comparison_$(date +%Y%m%d)
```

## Example: Custom Beta Values

```bash
python compare_psf_models.py \
    --config config_default.yaml \
    --output_path /path/to/output \
    --override rdp.beta_values=[0.001,0.01,0.1,1.0,10.0]
```

## Example: Subset of Modes

```bash
# Test modes 1-3 only (no full residual corrections)
python compare_psf_models.py \
    --config config_default.yaml \
    --output_path /path/to/output \
    --override reconstruction.modes=[1,2,3]
```

## Notes

### SIMIND Projector Correction Modes

- **Geometric residual only** (modes 3, 4):
  - `residual_correction=True, update_additive=False`
  - SIMIND runs **without** penetration (index 53=0)
  - Computes: `residual = b01_geom - STIR_linear`
  - Updates: `additive_new = additive_old + residual`

- **Full residual** (modes 5, 6):
  - `residual_correction=True, update_additive=True`
  - SIMIND runs **with** penetration (index 53=1)
  - Computes: `residual = b01_all - STIR_full`
  - Updates: `additive_new = additive_old + residual`

### Performance

- Modes 1-2 are fastest (no SIMIND)
- Modes 3-4 are moderate (SIMIND geometric only)
- Modes 5-6 are slowest (SIMIND with penetration)

### Dependencies

Requires:
- SIRF with STIR (CUDA-enabled for RDP prior)
- CIL (Core Imaging Library)
- sirf-simind-connection
- SIMIND binary (for modes 3-6)
- PyYAML

## Troubleshooting

### "CUDARelativeDifferencePrior not available"

Ensure STIR is built with CUDA support. Check:
```bash
python -c "from sirf.STIR import CUDARelativeDifferencePrior; print('OK')"
```

### SIMIND binary not found

Modes 3-6 require SIMIND. Ensure:
- SIMIND binary is in PATH or configured in SimindSimulator
- Output directory has write permissions

### Out of memory

Reduce in config file or via overrides:
- `svrg.num_epochs`
- `svrg.num_subsets`
- Or run fewer modes: `reconstruction.modes=[1,2]`

### YAML syntax errors

Check YAML formatting:
- Lists use `[item1, item2]` format
- Strings with spaces need quotes: `"path with spaces"`
- No tabs, only spaces for indentation

## MPI Parallel Execution

SIMIND simulations (modes 3-6) support MPI parallel execution for faster runtime.

### Configuration

Enable MPI in your config file:

```yaml
simind:
  config: "Discovery670.yaml"
  energy_lower: 36.84
  energy_upper: 126.16
  use_mpi: true  # Enable MPI
  num_mpi_cores: 6  # Number of cores
```

### Requirements

- MPI must be installed on your system
- SIMIND must be compiled with MPI support
- The `mpirun` command must be available in PATH

### Override MPI Settings

```bash
# Disable MPI
python compare_psf_models.py \
    --config config_default.yaml \
    --output_path /path/to/output \
    --override simind.use_mpi=false

# Change number of cores
python compare_psf_models.py \
    --config config_default.yaml \
    --output_path /path/to/output \
    --override simind.num_mpi_cores=12
```

### Command Generated

When MPI is enabled, the script runs:
```bash
mpirun -np 6 simind output_prefix output_prefix -p /MP:6 [other switches]
```

When MPI is disabled:
```bash
simind output_prefix output_prefix [switches]
```

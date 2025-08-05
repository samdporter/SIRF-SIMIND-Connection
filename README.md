# SIRF-SIMIND-Connection

A Python wrapper for seamless integration between SIRF (Synergistic Image Reconstruction Framework) and SIMIND Monte Carlo simulator for SPECT imaging applications.

## Features

- **Monte Carlo SPECT Simulation**: Run SIMIND simulations using SIRF data types
- **Dual Scoring Routines**: SCATTWIN (traditional) and PENETRATE (detailed interaction analysis)
- **DICOM to STIR Conversion**: Convert SPECT DICOM files to STIR format
- **Attenuation Modeling**: Convert between Hounsfield Units, attenuation coefficients, and density maps
- **Flexible Configuration**: YAML-based configuration with validation
- **Energy Window Support**: Multiple energy window simulation and analysis
- **Scanner Configurations**: Pre-configured templates for common SPECT scanners

## Installation

### Prerequisites

#### 1. SIRF Installation
Install SIRF following the [official instructions](https://github.com/SyneRBI/SIRF). The easiest way is using [docker](https://github.com/SyneRBI/SIRF-SuperBuild/blob/master/docker/README.md):

#### 2. SIMIND Installation

SIMIND is a Monte Carlo simulation program for SPECT imaging. You need to download and install it separately:

1. **Register and Download**: 
   - Go to [SIMIND download page](https://simind.blogg.lu.se/downloads/)
   - Download the appropriate version for your system (Windows, Linux, or macOS)

2. **Installation**:
   - Follow installation instructions for your specific OS

#### 3. Python Requirements
- Python 3.8 or higher
- NumPy, PyDICOM, PyYAML (automatically installed)

### Install SIRF-SIMIND-Connection

#### From source (recommended during development):

```bash
git clone https://github.com/samdporter/SIRF-SIMIND-Connection.git
cd SIRF-SIMIND-Connection
pip install -e ".[dev]"  # Install with development dependencies
```

#### From PyPI (when available):

```bash
pip install SIRF-SIMIND-Connection
```

### Verify Installation

```bash
# Quick verification
cd scripts/
python verify_installation.py

# Run all examples
python run_all_examples.py
```

## Quick Start

```python
from sirf_simind_connection import (
    SimindSimulator, SimulationConfig, configs, utils
)
from sirf_simind_connection.core.components import ScoringRoutine

# Create phantom and attenuation map
phantom = utils.stir_utils.create_simple_phantom()
mu_map = utils.stir_utils.create_attenuation_map(phantom)

# Load scanner configuration
config = SimulationConfig(configs.get("input.smc"))
config.import_yaml(configs.get("AnyScan.yaml"))

# Create simulator
simulator = SimindSimulator(
    config_source=config,
    output_dir='output',
    output_prefix='sim',
    photon_multiplier=10,
    scoring_routine=ScoringRoutine.SCATTWIN
)

# Set inputs
simulator.set_source(phantom)
simulator.set_mu_map(mu_map)
simulator.set_energy_windows([126], [154], [0])  # Tc-99m ± 10%
simulator.add_config_value(1, 140.0)  # 140 keV

# Run simulation
simulator.run_simulation()

# Get results
total_counts = simulator.get_total_output(window=1)
scatter_counts = simulator.get_scatter_output(window=1)
primary_counts = total_counts - scatter_counts
```

## Examples

Run all examples from the scripts directory:

```bash
cd scripts/
python run_all_examples.py  # Run all examples (10-25 minutes)
python verify_installation.py  # Quick verification
```

Individual examples:

1. **[Basic simulation](examples/01_basic_simulation.py)** - Simple phantom with SCATTWIN
2. **[DICOM conversion](examples/02_dicom_conversion.py)** - Convert DICOM to STIR format
3. **[Multi-energy windows](examples/03_multi_window.py)** - TEW scatter correction
4. **[Custom configurations](examples/04_custom_config.py)** - YAML workflow
5. **[Scoring routine comparison](examples/05_scattwin_vs_penetrate_comparison.py)** - SCATTWIN vs PENETRATE

## Scoring Routines

### SCATTWIN (Traditional)
- Energy-window based analysis
- Outputs: total, scatter, primary, air per window
- Best for: Clinical SPECT, scatter correction (TEW, DEW)

### PENETRATE (Detailed)
- Interaction component analysis  
- Outputs: 19 different interaction types
- Best for: Collimator design, physics research

## Scanner Configurations

Pre-configured templates are available in the `configs/` directory:
- `Discovery670.yaml`: GE Discovery 670 SPECT/CT
- `AnyScan.yaml`: Mediso AnyScan Trio SPECTCT
- Create your own by exporting from existing `.smc` files

## Documentation

Full documentation will be available at [Read the Docs](https://SIRF-SIMIND-Connection.readthedocs.io/) (when deployed).

## Troubleshooting

### SIMIND not found
If you get "SIMIND executable not found", ensure:
1. SIMIND is properly installed
2. The `simind` executable is in your system PATH

### Import errors
```bash
cd scripts/
python verify_installation.py  # Check dependencies
```

### File format issues
- SIMIND uses specific file formats for density maps (`.dmi`) and source maps (`.smi`)
- This wrapper handles conversions automatically

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development setup
```bash
# Clone the repository
git clone https://github.com/samdporter/SIRF-SIMIND-Connection.git
cd SIRF-SIMIND-Connection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests and examples
cd scripts/
python run_all_examples.py
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{sirf_simind_connection,
  author = {Sam Porter, Rebecca Gillen and Efstathios Varzakis},
  title = {SIRF-SIMIND-Connection: A Python wrapper for SPECT Monte Carlo simulations},
  year = {2025},
  url = {https://github.com/samdporter/SIRF-SIMIND-Connection}
}
```

## Acknowledgments

- SIRF developers
- SIMIND developers (Michael Ljungberg and colleagues at Lund University)
- Kris Thielemans, Daniel Deidda (PhD project supervisors)
- Contributors and users of this package
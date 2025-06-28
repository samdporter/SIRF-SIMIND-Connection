# SIRF-SIMIND-Connection

A Python wrapper for seamless integration between SIRF (Synergistic Image Reconstruction Framework) and SIMIND Monte Carlo simulator for SPECT imaging applications.

## Features

- **Monte Carlo SPECT Simulation**: Run SIMIND simulations using SIRF data types
- **DICOM to STIR Conversion**: Convert SPECT DICOM files to STIR format
- **Attenuation Modeling**: Convert between Hounsfield Units, attenuation coefficients, and density maps (simple - more to come)
- **Flexible Configuration**: YAML-based configuration with validation
- **Energy Window Support**: Multiple energy window simulation and analysis
- **Scatter Correction**: Integrated scatter simulation and correction (alpha phase)
- **Scanner Configurations**: Pre-configured templates for some common SPECT scanners

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
   
   **Linux/macOS**:
   ```bash
   # Extract the archive
   tar -xzf simind_*.tar.gz
   cd simind_*
   
   # Add SIMIND to your PATH
   echo 'export PATH=$PATH:/path/to/simind/bin' >> ~/.bashrc
   source ~/.bashrc
   
   # Verify installation
   simind --version
   ```
   
   **Windows**:
   - Extract the ZIP file to a directory (e.g., `C:\simind`)
   - Add `C:\simind\bin` to your system PATH:
     - Right-click "This PC" → Properties → Advanced system settings
     - Environment Variables → Edit PATH → Add the SIMIND bin directory
   - Open a new command prompt and verify: `simind --version`

#### 3. Python Requirements
- Python 3.8 or higher
- NumPy, PyDICOM, PyYAML (automatically installed)

### Install SIRF-SIMIND-Connection

#### From source (recommended during development):

```bash
git clone https://github.com/yourusername/SIRF-SIMIND-Connection.git
cd SIRF-SIMIND-Connection
pip install -e ".[dev]"  # Install with development dependencies
```

#### From PyPI (when available):

```bash
pip install SIRF-SIMIND-Connection
```

### Verify Installation

```python
import sirf_simind_connection
print(sirf_simind_connection.__version__)

# Check if SIMIND is accessible
from sirf_simind_connection import SimindSimulator
# This will raise an error if SIMIND is not found
```

## Quick Start

```python
from sirf_simind_connection import SimindSimulator, SimulationConfig
from sirf.STIR import ImageData

# Load or use provided scanner configuration
config = SimulationConfig(configs.get("input.smc")) # Currently we require a template .smc file but this may change in later iterations
config.import_yaml(configs.get("AnyScan.yaml"))  # Load a scanner configuration (if different from .smc file)
config_file = config.save_file(str(output_dir / "sim_config.smc")) # this also returns a path - required for SIMIND simulations

# Load your data
source_image = ImageData('source.hv')
attenuation_map = ImageData('mu_map.hv')

# Create simulator
simulator = SimindSimulator(
    template_smc_file_path=config,
    output_dir='output',
    source=source_image,
    mu_map=attenuation_map
)

# Set energy windows (e.g., 140 keV ± 10% for Tc-99m)
simulator.set_windows(
    lower_bounds=[126],
    upper_bounds=[154],
    scatter_orders=[0] # note that using scatter_order !=1 will only result in scatter files
)

# Run simulation
simulator.run_simulation()

# Get results
total_counts = simulator.get_output_total(window=1)
scatter = simulator.get_output_scatter(window=1)
```

## Scanner Configurations

Pre-configured templates are available in the `configs/` directory:
- `Discovery670.yaml`: GE Discovery 670 SPECT/CT
- `AnyScan.yaml`: Mediso AnyScan Trio SPECTCT
- Create your own by exporting from existing `.smc` files

## Documentation

Full documentation will be available at [Read the Docs](https://SIRF-SIMIND-Connection.readthedocs.io/) (when deployed).

### Tutorials
- [Basic SIMIND simulation](examples/01_basic_simulation.py)
- [DICOM to STIR conversion](examples/02_dicom_conversion.py)
- [Multi-energy window simulation](examples/03_multi_window.py)
- [Creating custom configurations](examples/04_custom_config.py)

## Troubleshooting

### SIMIND not found
If you get "SIMIND executable not found", ensure:
1. SIMIND is properly installed
2. The `simind` executable is in your system PATH (explained on the website)

### File format issues
- SIMIND uses specific file formats for density maps (`.dmi`) and source maps (`.smi`)
- This wrapper handles conversions automatically

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development setup
```bash
# Clone the repository
git clone https://github.com/yourusername/SIRF-SIMIND-Connection.git
cd SIRF-SIMIND-Connection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
flake8 sirf_simind_connection
black --check sirf_simind_connection
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
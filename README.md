# SIRF-SIMIND-Connection

[![Tests](https://github.com/samdporter/SIRF-SIMIND-Connection/workflows/Tests/badge.svg)](https://github.com/samdporter/SIRF-SIMIND-Connection/actions)
[![Documentation Status](https://readthedocs.org/projects/sirf-simind-connection/badge/?version=latest)](https://sirf-simind-connection.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

A Python wrapper for SIRF and SIMIND integration for SPECT imaging.

## Quick Links
- [Full Documentation](https://SIRF-SIMIND-Connection.readthedocs.io/)
- [Installation](https://SIRF-SIMIND-Connection.readthedocs.io/en/latest/installation.html)
- [Backend Support](https://sirf-simind-connection.readthedocs.io/en/latest/backends.html) - SIRF and STIR Python compatibility

## Key Features
- **Dual Backend Support** - Works with both SIRF and STIR Python
- **Connector/Adaptor API** - Python Connector plus STIR/SIRF/PyTomography adaptors
- SIRF integrated Monte Carlo SPECT Simulation using SIMIND
- Dual Scoring Routines (SCATTWIN/PENETRATE)
- DICOM-driven adaptor examples (STIR/SIRF/PyTomography)
- **Advanced Schneider2000 Density Conversion** - Clinically validated HU-to-density mapping with 44 tissue segments

## Installation

### Basic Installation

```bash
pip install sirf-simind-connection
```

### Backend Requirements

SIRF-SIMIND-Connection requires either **SIRF** or **STIR Python** as a backend. The backend is auto-detected at runtime, with SIRF preferred if both are available. See the [backend guide](https://sirf-simind-connection.readthedocs.io/en/latest/backends.html) for details.

#### Option 1: STIR Python (Recommended for basic usage)

Install from conda:
```bash
conda install -c conda-forge stir "numpy<2.0"
```

**Important**: As of November 2025, the STIR conda package (v6.3.0) requires NumPy < 2.0 due to binary compatibility issues with NumPy 2.x. The package was compiled against NumPy 1.x and will crash with memory errors if NumPy 2.x is installed. This should be resolved in future STIR releases.

Or build from source:
```bash
git clone https://github.com/UCL/STIR.git
cd STIR
# Follow build instructions in the repository
```

#### Option 2: SIRF (Required for advanced features)

SIRF is required for:
- Coordinator/Projector functionality
- CIL integration
- SIRF-native OSEM reconstruction (example 07B)

Install from source:
```bash
git clone https://github.com/SyneRBI/SIRF.git
cd SIRF
# Follow build instructions in the repository
```

**Note**: SIRF includes STIR, so you don't need to install STIR separately if using SIRF.

### SIMIND Requirement

SIMIND is **not included** in this repository and must be installed separately.

Use the official SIMIND resources:

- SIMIND site (Medical Radiation Physics, Lund University): https://www.msf.lu.se/en/research/simind-monte-carlo-program
- SIMIND manual/docs: https://www.msf.lu.se/en/research/simind-monte-carlo-program/manual

For quick local use in this repository, place your local SIMIND installation under:

```text
./simind
```

and make sure the executable is available at:

```text
./simind/simind
```

and ensure SIMIND data files are available under:

```text
./simind/smc_dir/
```

Docker scripts are configured to use this repo-local layout and automatically
wire SIMIND paths when the binary exists at `./simind/simind`.

## Quick Start
```python
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

# Access results as native SIRF/STIR objects when needed
native_outputs = simulator.get_outputs(native=True)
sirf_tot = simulator.get_total_output(native=True)
```

### Advanced Density Conversion

```python
from sirf_simind_connection.converters.attenuation import hu_to_density_schneider
import numpy as np

# Convert HU image to densities using Schneider2000 model
hu_image = np.array([[-1000, 0, 500], [800, 1200, 2000]])
density_map = hu_to_density_schneider(hu_image)  # 44-segment clinical model
```

### Pure Python Connector (NumPy Outputs)

```python
from sirf_simind_connection import RuntimeOperator, SimindPythonConnector
from sirf_simind_connection.configs import get

connector = SimindPythonConnector(
    config_source=get("AnyScan.yaml"),
    output_dir="output/python_connector",
    output_prefix="case01",
    quantization_scale=1.0,
)

outputs = connector.run(RuntimeOperator(switches={"NN": 1, "RR": 12345}))
total = outputs["tot_w1"]
print(total.projection.shape)
print(total.header_path)
```

For minimal toy runs with smaller projection/image settings, use
`get("Example.yaml")`.

`quantization_scale` controls source integer quantization before writing SIMIND
`.smi` files:

- `1.0`: uses full internal source integer range (best numeric precision)
- `< 1.0`: reduces integer scale for faster toy runs but increases rounding error

SIMIND treats source maps as integer weights; absolute activity/time scaling is
controlled separately by simulation settings.

## Docker Environments

Dedicated container definitions are provided in [`docker/`](docker/):

- `docker/stir/Dockerfile`
- `docker/sirf/Dockerfile`
- `docker/pytomography/Dockerfile`

Use Docker Compose:

```bash
docker compose -f docker/compose.yaml build stir
docker compose -f docker/compose.yaml run --rm stir
```

Run per-backend validation and separated examples:

```bash
bash scripts/run_container_validation.sh
bash scripts/run_container_validation.sh --with-simind
bash scripts/run_container_examples.sh
```

## Contributing
Please see our [Contributing Guide](CONTRIBUTING.md).

## License
Apache License 2.0

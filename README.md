# SIRF-SIMIND-Connection

[![Tests](https://github.com/samdporter/SIRF-SIMIND-Connection/workflows/Tests/badge.svg)](https://github.com/samdporter/SIRF-SIMIND-Connection/actions)
[![Documentation Status](https://readthedocs.org/projects/sirf-simind-connection/badge/?version=latest)](https://sirf-simind-connection.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

A Python wrapper for SIRF and SIMIND integration for SPECT imaging.

## Quick Links
- [Full Documentation](https://SIRF-SIMIND-Connection.readthedocs.io/)
- [Installation](https://SIRF-SIMIND-Connection.readthedocs.io/en/latest/installation.html)

## Key Features
- Monte Carlo SPECT Simulation
- Dual Scoring Routines (SCATTWIN/PENETRATE)  
- DICOM to STIR Conversion
- **Advanced Schneider2000 Density Conversion** - Clinically validated HU-to-density mapping with 44 tissue segments

## Installation

```bash
pip install sirf-simind-connection
```

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
```

### Advanced Density Conversion

```python
from sirf_simind_connection.converters.attenuation import hu_to_density_schneider
import numpy as np

# Convert HU image to densities using Schneider2000 model
hu_image = np.array([[-1000, 0, 500], [800, 1200, 2000]])
density_map = hu_to_density_schneider(hu_image)  # 44-segment clinical model
```

## Contributing
Please see our [Contributing Guide](CONTRIBUTING.md).

## License
Apache License 2.0

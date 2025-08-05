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
- Monte Carlo SPECT Simulation.
- Dual Scoring Routines.
- DICOM to STIR Conversion.

## Quick Start
```python
from sirf_simind_connection import SimindSimulator, SimulationConfig
config = SimulationConfig("config.yaml")
simulator = SimindSimulator(config)
simulator.run_simulation()
```

## Contributing
Please see our [Contributing Guide](CONTRIBUTING.md).

## License
Apache License 2.0

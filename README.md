# SIRF-SIMIND-Connection

[![Tests](https://github.com/samdporter/SIRF-SIMIND-Connection/workflows/Tests/badge.svg)](https://github.com/samdporter/SIRF-SIMIND-Connection/actions)
[![Documentation Status](https://readthedocs.org/projects/sirf-simind-connection/badge/?version=latest)](https://sirf-simind-connection.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

A Python wrapper integrating SIRF and SIMIND for Monte Carlo SPECT imaging simulations and reconstruction.

## Documentation

📚 **[Full Documentation on ReadTheDocs](https://sirf-simind-connection.readthedocs.io/)**

## Quick Links

- [Installation Guide](https://sirf-simind-connection.readthedocs.io/en/latest/installation.html)
- [Usage Guide](https://sirf-simind-connection.readthedocs.io/en/latest/usage.html)
- [Examples](https://sirf-simind-connection.readthedocs.io/en/latest/examples.html)
- [API Reference](https://sirf-simind-connection.readthedocs.io/en/latest/api.html)
- [SimindCoordinator Architecture](https://sirf-simind-connection.readthedocs.io/en/latest/coordinator_architecture.html)
- [Testing](https://sirf-simind-connection.readthedocs.io/en/latest/testing.html)

## Features

- **Monte Carlo SPECT Simulation** - Fast SCATTWIN or detailed PENETRATE scoring routines
- **SimindProjector** - Drop-in replacement for SIRF AcquisitionModel with MC corrections
- **SimindCoordinator** - Efficient multi-subset SIMIND simulation management
- **Schneider2000 Density Conversion** - Clinically validated HU-to-density mapping (44 tissue segments)
- **Format Conversion** - SIMIND ↔ STIR and DICOM → STIR conversion utilities

## Installation

```bash
pip install sirf-simind-connection
```

**Dependencies:** Python 3.8+, SIRF, SIMIND binary

**Optional:** CIL (Core Imaging Library) for advanced reconstruction algorithms

See the [installation guide](https://sirf-simind-connection.readthedocs.io/en/latest/installation.html) for detailed instructions.

## Quick Start

### Basic Simulation

```python
from sirf_simind_connection import SimindSimulator, SimulationConfig
from sirf_simind_connection.configs import get
from sirf_simind_connection.utils.stir_utils import create_simple_phantom, create_attenuation_map

# Create phantom
phantom = create_simple_phantom()
mu_map = create_attenuation_map(phantom)

# Configure and run SIMIND
config = SimulationConfig(get("AnyScan.yaml"))
simulator = SimindSimulator(config, output_dir='output')
simulator.set_source(phantom)
simulator.set_mu_map(mu_map)
simulator.set_energy_windows([126], [154], [0])  # Tc-99m
projections = simulator.run_simulation()
```

### OSEM with Monte Carlo Corrections

```python
from sirf.STIR import OSMAPOSLReconstructor, AcquisitionModelUsingRayTracingMatrix
from sirf_simind_connection import SimindProjector

# Create SimindProjector (drop-in for SIRF AcquisitionModel)
stir_am = AcquisitionModelUsingRayTracingMatrix()
projector = SimindProjector(
    simind_simulator=simulator,
    stir_projector=stir_am,
    correction_update_interval=3,
    residual_correction=True,
)
projector.set_up(measured_data, initial_image)

# Use with OSEM
recon = OSMAPOSLReconstructor()
recon.set_acquisition_model(projector)
recon.set_input(measured_data)
recon.set_num_subsets(6)
recon.set_num_subiterations(24)
recon.set_up(initial_image)
recon.process()
```

See the [usage guide](https://sirf-simind-connection.readthedocs.io/en/latest/usage.html) and [examples](https://sirf-simind-connection.readthedocs.io/en/latest/examples.html) for more details.

## Examples

The package includes 8 example scripts demonstrating various features:

- **01-06**: Basic SIMIND functionality (simulation, conversion, configuration, density conversion)
- **07**: SimindProjector with OSEM reconstruction
- **08**: SimindCoordinator for efficient multi-subset algorithms

Run examples individually:

```bash
python examples/01_basic_simulation.py
python examples/07_simind_projector_osem.py
```

See the [examples documentation](https://sirf-simind-connection.readthedocs.io/en/latest/examples.html) for detailed descriptions.

## SimindProjector: Monte Carlo Corrections

Three correction modes for integrating SIMIND Monte Carlo into reconstruction:

- **Mode A (Residual)**: Fast resolution corrections without penetration physics
- **Mode B (Additive)**: Scatter estimation with full penetration modeling
- **Mode C (Both)**: Comprehensive scatter and resolution corrections

Learn more in the [API documentation](https://sirf-simind-connection.readthedocs.io/en/latest/api.html).

## SimindCoordinator: Efficient Subset Algorithms

For subset-based reconstruction (OSEM, SPDHG, SVRG):

- **One simulation for all subsets** vs N separate simulations (major efficiency gain!)
- **MPI-parallelized** SIMIND execution
- **CIL integration** for advanced algorithms

See the [coordinator architecture guide](https://sirf-simind-connection.readthedocs.io/en/latest/coordinator_architecture.html) for details.

## Testing

```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/test_coordinator.py        # Coordinator tests
pytest tests/test_cil_partitioner.py    # CIL partitioner tests
pytest -m unit                          # Fast unit tests only
pytest -m requires_sirf                 # SIRF integration tests
```

See the [testing guide](https://sirf-simind-connection.readthedocs.io/en/latest/testing.html) for comprehensive information.

## Contributing

Contributions welcome! Please see our [Contributing Guide](https://sirf-simind-connection.readthedocs.io/en/latest/contributing.html).

## Citation

If you use this software in your research, please cite:

```
@software{porter2025sirf_simind,
  author = {Porter, S., Gillen, R. Varzakis, E., Deidda, D. Thielemans, K.},
  title = {SIRF-SIMIND-Connection: Integrating SIRF, SIMIND and CIL for SPECT},
  year = {2025},
  url = {https://github.com/samdporter/SIRF-SIMIND-Connection}
}
```

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

- SIRF development team
- My PhD colleagues, Rebecca Gillen and Efstathios Varzakis
- SIMIND development team (Michael Ljungberg et al.)
- CIL development team

---

**Project:** https://github.com/samdporter/SIRF-SIMIND-Connection

**Documentation:** https://sirf-simind-connection.readthedocs.io/

**Issues:** https://github.com/samdporter/SIRF-SIMIND-Connection/issues

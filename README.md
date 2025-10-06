# SIRF-SIMIND-Connection

[![Tests](https://github.com/samdporter/SIRF-SIMIND-Connection/workflows/Tests/badge.svg)](https://github.com/samdporter/SIRF-SIMIND-Connection/actions)
[![Documentation Status](https://readthedocs.org/projects/sirf-simind-connection/badge/?version=latest)](https://sirf-simind-connection.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

A Python wrapper integrating SIRF and SIMIND for Monte Carlo SPECT imaging simulations and reconstruction.

## Features

- **Monte Carlo SPECT Simulation** - Fast SCATTWIN or detailed PENETRATE scoring
- **SimindProjector** - Drop-in replacement for SIRF AcquisitionModel with MC corrections
- **SimindCoordinator** - Efficient multi-subset SIMIND simulation management
- **Schneider2000 Density Conversion** - Clinically validated HU-to-density (44 tissue segments)
- **Format Conversion** - SIMIND ↔ STIR and DICOM → STIR

## Quick Links

- **[Full Documentation](docs/README.md)** - Complete API reference and guides
- **[Examples](docs/examples.md)** - Runnable example scripts (01-08)
- **[Coordinator Architecture](docs/coordinator_architecture.md)** - Advanced design details

## Installation

```bash
pip install sirf-simind-connection
```

**Dependencies:** Python 3.8+, SIRF, SIMIND

**Optional:** CIL (for advanced reconstruction algorithms)

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
    residual_correction=True,  # Mode A: fast corrections
)
projector.set_up(measured_data, initial_image)

# Use with OSEM
recon = OSMAPOSLReconstructor()
recon.set_acquisition_model(projector)  # ← Use SimindProjector instead
recon.set_input(measured_data)
recon.set_num_subsets(6)
recon.set_num_subiterations(24)
recon.set_up(initial_image)
recon.process()
```

### Density Conversion (Schneider2000)

```python
from sirf_simind_connection.converters.attenuation import hu_to_density_schneider

# Convert HU image to densities (44-segment clinical model)
density_map = hu_to_density_schneider(ct_hu_image)
```

## Examples

See [docs/examples.md](docs/examples.md) for runnable scripts:

- **01-06**: Basic SIMIND (simulation, conversion, configuration)
- **07**: SimindProjector + OSEM reconstruction
- **08**: SimindCoordinator for multi-subset algorithms

## Documentation Structure

```
docs/
├── README.md                      # Complete API reference and guides
├── examples.md                    # Detailed example documentation
└── coordinator_architecture.md    # SimindCoordinator design details
```

## SimindProjector: Three Correction Modes

**Mode A**: Residual correction (fast, no penetration)
- Corrects resolution modeling differences
- `residual = SIMIND_geometric - STIR_linear`

**Mode B**: Additive update (scatter estimation)
- Updates scatter with SIMIND
- `scatter = SIMIND_all - SIMIND_primary`

**Mode C**: Both (most accurate)
- Comprehensive scatter and resolution corrections

See [docs/README.md#simindprojector](docs/README.md#simindprojector) for details.

## SimindCoordinator: Efficient Subset Algorithms

For OSEM, SPDHG, SVRG with subsets:

- **One simulation** for all subsets (vs N separate simulations)
- **MPI-parallelized** SIMIND execution
- **CIL integration** for advanced algorithms

See [docs/coordinator_architecture.md](docs/coordinator_architecture.md) for architecture.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Testing

```bash
# Run all tests
pytest

# Run specific markers
pytest -m unit                  # Fast unit tests
pytest -m requires_sirf         # SIRF integration tests
pytest -m requires_simind       # SIMIND simulation tests
```

## Citation

If you use this software in your research, please cite:

```
[Citation information to be added]
```

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

- SIRF development team
- SIMIND development team (Michael Ljungberg et al.)
- Schneider et al. for the clinical density model

---

**Project:** https://github.com/samdporter/SIRF-SIMIND-Connection

**Issues:** https://github.com/samdporter/SIRF-SIMIND-Connection/issues

**Documentation:** [docs/README.md](docs/README.md)

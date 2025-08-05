.. _usage:

Usage Guide
============

Getting Started
----------------

A quick example to get started with SIRF-SIMIND-Connection:

```python
from sirf_simind_connection import SimindSimulator, SimulationConfig
from sirf_simind_connection.configs import get

config = SimulationConfig(get("AnyScan.yaml"))
simulator = SimindSimulator(config, output_dir='output')
simulator.run_simulation()

result = simulator.get_total_output(window=1)
print("Simulation completed successfully.")
```

Detailed Use Cases
--------------------

1. **Basic Simulation** - Learn how to set up and run simple simulations.
2. **Advanced Configuration** - Using custom YAML configurations.
3. **Extensive Output Analysis** - Understand the output from SCATTWIN vs PENETRATE routines.

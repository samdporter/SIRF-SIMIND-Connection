#!/usr/bin/env python3
"""Debug script to check if orbit data is being set correctly."""

from sirf.STIR import AcquisitionData

from sirf_simind_connection import SimindSimulator, SimulationConfig
from sirf_simind_connection.configs import get
from sirf_simind_connection.core.components import ScoringRoutine


# Load template
template_path = (
    "/home/storage/prepared_data/phantom_data/manc_nema_phantom_data/SPECT/peak.hs"
)
template = AcquisitionData(template_path)

# Create simulator
config = SimulationConfig(get("Discovery670.yaml"))
simulator = SimindSimulator(
    config, output_dir="/tmp/debug_orbit", scoring_routine=ScoringRoutine.PENETRATE
)

# Set template
print("Setting template sinogram...")
simulator.set_template_sinogram(template)

# Check values
print("\nAfter set_template_sinogram:")
print(f"  non_circular_orbit: {simulator.non_circular_orbit}")
print(f"  orbit_radii length: {len(simulator.orbit_radii)}")
if simulator.orbit_radii:
    print(f"  orbit_radii first 5: {simulator.orbit_radii[:5]}")
    print(f"  orbit_radii last 5: {simulator.orbit_radii[-5:]}")
else:
    print("  orbit_radii: EMPTY!")

print("\nExpected:")
print("  non_circular_orbit: True")
print("  orbit_radii length: 120")
print("  first radii: [134.0, 134.0, 134.0, 134.0, 134.0]")

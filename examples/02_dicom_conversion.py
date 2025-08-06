#!/usr/bin/env python
"""
DICOM to STIR Conversion Example

This example demonstrates how to convert SPECT DICOM files to STIR format
and prepare them for SIMIND simulation.

Note: This example primarily uses the builder classes and doesn't require
the new SimindSimulator API, so minimal changes are needed.
"""

import sys
from pathlib import Path

from sirf_simind_connection.builders import STIRSPECTAcquisitionDataBuilder


def convert_dicom_to_stir(dicom_file, output_dir):
    """
    Convert a SPECT DICOM file to STIR format.

    Args:
        dicom_file: Path to DICOM file
        output_dir: Directory for output files
    """
    print(f"Reading DICOM file: {dicom_file}")

    # Create builder and update from DICOM
    builder = STIRSPECTAcquisitionDataBuilder()
    builder.update_header_from_dicom(dicom_file)

    # Build acquisition data
    print("Converting to STIR format...")

    # Check if multiple energy windows
    if hasattr(builder, "energy_windows") and len(builder.energy_windows) > 1:
        print(f"Found {len(builder.energy_windows)} energy windows:")
        for i, ew in enumerate(builder.energy_windows):
            print(f"  Window {i + 1}: {ew['lower']}-{ew['upper']} keV")

        # Build multi-energy acquisition data
        output_base = output_dir / Path(dicom_file).stem
        acq_data_list = builder.build_multi_energy(str(output_base))

        print(f"Created {len(acq_data_list)} STIR files")
        return acq_data_list
    else:
        # Single energy window
        output_path = output_dir / (Path(dicom_file).stem + ".hs")
        acq_data = builder.build(str(output_path.with_suffix("")))
        print(f"Created STIR file: {output_path}")
        return [acq_data]


def analyze_acquisition(acq_data):
    """Print information about the acquisition data."""
    info = acq_data.get_info()
    print("\nAcquisition Information:")
    print(info)

    # Extract key parameters
    print("\nKey Parameters:")
    print(f"  Total counts: {acq_data.sum():.0f}")
    print(f"  Max counts per pixel: {acq_data.max():.0f}")
    print(f"  Mean counts per pixel: {acq_data.sum() / acq_data.size:.2f}")


def show_new_api_usage(acq_data):
    """Show how converted data would be used with the new API."""
    print("\n=== Using with New SimindSimulator API ===")
    print("Once you have converted DICOM data, you can use it with the new API:")
    print(
        """
    from sirf_simind_connection import SimindSimulator, SimulationConfig
    from sirf_simind_connection.core.components import ScoringRoutine

    # Load your config
    config = SimulationConfig('your_config.smc')

    # Create simulator
    simulator = SimindSimulator(
        config_source=config,
        output_dir='output',
        output_prefix='sim',
        photon_multiplier=10,
        scoring_routine=ScoringRoutine.SCATTWIN
    )

    # Set your source, attenuation map, and template sinogram
    simulator.set_source(source_image)
    simulator.set_mu_map(attenuation_map)
    simulator.set_template_sinogram(converted_acq_data)  # Use converted DICOM data!

    # Set energy windows if needed
    simulator.set_energy_windows([126], [154], [0])

    # Run simulation
    simulator.run_simulation()
    """
    )


def main():
    """Main conversion routine."""
    if len(sys.argv) < 2:
        print("Usage: python 02_dicom_conversion.py <dicom_file> [output_dir]")
        print("\nExample:")
        print("  python 02_dicom_conversion.py patient_spect.dcm output/")
        sys.exit(1)

    dicom_file = Path(sys.argv[1])
    if not dicom_file.exists():
        print(f"Error: DICOM file not found: {dicom_file}")
        sys.exit(1)

    # Set output directory
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = Path("output/dicom_conversion")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Convert DICOM to STIR
        acq_data_list = convert_dicom_to_stir(str(dicom_file), output_dir)

        # Analyze each energy window
        for i, acq_data in enumerate(acq_data_list):
            if len(acq_data_list) > 1:
                print(f"\n=== Energy Window {i + 1} ===")
            analyze_acquisition(acq_data)

        print(f"\nConversion complete! Output saved to: {output_dir}")

        # Show how to use with new API
        if acq_data_list:
            show_new_api_usage(acq_data_list[0])

    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

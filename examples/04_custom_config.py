#!/usr/bin/env python
"""
Custom Configuration Example (Updated)

This example shows how to create and modify SIMIND configurations,
including exporting to YAML for easy editing and version control.
"""

from pathlib import Path

from sirf_simind_connection import SimulationConfig, configs

TEMPLATE_PATH = configs.get("input.smc")  # Path to a template configuration file


def create_custom_lehr_config():
    """Create a custom Low-Energy High-Resolution (LEHR) collimator config."""
    # Start with a template

    config = SimulationConfig(str(TEMPLATE_PATH))

    # Modify for LEHR collimator
    print("Configuring LEHR collimator...")

    # LEHR collimator specifications
    config.set_value("hole_size_x", 0.111)  # 1.11 mm diameter
    config.set_value("hole_size_y", 0.111)  # Round holes
    config.set_value("collimator_thickness", 2.405)  # 24.05 mm
    config.set_value("distance_between_holes_x", 0.16)  # 1.6 mm
    config.set_value("distance_between_holes_y", 0.16)

    # Crystal parameters for high resolution
    config.set_value("intrinsic_resolution", 0.32)  # 3.2 mm FWHM

    # Energy resolution for modern camera
    config.set_value("energy_resolution", 9.5)  # 9.5% at 140 keV

    return config


def create_custom_hegp_config():
    """Create a custom High-Energy General-Purpose (HEGP) collimator config."""

    config = SimulationConfig(str(TEMPLATE_PATH))

    print("Configuring HEGP collimator...")

    # HEGP collimator specifications
    config.set_value("hole_size_x", 0.40)  # 4.0 mm diameter
    config.set_value("hole_size_y", 0.40)
    config.set_value("collimator_thickness", 5.9)  # 59 mm
    config.set_value("distance_between_holes_x", 0.50)  # 5.0 mm
    config.set_value("distance_between_holes_y", 0.50)

    # Set for higher energy (e.g., I-131 at 364 keV)
    config.set_value("photon_energy", 364.0)
    config.set_value("emitted_photons_per_decay", 0.812)

    # Adjust energy windows
    config.set_value("upper_window_threshold", 400)
    config.set_value("lower_window_threshold", 328)

    return config


def create_custom_cardiac_config():
    """Create a configuration optimized for cardiac SPECT."""

    config = SimulationConfig(str(TEMPLATE_PATH))

    print("Configuring for cardiac SPECT...")

    # Cardiac-specific settings
    config.set_value("pixel_size_simulated_image", 0.64)  # 6.4 mm pixels
    config.set_value("matrix_size_image_i", 64)
    config.set_value("matrix_size_image_j", 64)

    # 180-degree acquisition (RAO to LPO)
    config.set_value("spect_no_projections", 32)  # 32 views
    config.set_value("spect_rotation", 3)  # 180 degrees CW
    config.set_value("spect_starting_angle", 45)  # Start at RAO

    # Cardiac protocols often use zoom
    config.set_value("camera_offset_x", 10.0)  # 10 cm offset

    return config


def demonstrate_yaml_workflow():
    """Show how to use YAML for configuration management."""
    output_dir = Path("output/custom_configs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create and export LEHR config
    print("\n=== Creating LEHR Configuration ===")
    lehr_config = create_custom_lehr_config()
    lehr_yaml = output_dir / "lehr_collimator.yaml"
    lehr_config.export_yaml(str(lehr_yaml))

    # Create and export HEGP config
    print("\n=== Creating HEGP Configuration ===")
    hegp_config = create_custom_hegp_config()
    hegp_yaml = output_dir / "hegp_collimator.yaml"
    hegp_config.export_yaml(str(hegp_yaml))

    # Create and export cardiac config
    print("\n=== Creating Cardiac Configuration ===")
    cardiac_config = create_custom_cardiac_config()
    cardiac_yaml = output_dir / "cardiac_spect.yaml"
    cardiac_config.export_yaml(str(cardiac_yaml))

    # Demonstrate loading and modifying YAML
    print("\n=== Modifying Configuration via YAML ===")
    print(f"Loading {lehr_yaml}")

    # Create new config from YAML
    modified_config = SimulationConfig(str(TEMPLATE_PATH))
    modified_config.import_yaml(str(lehr_yaml))

    # Make additional modifications
    modified_config.set_value("number_photon_histories", 1e7)  # 10 million
    modified_config.set_flag("write_pulse_height_distribution_to_file", True)

    # Save as new config
    modified_yaml = output_dir / "lehr_high_stats.yaml"
    modified_config.export_yaml(str(modified_yaml))
    print(f"Saved modified configuration to {modified_yaml}")

    # Validate parameters
    print("\n=== Validating Configuration ===")
    if modified_config.validate_parameters():
        print("Configuration validation passed!")

    print(f"\nAll configurations saved to: {output_dir}")
    print("\nYou can now:")
    print("1. Edit the YAML files directly")
    print("2. Use them with SimindSimulator:")
    print("   config = SimulationConfig('lehr_collimator.yaml')")
    print("   simulator = SimindSimulator(config_source=config, ...)")


def demonstrate_new_api_usage():
    """Show how to use configurations with the new API."""
    output_dir = Path("output/custom_configs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Demonstrating New API Usage ===")

    # Create a custom config
    custom_config = create_custom_lehr_config()

    # Show how it would be used with the new SimindSimulator API
    print("\nWith the new API, you would use this config like:")
    print(
        """
    from sirf_simind_connection import SimindSimulator
    from sirf_simind_connection.core.components import ScoringRoutine
    
    # Option 1: Use the config object directly
    simulator = SimindSimulator(
        config_source=custom_config,  # Pass the config object
        output_dir='output',
        output_prefix='lehr_sim',
        photon_multiplier=10,
        scoring_routine=ScoringRoutine.SCATTWIN
    )
    
    # Set your inputs
    simulator.set_source(phantom)
    simulator.set_mu_map(mu_map)
    simulator.set_energy_windows([126], [154], [0])
    
    # Run simulation
    simulator.run_simulation()
    
    ----------
    
    # Option 2: Use a saved YAML file
    simulator = SimindSimulator(
        config_source='lehr_collimator.yaml',  # Pass YAML path
        output_dir='output',
        output_prefix='lehr_sim',
        photon_multiplier=10,
        scoring_routine=ScoringRoutine.SCATTWIN
    )
    """
    )

    # Save an example YAML that could be loaded
    example_yaml = output_dir / "example_usage.yaml"
    custom_config.export_yaml(str(example_yaml))
    print(f"\nExample config saved to: {example_yaml}")


def main():
    """Run the configuration examples."""
    demonstrate_yaml_workflow()
    demonstrate_new_api_usage()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8
"""
Run a simulation using SIMIND and STIR,
generate simulated sinograms and compare with measured data.
python script.py --simulation_config simulation_config.yaml --scanner_config scanner_config.yaml

Updated to work with the refactored SimindSimulator architecture.
"""

import os
import argparse
import time
import subprocess
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sirf.STIR import (ImageData, AcquisitionData, SPECTUBMatrix,
                       AcquisitionModelUsingMatrix, MessageRedirector,
                       SeparableGaussianImageFilter)
from sirf_simind_connection import SimindSimulator, SimulationConfig


msg = MessageRedirector()


def get_config_value(config_file, key_path):
    """Get a value from YAML config using yq-style path notation."""
    try:
        result = subprocess.run(
            ['yq', 'e', key_path, config_file],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Failed to get config value {key_path}: {e}")


def convert_config_value(val):
    """Convert string config values to appropriate types."""
    if val.lower() == 'false':
        return False
    elif val.lower() == 'true':
        return True
    # Try to convert to number (int first, then float)
    try:
        if '.' in val:
            return float(val)
        else:
            return int(val)
    except ValueError:
        return val


def load_simulation_config(config_file):
    """Load simulation parameters from config file."""
    def get_val(path):
        val = get_config_value(config_file, path)
        return convert_config_value(val)

    config = {
        # Project settings
        'data_dir': get_val('.project.data_dir'),
        'image_path': get_val('.project.image_path'),
        'output_dir': get_val('.project.output_dir'),
        'output_prefix': get_val('.project.output_prefix'),
        'simind_parent_dir': get_val('.project.simind_parent_dir'),
        'axial_slice': get_val('.project.axial_slice'),
        
        # Data files
        'mu_map_filename': get_val('.data.mu_map_filename'),
        'measured_data_filename': get_val('.data.measured_data_filename'),

        # Simulation parameters
        'total_activity': get_val('.project.total_activity'),
        'photon_energy': get_val('.simulation.photon_energy'),
        'window_lower': get_val('.simulation.window_lower'),
        'window_upper': get_val('.simulation.window_upper'),
        'scoring_routine': get_val('.simulation.scoring_routine'),
        'collimator_routine': get_val('.simulation.collimator_routine'),
        'photon_direction': get_val('.simulation.photon_direction'),
        'source_type': get_val('.simulation.source_type'),
        'collimator': get_val('.simulation.collimator'),
        'photon_multiplier': get_val('.simulation.photon_multiplier'),
        'time_per_projection': get_val('.simulation.time_per_projection'),
    }
    
    print(f"Loaded configuration with {len(config)} parameters")
    return config


def prepare_image_data(image_path, threshold_fraction=0.01):
    """
    Load and prepare image data for simulation.
    
    Parameters
    ----------
    image_path : str
        Path to the image file
    threshold_fraction : float
        Fraction of max value to use as threshold for zeroing low values
        
    Returns
    -------
    ImageData
        Prepared image with low values set to zero
    """
    print(f"Loading image from: {image_path}")
    image = ImageData(image_path)
    
    # Zero out low values to save simulation time
    image_array = image.as_array()
    threshold = threshold_fraction * image.max()
    low_value_count = np.sum(image_array < threshold)
    
    image_array[image_array < threshold] = 0
    image.fill(image_array)
    
    print(f"Set {low_value_count} voxels below threshold ({threshold:.3f}) to zero")
    print(f"Image dimensions: {image.dimensions()}, max value: {image.max():.2f}")
    
    return image


def prepare_mu_map(mu_map_filename, data_dir, reference_image):
    """
    Load or create attenuation map.
    
    Parameters
    ----------
    mu_map_filename : str or None
        Filename of mu map, or None to create uniform zero map
    data_dir : str
        Directory containing data files
    reference_image : ImageData
        Reference image for creating uniform mu map
        
    Returns
    -------
    ImageData
        Attenuation map
    """
    if mu_map_filename:
        mu_map_path = os.path.join(data_dir, mu_map_filename)
        print(f"Loading mu map from: {mu_map_path}")
        mu_map = ImageData(mu_map_path)
        print(f"Mu map max value: {mu_map.max():.4f}")
    else:
        print("Creating uniform zero mu map")
        mu_map = reference_image.get_uniform_copy(0)
    
    return mu_map


def blur_image(image, fwhms=(6.7, 6.7, 6.7)):
    """
    Apply a Gaussian blur to the image.

    Parameters
    ----------
    image : ImageData
        The input image to be blurred.
    fwhms : tuple of float, optional
        Full width at half maximum for the Gaussian kernel in each dimension.
        Default is (6.7, 6.7, 6.7).

    Returns
    -------
    ImageData
        Blurred image.
    """
    filter = SeparableGaussianImageFilter()
    filter.set_fwhms(fwhms)
    filter.apply(image)
    return image


def setup_simulator(sim_config, scanner_config_path, image, mu_map, measured_data):
    """
    Set up the SIMIND simulator with all configuration parameters.
    
    Parameters
    ----------
    sim_config : dict
        Simulation configuration dictionary
    scanner_config_path : str
        Path to scanner configuration file
    image : ImageData
        Source image
    mu_map : ImageData
        Attenuation map
    measured_data : AcquisitionData
        Template sinogram data
        
    Returns
    -------
    SimindSimulator
        Configured simulator
    """
    print("Setting up SIMIND simulator...")
    
    # Create scanner config file in output directory
    scanner_config = SimulationConfig(scanner_config_path)
    # Add comment
    scanner_config.set_comment("Demonstration of SIMIND simulation with refactored architecture")

    # Initialize simulator with the new architecture
    simulator = SimindSimulator(
        config_source=scanner_config,
        output_dir=sim_config['output_dir'],
        output_prefix=sim_config['output_prefix'],
        photon_multiplier=sim_config['photon_multiplier']  # Set directly in constructor
    )

    # Set basic inputs using the new API
    simulator.set_source(image)
    simulator.set_mu_map(mu_map)
    simulator.set_template_sinogram(measured_data)
    
    # Set energy windows using the new API
    simulator.set_energy_windows(
        lower_bounds=sim_config['window_lower'],
        upper_bounds=sim_config['window_upper'],
        scatter_orders=0
    )
    
    # Add configuration parameters using the new API names
    simulator.add_config_value("photon_energy", sim_config['photon_energy'])
    simulator.add_config_value("scoring_routine", sim_config['scoring_routine'])
    simulator.add_config_value("collimator_routine", sim_config['collimator_routine'])
    simulator.add_config_value("photon_direction", sim_config['photon_direction'])
    
    # Calculate source activity
    total_source_activity = sim_config['total_activity'] * sim_config['time_per_projection']
    simulator.add_config_value("source_activity", total_source_activity)
    
    # Set step size based on image resolution
    min_voxel_size = min(image.voxel_sizes())
    step_size = min_voxel_size / 5
    simulator.add_config_value("step_size_photon_path_simulation", step_size)
    
    # Set cutoff energy
    cutoff_energy = sim_config['window_lower'] * 0.75
    simulator.add_config_value("cutoff_energy_terminate_photon_history", cutoff_energy)
    
    # Add runtime switches using the updated API
    simulator.add_runtime_switch("CC", sim_config['collimator'])
    simulator.add_runtime_switch("FI", sim_config['source_type'])
    # Note: NN (photon_multiplier) is already set in constructor

    
    print(f"Simulator configured with:")
    print(f"  - Energy window: {sim_config['window_lower']}-{sim_config['window_upper']} keV")
    print(f"  - Photon multiplier: {sim_config['photon_multiplier']}")
    print(f"  - Collimator: {sim_config['collimator']}")
    print(f"  - Source activity: {total_source_activity:.2e}")
    print(f"  - Step size: {step_size:.3f} mm")
    
    return simulator


def run_simulation_with_error_handling(simulator):
    """
    Run simulation with proper error handling for the new architecture.
    
    Parameters
    ----------
    simulator : SimindSimulator
        Configured simulator
        
    Returns
    -------
    dict
        Dictionary with simulation outputs
    """
    print("Running SIMIND simulation...")
    
    try:
        # Run simulation using the new API
        simulator.run_simulation()
        print("SIMIND simulation completed successfully")
        
        # Get outputs using the new API method names
        simind_total = simulator.get_total_output(window=1)
        simind_scatter = simulator.get_scatter_output(window=1)
        
        # Calculate true (primary) counts
        simind_true = simind_total - simind_scatter
        
        outputs = {
            'total': simind_total,
            'scatter': simind_scatter,
            'true': simind_true
        }
        
        # Log count statistics
        print(f"Simulation results:")
        for key, data in outputs.items():
            print(f"  - {key.capitalize()} counts: {data.sum():.0f}")
        
        return outputs
        
    except ValueError as e:
        print(f"Configuration/validation error: {e}")
        raise
    except subprocess.CalledProcessError as e:
        print(f"SIMIND execution failed: {e}")
        raise
    except FileNotFoundError as e:
        print(f"Required file not found: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during simulation: {e}")
        raise


def save_count_statistics(outputs, measured_data, base_filename, output_dir):
    """Save count statistics to CSV file."""
    counts = {
        "simind_total": outputs['total'].sum(),
        "simind_true": outputs['true'].sum(),
        "simind_scatter": outputs['scatter'].sum(),
        "measured_total": measured_data.sum(),
    }
    
    # Calculate scatter fraction
    if counts["simind_total"] > 0:
        counts["scatter_fraction"] = counts["simind_scatter"] / counts["simind_total"]
    else:
        counts["scatter_fraction"] = 0
    
    # Save to CSV
    csv_path = os.path.join(output_dir, base_filename + ".csv")
    pd.DataFrame([counts]).to_csv(csv_path, index=False)
    print(f"Count statistics saved to: {csv_path}")
    
    return counts


def plot_comparison(data_list, slice_index, orientation, base_output_filename, output_dir,
                    profile_method='index', profile_index=60, font_size=14, colormap='viridis'):
    """
    Plot slice comparisons of the sinograms (axial or coronal) with an image grid and a line plot.
    """
    # Convert to int in case it's a float from config
    slice_index = int(slice_index)
    profile_index = int(profile_index)

    # Determine vmax over all datasets for consistent color scaling
    if orientation == 'axial':
        vmax = max(data[0][slice_index].max() for data, _ in data_list)
    elif orientation == 'coronal':
        vmax = max(data[0][:, :, slice_index].max() for data, _ in data_list)
    else:
        raise ValueError("orientation must be 'axial' or 'coronal'")

    n = len(data_list)
    fig = plt.figure(figsize=(n * 4, 14))
    gs = GridSpec(3, n, height_ratios=[2, 0.15, 3])

    # Row of images
    ax_images = [fig.add_subplot(gs[0, i]) for i in range(n)]
    for i, (data, title) in enumerate(data_list):
        arr = data[0]
        if orientation == 'axial':
            slice_img = arr[slice_index, :, :]
        else:  # coronal
            slice_img = arr[:, :, slice_index]

        im = ax_images[i].imshow(slice_img, vmin=0, vmax=vmax, cmap=colormap)
        total_counts = np.trunc(arr.sum())
        ax_images[i].set_title(f"{title}: {total_counts:.0f}", fontsize=font_size)
        ax_images[i].axis('off')

    # Colorbar spanning entire row
    cbar_ax = fig.add_subplot(gs[1, :])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', pad=0.02)
    cbar_ax.set_xlabel('Counts', fontsize=font_size)
    cbar_ax.xaxis.set_label_position('top')

    # Line plot row
    ax_line = fig.add_subplot(gs[2, :])
    colours = plt.cm.get_cmap(colormap)(np.linspace(0, 1, n))
    for i, (data, title) in enumerate(data_list):
        arr = data[0]
        if orientation == 'axial':
            slice_img = arr[slice_index, :, :]
        else:
            slice_img = arr[:, :, slice_index]

        if profile_method == 'index':
            profile = slice_img[profile_index, :]
            ylabel = f'Intensity (row {profile_index})'
        elif profile_method == 'sum':
            profile = slice_img.sum(axis=0)
            ylabel = 'Summed Intensity'
        else:
            raise ValueError("profile_method must be 'index' or 'sum'")

        ax_line.plot(profile, linewidth=2, color=colours[i], linestyle='-',
                     label=title)

    ax_line.set_xlabel('Projection angle', fontsize=font_size)
    ax_line.set_ylabel(ylabel, fontsize=font_size)
    ax_line.set_title(f'Profile Through Sinogram ({profile_method})', fontsize=font_size + 2)
    ax_line.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_line.legend(loc='upper left', fontsize=font_size)
    ax_line.set_xlim(0, slice_img.shape[1])

    plt.tight_layout()

    # Save plot
    fname = f"comparison_{orientation}_{profile_method}_{base_output_filename}.png"
    filename_full = os.path.join(output_dir, fname)
    plt.savefig(filename_full, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {fname}")


def generate_plots(outputs, measured_data, sim_config, base_filename):
    """Generate all comparison plots."""
    print("Generating comparison plots...")
    
    # Prepare data for plotting
    data_list = [
        (outputs['total'].as_array(), "simind total"),
        (measured_data.as_array(), "measured"),
        (outputs['true'].as_array(), "simind true"),
        (outputs['scatter'].as_array(), "simind scatter"),
    ]
    
    # Filter out None values
    data_list = [(data, title) for data, title in data_list if data is not None]
    
    slice_index = sim_config['axial_slice']
    
    # Generate all plot combinations
    orientations = ['axial', 'coronal']
    methods = ['sum', 'index']
    
    for orientation in orientations:
        for method in methods:
            plot_comparison(
                data_list, slice_index,
                orientation=orientation,
                base_output_filename=base_filename,
                output_dir=sim_config['output_dir'],
                profile_method=method,
                profile_index=60,
                font_size=14,
                colormap='viridis'
            )
    
    print(f"Generated {len(orientations) * len(methods)} comparison plots")


def main(args):
    """Main simulation workflow."""
    print("Starting SIMIND simulation workflow...")
    
    # Load configurations
    print(f"Loading simulation config: {args.simulation_config}")
    sim_config = load_simulation_config(args.simulation_config)
    
    # Ensure output directory exists
    os.makedirs(sim_config['output_dir'], exist_ok=True)
    
    # Prepare data
    image = prepare_image_data(sim_config['image_path'])
    mu_map = prepare_mu_map(
        sim_config['mu_map_filename'], 
        sim_config['data_dir'], 
        image
    )
    
    measured_data_path = os.path.join(
        sim_config['data_dir'], sim_config['measured_data_filename']
    )
    print(f"Loading measured data: {measured_data_path}")
    measured_data = AcquisitionData(measured_data_path)
    
    # Change working directory if needed
    original_cwd = os.getcwd()
    if sim_config['simind_parent_dir']:
        print(f"Changing to SIMIND directory: {sim_config['simind_parent_dir']}")
        os.chdir(sim_config['simind_parent_dir'])
    
    try:
        # Set up and run simulation
        simulator = setup_simulator(
            sim_config, args.scanner_config, 
            image, mu_map, measured_data
        )
        
        outputs = run_simulation_with_error_handling(simulator)
        
        # Generate output filename base
        base_filename = (f"NN{sim_config['photon_multiplier']}_"
                        f"CC{sim_config['collimator']}_"
                        f"FI{sim_config['source_type']}")
        
        # Save statistics and generate plots
        counts = save_count_statistics(
            outputs, measured_data, base_filename, sim_config['output_dir']
        )
        
        generate_plots(outputs, measured_data, sim_config, base_filename)
        
        print(f"\nSimulation completed successfully!")
        print(f"Output directory: {sim_config['output_dir']}")
        print(f"Scatter fraction: {counts['scatter_fraction']:.3f}")
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run a simulation using SIMIND and STIR (updated for refactored architecture)'
    )
    
    parser.add_argument('--simulation_config', type=str, required=True,
                        help='Path to simulation YAML configuration file')
    parser.add_argument('--scanner_config', type=str, required=True,
                        help='Path to scanner YAML configuration file')

    args = parser.parse_args()

    try:
        start_time = time.time()
        main(args)
        elapsed_time = time.time() - start_time
        print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
        
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Simulation error: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
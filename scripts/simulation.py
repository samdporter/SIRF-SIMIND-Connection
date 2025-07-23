#!/usr/bin/env python
# coding: utf-8
"""
Run a simulation using SIMIND and STIR,
generate simulated sinograms and compare with measured data.
"""

import os
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sirf.STIR import (ImageData, AcquisitionData, SPECTUBMatrix,
                       AcquisitionModelUsingMatrix, MessageRedirector)
from sirf_simind_connection import SimindSimulator


msg = MessageRedirector()


def get_acquisition_model(measured_data, additive_data, image, mu_map_stir):
    """
    Create and set up the SPECT acquisition model.

    Parameters
    ----------
    measured_data : AcquisitionData
    additive_data : AcquisitionData
    image : ImageData
    mu_map_stir : ImageData

    Returns
    -------
    AcquisitionModelUsingMatrix
    """
    acq_matrix = SPECTUBMatrix()
    acq_matrix.set_attenuation_image(mu_map_stir)
    acq_matrix.set_keep_all_views_in_cache(True)
    acq_matrix.set_resolution_model(0.9323, 0.03, True)
    acq_model = AcquisitionModelUsingMatrix(acq_matrix)
    try:
        acq_model.set_additive_term(additive_data)
    except Exception as e:
        print(e)
        print("Could not set additive data")
    acq_model.set_up(measured_data, image)
    return acq_model


def lower_threshold_image(image, threshold):
    """
    Zero out image values below the threshold to save simulation time.

    Parameters
    ----------
    image : ImageData
    threshold : float

    Returns
    -------
    ImageData
        Modified image with low values set to zero.
    """
    image_array = image.as_array()
    image_array[image_array < threshold] = 0
    image.fill(image_array)
    return image


def plot_comparison(data_list, slice_index, orientation, base_output_filename, output_dir,
                    profile_method='index', profile_index=60, font_size=14, colormap='viridis'):
    """
    Plot slice comparisons of the sinograms (axial or coronal) with an image grid and a line plot.

    Parameters
    ----------
    data_list : list of tuple
        List of tuples (data_array, title) for each dataset. data_array is a 3D numpy array.
    slice_index : int
        Slice index along the chosen orientation to display (axial or coronal).
    orientation : str
        Either 'axial' or 'coronal'. If 'axial', slices are taken as data[:, slice_index, :];
        if 'coronal', slices are taken as data[:, :, slice_index].
    base_output_filename : str
        Base string to prepend to the output filename.
    output_dir : str
        Directory where the output image is saved.
    profile_method : {'index', 'sum'}, optional
        If 'index', extract a single‐row profile at profile_index; 
        if 'sum', sum across the first axis of the 2D slice to get a profile vs. projection angle.
    profile_index : int, optional
        Row index at which to extract a 1D profile when profile_method='index'.
    font_size : int, optional
        Font size for titles and labels.
    colormap : str, optional
        Colormap to use for the images.
    """
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
        ax_images[i].set_title(f"{title}: {total_counts}", fontsize=font_size)
        ax_images[i].axis('off')

    # Colorbar spanning entire row
    cbar_ax = fig.add_subplot(gs[1, :])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', pad=0.02)
    cbar_ax.set_xlabel('Counts', fontsize=font_size)
    cbar_ax.xaxis.set_label_position('top')

    # Line‐plot row
    ax_line = fig.add_subplot(gs[2, :])
    colours = plt.cm.get_cmap(colormap)(np.linspace(0, 1, n))
    for i, (data, title) in enumerate(data_list):
        arr = data[0]
        if orientation == 'axial':
            slice_img = arr[slice_index, :, :]
            # slice_img shape: (num_rows, num_angles)
        else:
            slice_img = arr[:, :, slice_index]
            # slice_img shape: (num_rows, num_angles)

        if profile_method == 'index':
            profile = slice_img[profile_index, :]
        elif profile_method == 'sum':
            profile = slice_img.sum(axis=0)
        else:
            raise ValueError("profile_method must be 'index' or 'sum'")

        ax_line.plot(profile, linewidth=2, color=colours[i], linestyle='-',
                     label=title)

    ax_line.set_xlabel('Projection angle', fontsize=font_size)
    ax_line.set_ylabel('Intensity', fontsize=font_size)
    ax_line.set_title('Profile Through Sinogram', fontsize=font_size + 2)
    ax_line.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_line.legend(loc='upper left', fontsize=font_size)
    ax_line.set_xlim(0, slice_img.shape[1])

    plt.tight_layout()

    # Choose filename based on orientation
    if orientation == 'axial':
        fname = f"comparison_axial_{profile_method}_{base_output_filename}.png"
    else:
        fname = f"comparison_coronal_{profile_method}_{base_output_filename}.png"
    filename_full = os.path.join(output_dir, fname)

    plt.savefig(filename_full)
    plt.close()



def main(args):
    # Read images and data.
    image = ImageData(args.image_path)
    image = lower_threshold_image(image, 0.01 * image.max())
    mu_map = ImageData(args.mu_map_path)
    measured_data = AcquisitionData(args.measured_data_path)

    # Change working directory if needed.
    os.chdir(args.simind_parent_dir)

    # Set up simulator.
    simulator = SimindSimulator(
        template_smc_file_path=args.input_smc_file_path,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        source=image,
        mu_map=mu_map,
        template_sinogram=args.measured_data_path,
    )

    simulator.add_comment("Demonstration of SIMIND simulation")
    simulator.set_windows(args.window_lower, args.window_upper, 0)
    simulator.add_index("photon_energy", args.photopeak_energy)
    simulator.add_index("scoring_routine", args.scoring_routine)
    simulator.add_index("collimator_routine", args.collimator_routine)
    simulator.add_index("photon_direction", args.photon_direction)
    simulator.add_index("source_activity", args.total_activity * args.time_per_projection)
    simulator.add_index("crystal_thickness", args.crystal_thickness / 10)
    simulator.add_index("crystal_half_length_radius", args.crystal_half_length_radius / 10)
    simulator.add_index("crystal_half_width", args.crystal_half_width / 10)
    simulator.config.set_flag(11, args.flag_11)
    simulator.add_index("step_size_photon_path_simulation",
                        min(*image.voxel_sizes()) / 10)
    simulator.add_index("energy_resolution", 9.5)
    simulator.add_index("intrinsic_resolution", 0.28)
    simulator.add_index("cutoff_energy_terminate_photon_history", args.window_lower * 0.5)

    simulator.add_runtime_switch("CC", args.collimator)
    simulator.add_runtime_switch("NN", args.photon_multiplier)
    simulator.add_runtime_switch("FI", args.source_type)

    simulator.run_simulation()

    # Process simulation outputs.
    simind_total = simulator.get_output_total()
    simind_scatter = simulator.get_output_scatter()
    simind_true = simind_total - simind_scatter

    base_output_filename = (f"NN{args.photon_multiplier}_CC{args.collimator}_"
                            f"FI{args.source_type}_")

    counts = {
        "simind_total": simind_total.sum(),
        "simind_true": simind_true.sum(),
        "simind_scatter": simind_scatter.sum(),
    }
    pd.DataFrame([counts]).to_csv(
        os.path.join(args.output_dir, base_output_filename + ".csv"))

    # Prepare data for plotting.
    data_list = [
        (simind_total.as_array(), "simind total"),
        (measured_data.as_array(), "measured"),
        (simind_true.as_array(), "simind true"),
        (simind_scatter.as_array(), "simind scatter"),
    ]
    # Filter out None values.
    data_list = [(data, title) for data, title in data_list if data is not None]

    # Plot axial slice comparisons.
    plot_comparison(
        data_list, args.axial_slice,
        orientation='axial',
        base_output_filename=base_output_filename, output_dir=args.output_dir,
        profile_method='sum', font_size=14, colormap='viridis'
    )
    plot_comparison(
        data_list, args.axial_slice,
        orientation='axial',
        base_output_filename=base_output_filename, output_dir=args.output_dir,
        profile_method='index', profile_index=60, font_size=14, colormap='viridis'
    )
    # Plot coronal slice comparisons.
    plot_comparison(
        data_list, args.axial_slice,
        orientation='coronal',
        base_output_filename=base_output_filename, output_dir=args.output_dir,
        profile_method='sum', font_size=14, colormap='viridis'
    )
    plot_comparison(
        data_list, args.axial_slice,
        orientation='coronal',
        base_output_filename=base_output_filename, output_dir=args.output_dir,
        profile_method='index', profile_index=60, font_size=14, colormap='viridis'
    )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run a simulation using SIMIND and STIR'
    )
    parser.add_argument('--total_activity', type=float, default=258.423,
                        help='Total activity in MBq')
    parser.add_argument('--time_per_projection', type=int, default=43,
                        help='Time per projection in seconds')
    parser.add_argument('--photon_multiplier', type=float, default=10.,
                        help=('Number of photons simulated is calculated based on source '
                              'map. This number multiplies the calculated number of photons'))
    parser.add_argument('--photopeak_energy', type=float, default=208,
                        help='Photopeak energy in keV')
    parser.add_argument('--window_lower', type=float, default=187.56,
                        help='Lower window in keV')
    parser.add_argument('--window_upper', type=float, default=229.24,
                        help='Upper window in keV')
    parser.add_argument('--source_type', type=str, default='lu177',
                        help='Source type')
    parser.add_argument('--collimator', type=str, default='G8-MEGP',
                        help='Collimator')
    parser.add_argument('--kev_per_channel', type=float, default=10.,
                        help='keV per channel')
    parser.add_argument('--max_energy', type=float, default=498.3,
                        help='Max energy in keV')
    parser.add_argument('--mu_map_path', type=str,
                        default='data/Lu177/registered_CTAC.hv',
                        help='Path to mu map')
    parser.add_argument('--image_path', type=str,
                        default='data/Lu177/osem_image.hv',
                        help='Path to image')
    parser.add_argument('--simind_parent_dir', type=str, default='.',
                        help='Parent directory for SIMIND simulation')
    parser.add_argument('--measured_data_path', type=str,
                        default='data/Lu177/SPECTCT_NEMA_128_EM001_DS_en_1_Lu177_EM.hdr',
                        help='Path to measured data')
    parser.add_argument('--output_dir', type=str, default='simind_output',
                        help='Output directory')
    parser.add_argument('--output_prefix', type=str, default='output',
                        help='Output prefix')
    parser.add_argument('--input_smc_file_path', type=str,
                        default='input/input.smc',
                        help='Path to input smc file')
    parser.add_argument('--scoring_routine', type=int, default=1,
                        help='Scoring routine')
    parser.add_argument('--collimator_routine', type=int, default=0,
                        help='Collimator routine')
    parser.add_argument('--photon_direction', type=int, default=2,
                        help='Photon direction')
    parser.add_argument('--crystal_thickness', type=float, default=7.25,
                        help='Crystal thickness in mm')
    parser.add_argument('--crystal_half_length_radius', type=float,
                        default=393.6 / 2,
                        help='Crystal half length radius in mm')
    parser.add_argument('--crystal_half_width', type=float,
                        default=511.7 / 2,
                        help='Crystal half width in mm')
    parser.add_argument('--flag_11', type=bool, default=True,
                        help='Flag 11 - use collimator')
    parser.add_argument('--half_life', type=float,
                        default=6.647 * 24,
                        help='Half life of the isotope in hours')
    parser.add_argument('--axial_slice', type=int, default=65,
                        help='Axial slice to plot')

    args = parser.parse_args()

    try:

        start_time = time.time()

        main(args)

        print(
            "Simulation completed successfully!" 
            "Time taken: {:.2f} seconds".format(
            time.time() - start_time)
        )
    except Exception as e:
        print(e)
        raise e

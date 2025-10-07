import subprocess
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import cmasher
import matplotlib.pyplot as plt
import numpy as np
from sirf.STIR import AcquisitionData

from sirf_simind_connection.converters.simind_to_stir import SimindToStirConverter
from sirf_simind_connection.utils import get_array
from sirf_simind_connection.utils.simind_utils import create_window_file


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class EnergyWindow:
    """Define an energy window for simulation."""

    name: str
    low_kev: float
    high_kev: float

    def label(self) -> str:
        """Return label for plotting (using lower bound)."""
        return f"{self.low_kev:g}"

    def full_label(self) -> str:
        """Return full label with both bounds for plotting."""
        return f"{self.low_kev:g}-{self.high_kev:g}"


@dataclass
class SimulationConfig:
    """Central configuration for SIMIND simulations."""

    # Energy windows
    energy_windows: List[EnergyWindow] = None

    # Simulation parameters
    num_cores: int = 2
    change_filename: str = "simind.smc"
    collimator: str = "ma-mlegp"
    voxel_size: float = 0.2
    num_histories: int = 10
    photon_multiplier: int = 100000

    # Sweep parameters
    distances: List[float] = None
    penetration_options: List[bool] = None
    source_types: List[str] = None

    # Plotting parameters
    plot_y_index: int = 32
    plot_y_range: int = 1
    plot_figsize: Tuple[int, int] = (5, 5)

    # FWHM calculation parameters
    pixel_spacing_mm: float = 2.0

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.energy_windows is None:
            self.energy_windows = [
                EnergyWindow("w1", 36.84, 126.16),
                EnergyWindow("w2", 50.0, 150.0),
                EnergyWindow("w3", 75.0, 225.0),
            ]

        if self.distances is None:
            self.distances = [2, 5, 10, 20]

        if self.penetration_options is None:
            self.penetration_options = [True]  # , False]

        if self.source_types is None:
            self.source_types = ["y90_frey_no_bremss", "y90_frey"]


# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================


def run_simind_process(
    config: SimulationConfig,
    source_type: str,
    distance_to_col: float,
    penetration: bool,
) -> None:
    """Run SIMIND simulation with specified parameters."""
    penetration_switch = 1 if penetration else 0

    opts = (
        f"/MP"
        f"/CC:{config.collimator}"
        f"/PX:{config.voxel_size}"
        f"/TH:{config.voxel_size}"
        f"/26:{config.num_histories}"
        f"/NN:{config.photon_multiplier}"
        f"/CA:1"
        f"/FI:{source_type}"
        f"/12:{distance_to_col}"
        f"/53:{penetration_switch}"
    )

    args = [
        "mpirun",
        "-np",
        str(config.num_cores),
        "simind_mpi",
        config.change_filename,
        opts,
    ]

    subprocess.run(args, check=True)


def create_energy_window_file(
    windows: List[EnergyWindow], output_filename: str = "simind.win"
) -> None:
    """Create SIMIND energy window file from window definitions."""
    create_window_file(
        [w.low_kev for w in windows],
        [w.high_kev for w in windows],
        [0] * len(windows),
        output_filename=output_filename,
    )


def convert_simind_to_stir(
    converter: SimindToStirConverter,
    window_name: str,
    output_dir: Path,
) -> Tuple[str, str]:
    """Convert SIMIND .h00 files to STIR format.

    Returns:
        Tuple of (total_filepath, scatter_filepath)
    """
    h00_tot = f"simind_tot_{window_name}.h00"
    h00_sca = f"simind_sca_{window_name}.h00"
    hs_tot = output_dir / f"simind_tot_{window_name}.hs"
    hs_sca = output_dir / f"simind_sca_{window_name}.hs"

    tot = converter.convert_file(h00_tot, return_object=True)
    sca = converter.convert_file(h00_sca, return_object=True)

    tot.write(str(hs_tot))
    sca.write(str(hs_sca))

    return str(hs_tot), str(hs_sca)


def load_and_compute_primaries(
    windows: List[EnergyWindow],
    output_dir: Path,
) -> Tuple[
    Dict[str, AcquisitionData], Dict[str, AcquisitionData], Dict[str, AcquisitionData]
]:
    """Load total and scatter data, compute primary = total - scatter.

    Returns:
        Tuple of (totals, scatters, primaries) dictionaries keyed by window label
    """
    totals = {}
    scatters = {}
    primaries = {}

    for window in windows:
        label = window.label()
        hs_tot = output_dir / f"simind_tot_{window.name}.hs"
        hs_sca = output_dir / f"simind_sca_{window.name}.hs"

        atot = AcquisitionData(str(hs_tot))
        asca = AcquisitionData(str(hs_sca))
        apri = atot - asca

        totals[label] = atot
        scatters[label] = asca
        primaries[label] = apri

    return totals, scatters, primaries


def save_acquisition_data(
    data_dict: Dict[str, AcquisitionData], prefix: str, tag: str
) -> None:
    """Save acquisition data to files."""
    for label, data in data_dict.items():
        filename = f"{prefix}_{tag}"
        data.write(filename)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================


def plot_sinos(
    sinos: dict, y_idx: int = 32, y_range: int = 5, windows: List[EnergyWindow] = None
):
    """Plot sinograms with profiles.

    Args:
        sinos: Dictionary of sinogram data keyed by window label
        y_idx: Center y-index for profile averaging
        y_range: Range above/below y_idx to average (total width = y_range)
        windows: List of EnergyWindow objects for proper labeling (optional)
    """
    from matplotlib.patches import Rectangle

    # Create mapping from label to full label if windows provided
    label_map = {}
    if windows is not None:
        label_map = {w.label(): w.full_label() for w in windows}

    # Create figure with two separate subplot areas
    fig = plt.figure(figsize=(15, 10))

    # Top row: 3 sinograms
    gs_top = fig.add_gridspec(
        1, 3, top=0.58, bottom=0.35, left=0.05, right=0.98, wspace=0.3
    )
    axes_top = [fig.add_subplot(gs_top[0, i]) for i in range(3)]

    # Bottom: single profile plot
    gs_bottom = fig.add_gridspec(1, 1, top=0.28, bottom=0.05, left=0.08, right=0.95)
    ax_profile = fig.add_subplot(gs_bottom[0, 0])

    # Plot sinograms
    for i, (label, sino) in enumerate(sinos.items()):
        if i >= 3:  # Only plot first 3
            break
        arr = get_array(sino)
        im = axes_top[i].imshow(arr[0, :, 1], cmap=cmasher.fall)

        # Use full label if available
        display_label = label_map.get(label, label)
        axes_top[i].set_title(f"{display_label} keV", fontsize=12, pad=10)
        fig.colorbar(im, ax=axes_top[i], fraction=0.046, pad=0.04)

        if y_range != 1:
            # Add translucent rectangle showing averaging region
            img_height, img_width = arr[0, :, 1].shape
            start = y_idx - y_range // 2
            stop = y_idx + y_range // 2 + 1  # exclusive

            rect = Rectangle(
                xy=(0, start),
                width=img_width,
                height=(stop - start),
                linewidth=2,
                edgecolor="yellow",
                facecolor="yellow",
                alpha=0.2,
            )
            axes_top[i].add_patch(rect)

        else:
            axes_top[i].axhline(
                y=y_idx, color="yellow", linestyle="--", linewidth=2, alpha=0.7
            )

    if y_range != 1:
        # Add legend for the averaging region
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(
                facecolor="yellow",
                edgecolor="yellow",
                alpha=0.2,
                label=f"Profile region (y={start}:{stop - 1})",
            )
        ]
        axes_top[0].legend(handles=legend_elements, loc="upper right", fontsize=9)
    else:
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                color="yellow",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label=f"Profile line (y={y_idx})",
            )
        ]
        axes_top[0].legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Plot profiles
    for label, sino in sinos.items():
        arr = get_array(sino)[0]
        centre_line = np.sum(
            np.nan_to_num(arr[y_idx - y_range // 2 - 1 : y_idx + y_range // 2, :]),
            axis=(0, 1),
        )
        norm_line = centre_line / np.max(centre_line)

        display_label = label_map.get(label, label)
        ax_profile.plot(norm_line, label=f"{display_label} keV", linewidth=2)

    ax_profile.set_title(
        f"Normalised central profiles (averaged over y={start}:{stop - 1})"
        if y_range != 1
        else f"Normalised central profiles (y={y_idx})",
        fontsize=11,
        pad=10,
    )
    ax_profile.legend(fontsize=10)
    ax_profile.set_xlabel("Pixel index", fontsize=10)
    ax_profile.set_ylabel("Normalised intensity", fontsize=10)
    ax_profile.grid(alpha=0.3)

    return fig


def calculate_fwhm(x, y, peak_idx):
    """Calculate FWHM for a peak at given index."""
    try:
        peak_height = y[peak_idx]
        half_max = peak_height / 2

        # Find half-max crossing on left side
        left_idx = peak_idx
        while left_idx > 0 and y[left_idx] > half_max:
            left_idx -= 1

        # Find half-max crossing on right side
        right_idx = peak_idx
        while right_idx < len(y) - 1 and y[right_idx] > half_max:
            right_idx += 1

        # Interpolate for precise positions
        if left_idx < peak_idx and abs(y[left_idx + 1] - y[left_idx]) > 1e-10:
            x1, x2 = x[left_idx], x[left_idx + 1]
            y1, y2 = y[left_idx], y[left_idx + 1]
            left_x = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
        else:
            left_x = x[left_idx]

        if right_idx > peak_idx and abs(y[right_idx] - y[right_idx - 1]) > 1e-10:
            x1, x2 = x[right_idx - 1], x[right_idx]
            y1, y2 = y[right_idx - 1], y[right_idx]
            right_x = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
        else:
            right_x = x[right_idx]

        fwhm = right_x - left_x
        return fwhm, left_x, right_x, half_max

    except:
        return None, None, None, None


def plot_peak_profiles_arrays(
    arrays, *, labels=None, y_idx=32, y_range=5, figsize=(18, 3), pixel_spacing_mm=None
):
    """Plot peak profiles with FWHM of the largest peak.

    Args:
        arrays: List of 3D arrays to plot profiles from
        labels: Labels for each array series
        y_idx: Center y-index for profile averaging
        y_range: Range above/below y_idx to average
        figsize: Figure size
        pixel_spacing_mm: Physical size of each pixel in mm (if provided, FWHM shown in mm)
    """
    n_series = len(arrays)
    if labels is None:
        labels = [f"Series {i}" for i in range(n_series)]

    fig, ax = plt.subplots(1, 1, figsize=figsize, sharey=True)

    colors = plt.cm.tab10(np.linspace(0, 1, n_series))
    all_fwhms = []

    for i, (arr, lab) in enumerate(zip(arrays, labels)):
        # Extract and sum profile
        profile = np.sum(
            np.nan_to_num(arr[0, y_idx - y_range // 2 - 1 : y_idx + y_range // 2, :]),
            axis=(0, 1),
        )
        x = np.arange(len(profile))

        color = colors[i]
        ax.plot(x, profile, label=lab, color=color)

        # Find the largest peak
        peak_idx = np.argmax(profile)

        # Calculate FWHM
        fwhm, left_x, right_x, half_max = calculate_fwhm(x, profile, peak_idx)

        if fwhm is not None:
            # Plot peak marker
            ax.plot(
                x[peak_idx],
                profile[peak_idx],
                "o",
                color=color,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=2,
            )

            # Plot FWHM line
            ax.plot(
                [left_x, right_x],
                [half_max, half_max],
                color=color,
                linewidth=2,
                alpha=0.8,
            )

            # Format FWHM value
            if pixel_spacing_mm is not None:
                fwhm_display = fwhm * pixel_spacing_mm
                fwhm_units = "mm"
            else:
                fwhm_display = fwhm
                fwhm_units = "px"

            all_fwhms.append(fwhm_display)

            # Annotate FWHM
            ax.annotate(
                f"FWHM: {fwhm_display:.1f}{fwhm_units}",
                xy=(x[peak_idx], profile[peak_idx]),
                xytext=(x[peak_idx], profile[peak_idx] + 0.1),
                va="center",
                ha="right",
                fontsize=8,
                color=color,
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=color,
                    alpha=0.9,
                ),
            )

    ax.set_title(f"(y={y_idx - y_range // 2 - 1}:{y_idx + y_range // 2})")
    ax.set_xlabel("x-pixel")
    ax.grid(alpha=0.3)

    ax.set_ylabel("Normalized counts")
    fig.legend(loc="lower center", ncol=min(6, n_series), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig, all_fwhms


# ============================================================================
# WORKFLOW FUNCTIONS
# ============================================================================


def process_simulation_case(
    config: SimulationConfig,
    converter: SimindToStirConverter,
    distance: float,
    penetration: bool,
    source_type: str,
) -> bool:
    """Process a single simulation case.

    Returns:
        True if successful, False otherwise
    """
    tag = f"d{distance}_pen{int(penetration)}_{source_type}"
    output_dir = Path(f"sim_{tag}")
    output_dir.mkdir(exist_ok=True)

    # Run simulation
    try:
        run_simind_process(config, source_type, distance, penetration)
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {tag}: {e}")
        return False

    print(f"[OK] {tag}")

    # Convert files
    for window in config.energy_windows:
        convert_simind_to_stir(converter, window.name, output_dir)

    # Load and compute primaries
    totals, scatters, primaries = load_and_compute_primaries(
        config.energy_windows, output_dir
    )

    # Save data
    save_acquisition_data(totals, "totals", tag)
    save_acquisition_data(scatters, "scatter", tag)
    save_acquisition_data(primaries, "primary", tag)

    # Generate plots
    plot_sinos(
        primaries,
        y_idx=config.plot_y_index,
        y_range=config.plot_y_range,
        windows=config.energy_windows,
    ).savefig(output_dir / f"profiles_primary_{tag}.png", dpi=200)
    plot_sinos(
        totals,
        y_idx=config.plot_y_index,
        y_range=config.plot_y_range,
        windows=config.energy_windows,
    ).savefig(output_dir / f"profiles_total_{tag}.png", dpi=200)
    plot_sinos(
        scatters,
        y_idx=config.plot_y_index,
        y_range=config.plot_y_range,
        windows=config.energy_windows,
    ).savefig(output_dir / f"profiles_scatter_{tag}.png", dpi=200)
    plt.close("all")

    # Generate FWHM plots
    sorted_labels = sorted(primaries.keys(), key=lambda x: int(float(x)))
    fig, fwhms = plot_peak_profiles_arrays(
        arrays=[get_array(primaries[label]) for label in sorted_labels],
        labels=sorted_labels,
        y_idx=config.plot_y_index,
        y_range=config.plot_y_range,
        figsize=config.plot_figsize,
        pixel_spacing_mm=config.pixel_spacing_mm,
    )
    fig.savefig(output_dir / f"fwhm_profiles_primary_{tag}.png", dpi=200)
    plt.close("all")

    # Log FWHM results
    fwhm_log_path = output_dir / f"fwhm_results_{tag}.txt"
    with open(fwhm_log_path, "w") as f:
        f.write("FWHM Results (Primary Data)\n")
        f.write(
            f"Configuration: distance={distance}, penetration={penetration}, source_type={source_type}\n"
        )
        f.write("Window (keV)\tFWHM values\n")
        for label, fwhm in zip(sorted_labels, fwhms):
            f.write(f"{label}\t{fwhm:.2f}\n")

    return True


def run_simulations(config: SimulationConfig) -> None:
    """Run all simulation combinations specified in config."""
    create_energy_window_file(config.energy_windows)
    converter = SimindToStirConverter()

    # Generate all combinations
    cases = product(config.distances, config.penetration_options, config.source_types)

    # Process each case
    for distance, penetration, source_type in cases:
        process_simulation_case(config, converter, distance, penetration, source_type)


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Run SIMIND simulations with default configuration."""
    config = SimulationConfig()
    run_simulations(config)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
STIR adaptor example: SIMIND simulation followed by STIR OSEM reconstruction.

This example keeps runtime small:
- tiny phantom
- low projection count
- NN = 1 (photon multiplier)

Unit note:
- STIR image/template geometry is defined in mm.
- SIMIND geometry parameters are in cm.
- The adaptor handles the mm -> cm conversion internally.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import stir
import stirextra

from sirf_simind_connection import StirSimindAdaptor, configs
from sirf_simind_connection.builders import STIRSPECTImageDataBuilder


def _build_small_phantom(
    output_dir: Path,
) -> tuple[
    stir.FloatVoxelsOnCartesianGrid,
    stir.FloatVoxelsOnCartesianGrid,
]:
    dims = (32, 32, 32)  # z, y, x
    voxel_mm = (4.0, 4.0, 4.0)

    z, y, x = np.indices(dims)
    body = ((x - 16) ** 2 + (y - 16) ** 2 <= 10**2) & (np.abs(z - 16) <= 8)
    hot = (x - 22) ** 2 + (y - 16) ** 2 + (z - 16) ** 2 <= 4**2

    source = np.zeros(dims, dtype=np.float32)
    source[body] = 0.8
    source[hot] = 2.0

    mu_map = np.zeros_like(source)
    mu_map[body] = 0.15

    source_builder = STIRSPECTImageDataBuilder(
        {
            "!matrix size [1]": str(dims[2]),
            "!matrix size [2]": str(dims[1]),
            "!matrix size [3]": str(dims[0]),
            "scaling factor (mm/pixel) [1]": str(voxel_mm[2]),
            "scaling factor (mm/pixel) [2]": str(voxel_mm[1]),
            "scaling factor (mm/pixel) [3]": str(voxel_mm[0]),
        }
    )
    source_builder.set_pixel_array(source)
    source_base = output_dir / "stir_source"
    source_builder.build(output_path=source_base)

    mu_builder = STIRSPECTImageDataBuilder(
        {
            "!matrix size [1]": str(dims[2]),
            "!matrix size [2]": str(dims[1]),
            "!matrix size [3]": str(dims[0]),
            "scaling factor (mm/pixel) [1]": str(voxel_mm[2]),
            "scaling factor (mm/pixel) [2]": str(voxel_mm[1]),
            "scaling factor (mm/pixel) [3]": str(voxel_mm[0]),
        }
    )
    mu_builder.set_pixel_array(mu_map)
    mu_base = output_dir / "stir_mu"
    mu_builder.build(output_path=mu_base)

    source_image = stir.FloatVoxelsOnCartesianGrid.read_from_file(
        str(source_base.with_suffix(".hv"))
    )
    mu_image = stir.FloatVoxelsOnCartesianGrid.read_from_file(
        str(mu_base.with_suffix(".hv"))
    )

    return source_image, mu_image


def _run_stir_osem(
    projection: stir.ProjData,
    image_template: stir.FloatVoxelsOnCartesianGrid,
    mu_map: stir.FloatVoxelsOnCartesianGrid,
) -> stir.FloatVoxelsOnCartesianGrid:
    initial = image_template.get_empty_copy()
    initial.fill(1.0)

    # With direct STIR bindings, calling intermediate set_up() on projector/objective
    # can collapse SPECTUB reconstructions to zero. Configure the chain, then set up
    # only the reconstructor.
    proj_matrix = stir.ProjMatrixByBinSPECTUB()
    proj_matrix.set_attenuation_image_sptr(mu_map)
    proj_matrix.set_resolution_model(0.0, 0.0, False)
    proj_matrix.set_keep_all_views_in_cache(True)

    projector_pair = stir.ProjectorByBinPairUsingProjMatrixByBin()
    projector_pair.set_proj_matrix_sptr(proj_matrix)

    objective = stir.PoissonLogLikelihoodWithLinearModelForMeanAndProjData3DFloat()
    objective.set_projector_pair_sptr(projector_pair)
    objective.set_proj_data_sptr(projection)
    objective.set_num_subsets(2)

    recon = stir.OSMAPOSLReconstruction3DFloat()
    recon.set_objective_function(objective)
    recon.set_input_data(projection)
    recon.set_num_subsets(2)
    recon.set_num_subiterations(2)
    recon.set_disable_output(True)
    recon.set_up(initial)
    recon.reconstruct(initial)

    return initial


def _projection_plane(projection_array: np.ndarray) -> np.ndarray:
    if projection_array.ndim == 4:
        return projection_array[0, :, 0, :]
    if projection_array.ndim == 3:
        return projection_array[0, :, :]
    return projection_array


def _save_summary_plot(
    source_image: stir.FloatVoxelsOnCartesianGrid,
    projection: stir.ProjData,
    reconstruction: stir.FloatVoxelsOnCartesianGrid,
    output_path: Path,
) -> None:
    source_arr = stirextra.to_numpy(source_image)
    proj_arr = stirextra.to_numpy(projection)
    recon_arr = stirextra.to_numpy(reconstruction)

    source_slice = source_arr[source_arr.shape[0] // 2, :, :]
    recon_slice = recon_arr[recon_arr.shape[0] // 2, :, :]
    proj_slice = _projection_plane(proj_arr)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(source_slice, cmap="viridis")
    axes[0].set_title("Input source")
    axes[1].imshow(proj_slice, cmap="magma")
    axes[1].set_title("Projection (view 0)")
    axes[2].imshow(recon_slice, cmap="viridis")
    axes[2].set_title("OSEM reconstruction")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> None:
    if shutil.which("simind") is None:
        raise RuntimeError(
            "SIMIND executable not found in PATH. "
            "Follow the official SIMIND installation instructions first."
        )

    output_dir = Path("output/stir_adaptor_osem")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_image, mu_image = _build_small_phantom(output_dir)

    adaptor = StirSimindAdaptor(
        config_source=configs.get("Example.yaml"),
        output_dir=str(output_dir),
        output_prefix="stir_case01",
        photon_multiplier=1,
        quantization_scale=0.05,
    )
    adaptor.set_source(source_image)
    adaptor.set_mu_map(mu_image)
    adaptor.set_energy_windows([75], [225], [0])
    adaptor.add_runtime_switch("FI", "y90_tissue")
    adaptor.add_runtime_switch("CC", "ma-megp")
    adaptor.add_runtime_switch("NN", 1)
    adaptor.add_runtime_switch("RR", 12345)
    adaptor.add_config_value(1, 140.0)
    adaptor.add_config_value(19, 2)
    adaptor.add_config_value(53, 0)
    adaptor.run()

    projection = adaptor.get_total_output(window=1)
    projection_path = output_dir / "stir_total.hs"
    projection.write_to_file(str(projection_path))

    reconstruction = _run_stir_osem(projection, source_image, mu_image)
    recon_path = output_dir / "stir_osem_recon.hv"
    reconstruction.write_to_file(str(recon_path))
    plot_path = output_dir / "stir_osem_summary.png"
    _save_summary_plot(source_image, projection, reconstruction, plot_path)

    print(f"Projection sum: {projection.sum():.3f}")
    print(f"Reconstruction sum: {reconstruction.sum():.3f}")
    print(f"Projection file: {projection_path}")
    print(f"Reconstruction file: {recon_path}")
    print(f"Summary plot: {plot_path}")


if __name__ == "__main__":
    main()

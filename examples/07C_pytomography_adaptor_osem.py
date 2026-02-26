#!/usr/bin/env python
"""
PyTomography adaptor example: SIMIND simulation followed by PyTomography OSEM.

This example uses only PyTomography/torch data objects in the adaptor and
reconstruction path.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytomography import algorithms, likelihoods

from sirf_simind_connection import PyTomographySimindAdaptor, configs


def _build_small_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    dims = (32, 32, 32)  # x, y, z

    x, y, z = np.indices(dims)
    body = ((x - 16) ** 2 + (y - 16) ** 2 <= 10**2) & (np.abs(z - 16) <= 8)
    hot = (x - 22) ** 2 + (y - 16) ** 2 + (z - 16) ** 2 <= 4**2

    source = np.zeros(dims, dtype=np.float32)
    source[body] = 0.8
    source[hot] = 2.0

    mu_map = np.zeros_like(source)
    mu_map[body] = 0.15

    return torch.from_numpy(source), torch.from_numpy(mu_map)


def _projection_plane(projection_array: np.ndarray) -> np.ndarray:
    if projection_array.ndim == 4:
        return projection_array[0, 0, :, :]
    if projection_array.ndim == 3:
        return projection_array[0, :, :]
    return projection_array


def _save_summary_plot(
    source_tensor: torch.Tensor,
    projection_tensor: torch.Tensor,
    reconstruction: torch.Tensor,
    output_path: Path,
) -> None:
    source_arr = source_tensor.detach().cpu().numpy()
    proj_arr = projection_tensor.detach().cpu().numpy()
    recon_arr = reconstruction.detach().cpu().numpy()

    source_slice = source_arr[:, :, source_arr.shape[2] // 2]
    proj_slice = _projection_plane(proj_arr)
    recon_slice = recon_arr[:, :, recon_arr.shape[2] // 2]

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

    output_dir = Path("output/pytomography_adaptor_osem")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_tensor, mu_tensor = _build_small_tensors()

    adaptor = PyTomographySimindAdaptor(
        config_source=configs.get("Example.yaml"),
        output_dir=str(output_dir),
        output_prefix="pytomo_case01",
        photon_multiplier=1,
        voxel_size_mm=4.0,
        quantization_scale=0.05,
    )
    adaptor.set_source(source_tensor)
    adaptor.set_mu_map(mu_tensor)
    adaptor.set_energy_windows([75], [225], [0])
    adaptor.add_runtime_switch("FI", "y90_tissue")
    adaptor.add_runtime_switch("CC", "ma-megp")
    adaptor.add_runtime_switch("NN", 1)
    adaptor.add_runtime_switch("RR", 12345)
    adaptor.add_config_value(1, 140.0)
    adaptor.add_config_value(19, 2)
    adaptor.add_config_value(53, 0)
    adaptor.run()

    projection_tensor = adaptor.get_total_output(window=1).to(dtype=torch.float32)
    if projection_tensor.ndim != 3:
        raise ValueError(
            "Expected projection tensor shape [views, bins, axial], "
            f"got {tuple(projection_tensor.shape)}"
        )

    system_matrix = adaptor.build_system_matrix(
        key="tot_w1",
        use_psf=True,
        use_attenuation=True,
    )
    total_h00_path = adaptor.get_output_header_path("tot_w1")
    likelihood = likelihoods.PoissonLogLikelihood(
        system_matrix=system_matrix,
        projections=projection_tensor,
    )
    osem = algorithms.OSEM(likelihood=likelihood)
    reconstruction = osem(2, n_subsets=4)

    projection_path = output_dir / "pytomo_total_tensor.npy"
    recon_path = output_dir / "pytomo_osem_recon.npy"
    plot_path = output_dir / "pytomo_osem_summary.png"
    np.save(projection_path, projection_tensor.detach().cpu().numpy())
    np.save(recon_path, reconstruction.detach().cpu().numpy())
    _save_summary_plot(source_tensor, projection_tensor, reconstruction, plot_path)

    print(f"Projection sum: {float(projection_tensor.sum()):.3f}")
    print(f"Reconstruction sum: {float(reconstruction.sum()):.3f}")
    print(f"Projection header: {total_h00_path}")
    print(f"Projection tensor: {projection_path}")
    print(f"Reconstruction tensor: {recon_path}")
    print(f"Summary plot: {plot_path}")


if __name__ == "__main__":
    main()

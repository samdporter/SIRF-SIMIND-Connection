import argparse
import os
import time

from sirf.STIR import *

from sirf_simind_connection.utils import get_array


def parse_spect_res(x):
    vals = x.split(",")
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("spect_res must be 3 values: float,float,bool")
    return float(vals[0]), float(vals[1]), vals[2].lower() == "true"


parser = argparse.ArgumentParser(description="Reconstruct with OSEM")

parser.add_argument(
    "--data_path",
    type=str,
    default="/home/storage/copied_data/data/phantom_data/for_cluster/SPECT",
    help="data path",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="/home/storage/copied_data/data/phantom_data/for_cluster/SPECT",
    help="output path",
)
parser.add_argument("--num_subsets", type=int, default=12, help="number of subsets")
parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs")
# default additive path to None but expect string
parser.add_argument("--additive_path", type=str, default=None, help="additive path")
parser.add_argument("--smoothing", type=bool, default=False, help="smoothing")
parser.add_argument("--index", type=int, default=0, help="index")
parser.add_argument(
    "--mu_map_filename",
    type=str,
    default="umap_zoomed.hv",
    help="mu map filename or false",
)


def get_spect_data(path, mu_map_filename="umap_zoomed.hv"):
    spect_data = {}
    spect_data["acquisition_data"] = AcquisitionData(os.path.join(path, "peak.hs"))

    # Load initial/template image first
    try:
        spect_data["initial_image"] = ImageData(
            os.path.join(path, "initial_image.hv")
        ).maximum(0)
    except:
        spect_data["initial_image"] = ImageData(os.path.join(path, "template_image.hv"))
        spect_data["initial_image"].fill(1)

    # Handle attenuation map - same logic as simulation.py
    if mu_map_filename and mu_map_filename != "false":
        spect_data["attenuation"] = ImageData(os.path.join(path, mu_map_filename))
        # Apply necessary coordinate flip
        attn_arr = get_array(spect_data["attenuation"])
        attn_arr = np.flip(attn_arr, axis=-1)
        spect_data["attenuation"].fill(attn_arr)
    else:
        # Create uniform zero attenuation map (no attenuation) for PSF scans
        spect_data["attenuation"] = spect_data["initial_image"].get_uniform_copy(0)

    return spect_data


def get_spect_am(spect_data, keep_all_views_in_cache=False):
    spect_am_mat = SPECTUBMatrix()
    spect_am_mat.set_attenuation_image(spect_data["attenuation"])
    spect_am_mat.set_keep_all_views_in_cache(keep_all_views_in_cache)
    spect_am_mat.set_resolution_model(1.22, 0.031, False)
    spect_am = AcquisitionModelUsingMatrix(spect_am_mat)
    gauss = SeparableGaussianImageFilter()
    gauss.set_fwhms((13.4, 13.4, 13.4))
    spect_am.set_image_data_processor(gauss)
    if spect_data["additive"] is not None:
        spect_am.set_additive_term(spect_data["additive"])
    return spect_am


def get_reconstructor(data, acq_model, initial_image, num_subsets, num_epochs):
    recon = OSMAPOSLReconstructor()
    recon.set_objective_function(
        make_Poisson_loglikelihood(acq_data=data, acq_model=acq_model)
    )
    recon.set_num_subsets(num_subsets)
    recon.set_num_subiterations(num_subsets * num_epochs)
    recon.set_up(initial_image)
    return recon


def main(data_path, mu_map_filename="umap_zoomed.hv"):
    spect_data = get_spect_data(data_path, mu_map_filename)
    if args.additive_path is not None:
        spect_data["additive"] = AcquisitionData(args.additive_path)
    else:
        spect_data["additive"] = None
    spect_am = get_spect_am(spect_data, True)
    spect_init = spect_data["initial_image"]
    spect_recon = get_reconstructor(
        spect_data["acquisition_data"],
        spect_am,
        spect_init,
        args.num_subsets,
        args.num_epochs,
    )
    spect_recon.reconstruct(spect_init)

    recon_image = spect_recon.get_current_estimate()

    if args.smoothing:
        gauss = SeparableGaussianImageFilter()
        gauss.set_fwhms((5, 5, 5))
        gauss.apply(recon_image)

    return recon_image


if __name__ == "__main__":
    start_time = time.time()

    msg = MessageRedirector()

    args = parser.parse_args()
    suffix = f"osem_i{args.num_epochs}_s{args.num_subsets}"

    print(
        f"Reconstructing {args.data_path} with {args.num_epochs} epochs and {args.num_subsets} subsets"
    )

    spect = main(args.data_path, args.mu_map_filename)
    if args.smoothing:
        suffix += "_smoothed"
    spect.write(os.path.join(args.output_path, f"recon_{suffix}_{args.index}.hv"))

    print(f"Reconstruction done, saved to {args.output_path}")
    print(f"Elapsed time: {time.time() - start_time} s")

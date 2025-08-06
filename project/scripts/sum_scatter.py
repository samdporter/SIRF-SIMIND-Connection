#!/usr/bin/env python3
"""
Compute mean scatter image from SIMIND scatter outputs,
normalize by predicted true counts from the forward model.
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from sirf.Reg import ImageData as RegImageData
from sirf.STIR import (
    AcquisitionData,
    AcquisitionModelUsingMatrix,
    ImageData,
    SeparableGaussianImageFilter,
    SPECTUBMatrix,
)
from skimage.morphology import ball, erosion


# Try to import totalsegmentator, but don't fail if it's not available
TOTALSEGMENTATOR_AVAILABLE = False
try:
    from totalsegmentator.python_api import totalsegmentator
    TOTALSEGMENTATOR_AVAILABLE = True
    logging.info("totalsegmentator is available")
except ImportError as e:
    logging.warning(f"totalsegmentator not available: {e}")
    logging.info("Will use threshold-based segmentation as fallback")

# Constants (can be parameterized via CLI if desired)
EROSION_RADIUS = 4       # voxels for spherical erosion
THRESHOLD_FACTOR = 0.01  # fraction of max for forward mask

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def average_acquisition(files):
    """
    Load and average a list of AcquisitionData files.
    Returns averaged AcquisitionData or raises on failure.
    """
    count = 0
    sum_data = None
    for file in files:
        try:
            acq = AcquisitionData(str(file))
            if sum_data is None:
                sum_data = acq.get_uniform_copy(0)
            sum_data += acq
            count += 1
        except Exception as e:
            logging.warning(f"Unable to open {file}: {e}")
    if count == 0:
        raise ValueError("No valid acquisition files to average.")
    sum_data /= count
    return sum_data


def align_segmentation(seg_nii):
    """
    Align NIfTI segmentation to SIRF ImageData orientation.
    Rotates axes and flips as required.
    """
    data = seg_nii.get_fdata() == 1
    # Rotate and flip to match SIRF ordering
    data = np.rot90(data, axes=(0, 2))
    data = np.flip(data, axis=0)
    return data


def threshold_based_segmentation(attenuation, method='adaptive'):
    """
    Create body mask from attenuation map using thresholding.
    More robust than deep learning for attenuation maps.
    """
    attn_arr = attenuation.as_array()

    if method == 'adaptive':
        # Adaptive threshold based on statistics
        mean_val = np.mean(attn_arr)
        std_val = np.std(attn_arr)
        threshold = mean_val + 0.5 * std_val  # Adjust multiplier as needed

    elif method == 'otsu':
        # Otsu's method - automatically find optimal threshold
        try:
            from skimage.filters import threshold_otsu
            threshold = threshold_otsu(attn_arr)
        except ImportError:
            logging.warning("scikit-image not available for Otsu method, using adaptive")
            mean_val = np.mean(attn_arr)
            std_val = np.std(attn_arr)
            threshold = mean_val + 0.5 * std_val

    elif method == 'percentile':
        # Percentile-based threshold
        threshold = np.percentile(attn_arr[attn_arr > 0], 25)  # 25th percentile of non-zero values

    else:  # 'simple'
        # Simple percentage of maximum
        threshold = 0.05 * attn_arr.max()

    # Create binary mask
    mask = attn_arr > threshold

    # Remove small objects (noise) if scikit-image is available
    try:
        from skimage.morphology import remove_small_holes, remove_small_objects
        mask = remove_small_objects(mask, min_size=1000)
        mask = remove_small_holes(mask, area_threshold=1000)
    except ImportError:
        logging.warning("scikit-image not available for morphological operations")

    logging.info(f"Threshold-based segmentation: method={method}, threshold={threshold:.4f}, mask_volume={np.sum(mask)}")

    return mask


def totalsegmentator_segmentation(attenuation):
    """
    Use totalsegmentator for body segmentation.
    Returns mask or raises exception on failure.
    """
    if not TOTALSEGMENTATOR_AVAILABLE:
        raise ImportError("totalsegmentator not available")

    # Convert to NIfTI format for totalsegmentator
    tmp_nii = RegImageData(attenuation)
    tmp_nii.write("__tmp_attn.nii")

    try:
        seg_nii = nib.load("__tmp_attn.nii")
        seg = totalsegmentator(seg_nii, body_seg=True, task='body')
        mask = align_segmentation(seg)

        # Clean up temporary file
        try:
            os.remove("__tmp_attn.nii")
        except OSError:
            pass

        logging.info(f"totalsegmentator segmentation: mask_volume={np.sum(mask)}")
        return mask

    except Exception as e:
        # Clean up temporary file on error
        try:
            os.remove("__tmp_attn.nii")
        except OSError:
            pass
        raise e


def mask_and_forward(model, image, attenuation, erosion_radius, threshold_factor, segment=True, seg_method='adaptive', force_threshold=False):
    """
    Segment body, erode mask, apply to attenuation and forward-project.
    Tries totalsegmentator first, falls back to threshold method.
    Returns forward projection and attenuation-masked forward mask.
    """
    if segment:
        mask = None

        # Try totalsegmentator first (unless forced to use threshold)
        if not force_threshold and TOTALSEGMENTATOR_AVAILABLE:
            try:
                mask = totalsegmentator_segmentation(attenuation)
                logging.info("Successfully used totalsegmentator for body segmentation")
            except Exception as e:
                logging.warning(f"totalsegmentator failed: {e}")
                logging.info("Falling back to threshold-based segmentation")

        # Fall back to threshold method if totalsegmentator failed or not available
        if mask is None:
            mask = threshold_based_segmentation(attenuation, method=seg_method)
            logging.info(f"Using threshold-based segmentation ({seg_method} method)")

    else:
        # Simple threshold when segmentation is disabled
        mask = attenuation.as_array() > threshold_factor * attenuation.max()
        logging.info("Using simple threshold for body mask (segmentation disabled)")

    # Erode mask (spherical)
    selem = ball(erosion_radius)
    eroded = erosion(mask, selem)

    logging.info(f"Mask erosion: original_volume={np.sum(mask)}, eroded_volume={np.sum(eroded)}")

    # Apply eroded mask to attenuation
    attn_arr = attenuation.as_array()
    attn_arr[~eroded] = 0.0
    attenuation.fill(attn_arr)

    # Forward project attenuation
    fwd_attn = model.forward(attenuation)
    thresh = threshold_factor * fwd_attn.max()
    fwd_arr = fwd_attn.as_array() >= thresh
    fwd_attn.fill(fwd_arr)

    return model.forward(image), fwd_attn


def get_spect_data(data_dir):
    """
    Load SPECT data from directory:
    - peak.hs => acquisition_data
    - umap_zoomed.hv => attenuation image
    - initial or template image
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        logging.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    acq_path = data_dir / "peak.hs"
    attn_path = data_dir / "umap_zoomed.hv"
    if not acq_path.exists() or not attn_path.exists():
        logging.error("Expected files missing in data_dir.")
        sys.exit(1)

    spect_data = {
        "acquisition_data": AcquisitionData(str(acq_path)),
        "attenuation": ImageData(str(attn_path)),
    }
    # initial image fallback
    init_path = data_dir / "initial_image.hv"
    tmpl_path = data_dir / "template_image.hv"
    try:
        img = ImageData(str(init_path)).maximum(0)
    except Exception:
        img = ImageData(str(tmpl_path))
        img.fill(1)
    spect_data["initial_image"] = img
    return spect_data


def get_spect_am(spect_data, args, keep_cache=False):
    """
    Build AcquisitionModelUsingMatrix with attenuation and resolution modeling.
    """
    mat = SPECTUBMatrix()
    mat.set_attenuation_image(spect_data["attenuation"])
    mat.set_keep_all_views_in_cache(keep_cache)
    mat.set_resolution_model(
        args.spect_res[0], args.spect_res[1], args.spect_res[2]
    )
    gauss = SeparableGaussianImageFilter()
    gauss.set_fwhms(args.spect_gauss_fwhm)
    spect_am = AcquisitionModelUsingMatrix(mat)
    spect_am.set_image_data_processor(gauss)
    return spect_am


def parse_spect_res(x):
    vals = x.split(',')
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("spect_res must be 3 values: float,float,bool")
    return float(vals[0]), float(vals[1]), vals[2].lower() == 'true'


def main():
    parser = argparse.ArgumentParser(
        description="Compute and normalize mean scatter from SIMIND outputs."
    )
    parser.add_argument("--input_dir",          required=True,
                        help="Dir with scatter/total files.")
    parser.add_argument("--data_dir",           required=True,
                        help="Dir with acquisition & attenuation files.")
    parser.add_argument("--output_file_prefix", required=True,
                        help="Prefix path (no extension) for all outputs.")
    parser.add_argument("--scatter_pattern",
                       default="*_sca_w1.hs",
                        help="glob for scatter files (e.g. '*_iter${i}_*_sca_w1.hs')")
    parser.add_argument("--total_pattern",
                       default="*_tot_w1.hs",
                        help="glob for total   files (e.g. '*_iter${i}_*_tot_w1.hs')")
    parser.add_argument("--image_pattern",
                        default="recon_osem.hv",
                        help="glob for image   files (e.g. 'recon_osem_i*_s*_smoothed_${i}.hv')")
    parser.add_argument("--delete_files", action='store_true')
    parser.add_argument("--normalise", action='store_true',
                        help="Normalize scatter by forward projection of image.")
    parser.add_argument("--no_segment_body", action='store_false',
                        dest='segment_body',
                        help="Disable body segmentation from attenuation.")
    parser.add_argument("--force_threshold", action='store_true',
                        help="Force use of threshold-based segmentation (skip totalsegmentator).")
    parser.add_argument("--segmentation_method",
                        choices=['adaptive', 'otsu', 'percentile', 'simple'],
                        default='adaptive',
                        help="Method for threshold-based body segmentation (fallback).")
    parser.add_argument(
        "--spect_gauss_fwhm",
        type=float,
        nargs=3,
        default=(13.4, 13.4, 13.4),
        help="Gaussian FWHM for smoothing."
    )
    parser.add_argument(
    "--spect_res",
    type=parse_spect_res,
    default=(1.22, 0.03, False),
    help="Tuple of (float, float, bool) for SPECT resolution and use flag (e.g. 0.0923,0.03,True)"
    )
    args = parser.parse_args()

    # Log segmentation method that will be used
    if args.segment_body:
        if args.force_threshold:
            logging.info(f"Forced to use threshold-based segmentation ({args.segmentation_method})")
        elif TOTALSEGMENTATOR_AVAILABLE:
            logging.info(f"Will try totalsegmentator first, fallback to {args.segmentation_method}")
        else:
            logging.info(f"totalsegmentator not available, using {args.segmentation_method}")
    else:
        logging.info("Body segmentation disabled")

    input_dir = Path(args.input_dir)
    # gather files
    scatter_files = list(input_dir.glob(args.scatter_pattern))
    total_files   = list(input_dir.glob(args.total_pattern))
    if not scatter_files or not total_files:
        logging.error("No scatter or total files found.")
        sys.exit(1)

    # Average projections
    sum_scatter = average_acquisition(scatter_files)
    sum_total   = average_acquisition(total_files)
    sum_trues   = sum_total - sum_scatter

    # write unnormalized outputs
    sum_scatter.write(f"{args.output_file_prefix}_scatter_unscaled.hs")
    logging.info("Wrote unnormalized mean scatter.")
    sum_total.write(f"{args.output_file_prefix}_total_unscaled.hs")
    logging.info("Wrote unnormalized mean total.")
    sum_trues.write(f"{args.output_file_prefix}_trues_unscaled.hs")
    logging.info("Wrote unnormalized mean trues.")

    # Setup SPECT model
    spect_data = get_spect_data(args.data_dir)
    spect_am   = get_spect_am(spect_data, args, keep_cache=True)
    spect_am.set_up(spect_data["acquisition_data"], spect_data["initial_image"])

    # Normalize by forward true counts
    if args.normalise:
        image_files = list(input_dir.glob(args.image_pattern))
        if not image_files:
            logging.error("Recon image not found for forward projection.")
            sys.exit(1)
        image = ImageData(str(image_files[0]))

        forward, fwd_mask = mask_and_forward(
                spect_am,
                image,
                spect_data["attenuation"],
                EROSION_RADIUS,
                THRESHOLD_FACTOR,
                segment=args.segment_body,
                seg_method=args.segmentation_method,
                force_threshold=args.force_threshold
            )

        # compute counts for scaling
        true_masked  = sum_trues.clone()
        true_masked *= fwd_mask
        trues_count  = true_masked.sum()

        fwd_masked   = forward.clone()
        fwd_masked   *= fwd_mask
        fwd_count    = fwd_masked.sum()

        scale = fwd_count / trues_count
        logging.info(f"Scatter scaling factor: {scale:.4f}")
        with open(f"{args.output_file_prefix}_scatter_scaling.txt", 'w') as f:
            f.write(str(scale))

        sum_scatter *= scale
        sum_total   *= scale
        sum_trues   *= scale

        forward.write(f"{args.output_file_prefix}_forward.hs")
        logging.info("Wrote forward projection.")
        fwd_mask.write(f"{args.output_file_prefix}_fwd_mask.hs")
        logging.info("Wrote forward mask.")
        true_masked.write(f"{args.output_file_prefix}_trues_masked.hs")
        logging.info("Wrote masked true counts.")
        fwd_masked.write(f"{args.output_file_prefix}_fwd_masked.hs")
        logging.info("Wrote masked forward counts.")

    else:
        # normalise by total counts in measured / total counts in trues
        total_count = sum_total.sum()
        measured_count = spect_data["acquisition_data"].sum()
        if total_count == 0:
            logging.error("Total counts in mean total is zero, cannot normalize.")
            sys.exit(1)
        scale = measured_count / total_count
        logging.info(f"Scatter scaling factor: {scale:.4f}")
        sum_scatter *= scale
        sum_total   *= scale
        sum_trues   *= scale

    # Write outputs
    sum_scatter.write(f"{args.output_file_prefix}_scatter.hs")
    logging.info("Wrote mean scatter.")
    sum_total.write(f"{args.output_file_prefix}_total.hs")
    logging.info("Wrote mean total.")
    sum_trues.write(f"{args.output_file_prefix}_trues.hs")
    logging.info("Wrote mean trues.")

    if args.delete_files:
        for pattern in ("*_sca_w1.a00", "*_air_w1.a00", "*_tot_w1.a00"):
            for f in input_dir.glob(pattern):
                try:
                    f.unlink()
                except Exception as e:
                    logging.warning(f"Could not delete {f}: {e}")

if __name__ == '__main__':
    start = time.time()
    main()
    logging.info(f"Done in {time.time()-start:.1f} seconds.")

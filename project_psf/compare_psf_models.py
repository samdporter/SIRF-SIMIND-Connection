#!/usr/bin/env python3
"""
Compare SPECT PSF modeling approaches using RDP-regularized SVRG reconstruction.

This refactored entrypoint keeps orchestration thin and delegates to modular helpers:
- psf_compare.config: config loading, overrides, logging
- psf_compare.cluster: cluster sweep staging/submission
- psf_compare.modes: mode-specific reconstruction wiring
"""

import argparse
import gc
import logging
import os
import time

from psf_compare.cluster import run_cluster_sweep
from psf_compare.config import (
    apply_overrides,
    configure_logging,
    ensure_output_dir,
    get_data_path,
    get_solver_config,
    load_config,
)
from psf_compare.modes import (
    run_mode_1,
    run_mode_2,
    run_mode_3,
    run_mode_4,
    run_mode_5,
    run_mode_6,
    run_mode_7,
    run_mode_8,
    run_mode_9,
    run_mode_10,
    run_mode_11,
)
from psf_compare.preconditioning import _create_mask_from_attenuation
from psf_compare.spect_utils import get_spect_data
from sirf.STIR import AcquisitionData, MessageRedirector

from sirf_simind_connection.utils import get_array


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare SPECT PSF modeling approaches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Use default config
    python compare_psf_models.py --config config_default.yaml --output_path /path/to/output

    # Use quick test config
    python compare_psf_models.py --config config_quick_test.yaml --output_path /tmp/test

    # Override config values
    python compare_psf_models.py --config config_default.yaml --output_path /path/to/output \
        --override stochastic.num_epochs=10 --override rdp.beta_values=[0.01,0.1]
        """,
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for reconstructions",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Override config values (e.g., --override stochastic.num_epochs=10). "
            "Legacy keys (svrg.*) remain supported."
        ),
    )

    parser.add_argument(
        "--execution",
        choices=["local", "cluster"],
        default="local",
        help="Run locally (default) or stage/submit an SGE cluster sweep.",
    )

    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit jobs immediately in cluster mode (otherwise stage only).",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.override)
    solver_cfg = get_solver_config(config)

    configure_logging(config["output"]["verbose"])

    data_path = get_data_path(config)

    logging.info("=" * 80)
    logging.info("SPECT PSF Model Comparison")
    logging.info("=" * 80)
    logging.info("Config: %s", args.config)
    logging.info("Data: %s", data_path)
    logging.info("Output: %s", args.output_path)
    logging.info("Execution mode: %s", args.execution)
    logging.info("Modes: %s", config["reconstruction"]["modes"])
    logging.info("Beta values: %s", config["rdp"]["beta_values"])
    logging.info(
        "Stochastic solver: %s - %d epochs, %d subsets",
        solver_cfg.get("algorithm", "SVRG").upper(),
        solver_cfg["num_epochs"],
        solver_cfg["num_subsets"],
    )
    logging.info(
        "Energy window: [%s, %s] keV",
        config["simind"]["energy_lower"],
        config["simind"]["energy_upper"],
    )
    logging.info("=" * 80)

    args.script_path = os.path.abspath(__file__)
    output_root = ensure_output_dir(args.output_path)

    if args.execution == "cluster":
        run_cluster_sweep(args, config)
        return

    AcquisitionData.set_storage_scheme("memory")
    msg = MessageRedirector()

    start_time = time.time()
    spect_data = get_spect_data(data_path)
    mask_image = _create_mask_from_attenuation(spect_data.get("attenuation"), 0.05)
    if mask_image is not None:
        masked_initial = spect_data["initial_image"].clone()
        masked_initial.fill(get_array(masked_initial) * get_array(mask_image))
        spect_data["initial_image"] = masked_initial
        logging.info("Masked initial image using attenuation-derived FOV")

    logging.info("Data loading: %.2fs", time.time() - start_time)

    mode_funcs = {
        1: run_mode_1,
        2: run_mode_2,
        3: run_mode_3,
        4: run_mode_4,
        5: run_mode_5,
        6: run_mode_6,
        7: run_mode_7,
        8: run_mode_8,
        9: run_mode_9,
        10: run_mode_10,
        11: run_mode_11,
    }

    total_start = time.time()

    for mode in config["reconstruction"]["modes"]:
        mode_start = time.time()
        try:
            results = mode_funcs[mode](spect_data, config, output_root)
            logging.info("Mode %s (all betas): %.2fs", mode, time.time() - mode_start)

            # Explicitly delete results and force garbage collection
            logging.info("Cleaning up mode %s results from memory", mode)
            del results
            gc.collect()
        except Exception as exc:
            logging.error("Mode %s failed: %s", mode, exc, exc_info=True)

    total_time = time.time() - total_start
    logging.info("=" * 80)
    logging.info("Completed in %.2fs", total_time)
    logging.info("Results: %s", output_root)
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

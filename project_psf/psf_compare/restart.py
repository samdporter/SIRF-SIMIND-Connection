"""
Helpers to resume reconstructions from existing outputs.
"""

from __future__ import annotations

import csv
import glob
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from sirf.STIR import ImageData


@dataclass
class RestartState:
    """Container describing the state recovered from disk."""

    iteration: int
    image: ImageData
    image_path: str
    objective_history: List[Dict[str, float]]
    step_history: List[Dict[str, float]]


def restart_enabled(config: dict) -> bool:
    """Return True if restart support is enabled in the config."""
    restart_cfg = config.get("restart", {})
    return bool(restart_cfg.get("enabled", False))


def load_restart_state(
    config: dict,
    output_dir: str,
    image_prefix: str,
) -> Optional[RestartState]:
    """
    Load the latest checkpoint image and tracking CSVs for resuming.
    """
    if not restart_enabled(config):
        return None

    image_info = _find_latest_image(output_dir, image_prefix)
    if image_info is None:
        return None

    try:
        restart_image = ImageData(image_info["path"])
    except Exception as exc:
        logging.warning("Failed to load restart image %s: %s", image_info["path"], exc)
        return None

    logging.info(
        "Restart checkpoint: %s (iteration=%d)",
        image_info["path"],
        image_info["iteration"],
    )

    objective_path = os.path.join(output_dir, "objective.csv")
    step_size_path = os.path.join(output_dir, "step_sizes.csv")

    objective_rows = _load_csv_history(
        objective_path,
        expected_fields=("iteration", "objective"),
        converters={"iteration": int, "objective": float},
    )
    step_rows = _load_csv_history(
        step_size_path,
        expected_fields=(
            "iteration",
            "step_size",
            "cache_version",
            "armijo_ran",
            "triggered_armijo_next",
        ),
        converters={
            "iteration": int,
            "step_size": float,
            "cache_version": int,
            "armijo_ran": _to_bool,
            "triggered_armijo_next": _to_bool,
        },
    )

    return RestartState(
        iteration=image_info["iteration"],
        image=restart_image,
        image_path=image_info["path"],
        objective_history=objective_rows,
        step_history=step_rows,
    )


def _find_latest_image(output_dir: str, image_prefix: str) -> Optional[dict]:
    """
    Return the newest saved image_{iteration}.hv in output_dir.
    """
    if not image_prefix:
        image_prefix = "image"

    prefix_path = os.path.join(output_dir, image_prefix)
    pattern = f"{prefix_path}_*.hv"
    regex = re.compile(rf"^{re.escape(prefix_path)}_(\d+)\.hv$", re.IGNORECASE)

    latest = None
    for candidate in glob.glob(pattern):
        match = regex.match(candidate)
        if not match:
            continue

        try:
            iteration = int(match.group(1))
        except ValueError:
            continue

        if iteration <= 0:
            # Ignore the initial image_0 snapshot when searching for progress.
            continue

        if latest is None or iteration > latest["iteration"]:
            latest = {"path": candidate, "iteration": iteration}

    if latest is None:
        logging.info(
            "Restart requested but no checkpoint images found in %s", output_dir
        )

    return latest


def _load_csv_history(
    csv_path: str,
    expected_fields: Iterable[str],
    converters: Optional[Dict[str, Callable[[str], Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Load CSV rows while enforcing field presence and datatype conversions.
    """
    if not os.path.exists(csv_path):
        return []

    rows: List[Dict[str, float]] = []
    converters = converters or {}
    try:
        with open(csv_path, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            missing = set(expected_fields) - set(reader.fieldnames or [])
            if missing:
                logging.warning(
                    "Skipping restart preload for %s (missing columns: %s)",
                    csv_path,
                    ", ".join(sorted(missing)),
                )
                return []

            for row in reader:
                parsed = {}
                for field in expected_fields:
                    value = row.get(field)
                    if value is None:
                        break
                    if field in converters:
                        try:
                            parsed[field] = converters[field](value)
                        except (TypeError, ValueError):
                            parsed[field] = value
                    else:
                        parsed[field] = value
                else:
                    rows.append(parsed)  # only append if all fields present
    except OSError as exc:
        logging.warning("Failed to read %s: %s", csv_path, exc)

    return rows


def _to_bool(value: str) -> bool:
    """Convert CSV boolean strings to bools."""
    return str(value).strip().lower() in {"1", "true", "yes"}

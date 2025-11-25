"""
Configuration helpers for PSF model comparison.
"""

import logging
import os
from typing import Any, Dict, Iterable

import yaml


def configure_logging(verbose: bool = False) -> None:
    """Configure module-wide logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration from disk."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """
    Apply command-line style overrides to a nested config dictionary.

    Example: ["stochastic.num_epochs=10", "rdp.beta_values=[0.01,0.1]"].
    """
    for override in overrides:
        if "=" not in override:
            logging.warning("Invalid override format: %s", override)
            continue

        key_path, value = override.split("=", 1)
        keys = key_path.split(".")

        target = config
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]

        try:
            import ast

            parsed_value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed_value = value

        target[keys[-1]] = parsed_value
        logging.info("Override: %s = %s", key_path, parsed_value)

    return config


def get_solver_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return stochastic solver configuration, supporting legacy 'svrg' blocks.
    """
    solver_cfg = config.get("stochastic")
    if solver_cfg is None:
        solver_cfg = config.get("svrg")
    if solver_cfg is None:
        raise KeyError(
            "Missing 'stochastic' configuration. "
            "Provide config['stochastic'] or legacy config['svrg']."
        )
    return solver_cfg


def ensure_output_dir(path: str) -> str:
    """Create output directory if needed and return absolute path."""
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path

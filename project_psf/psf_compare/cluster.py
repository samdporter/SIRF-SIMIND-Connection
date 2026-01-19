"""
Cluster sweep helpers for PSF comparison.
"""

import logging
import os
import shlex
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Sequence, Tuple


def _format_literal(value) -> str:
    """Return a string representation safe for literal evaluation."""
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    return repr(value)


def _get_data_paths(config: Dict) -> List[str]:
    """Extract data paths from config, supporting both single and multiple paths."""
    if "data_paths" in config and config["data_paths"]:
        paths = config["data_paths"]
        if isinstance(paths, str):
            return [paths]
        return list(paths)
    if "data_path" in config and config["data_path"]:
        return [config["data_path"]]
    return []


def _get_data_source_name(data_path: str) -> str:
    """Extract a short identifier from a data path for output organization."""
    import os

    path = data_path.rstrip("/")
    # Use parent directory name if last component is generic (e.g., "SPECT")
    basename = os.path.basename(path)
    if basename.upper() in ("SPECT", "DATA", "OUTPUT"):
        parent = os.path.basename(os.path.dirname(path))
        return parent if parent else basename
    return basename


def _generate_sweep_combinations(
    config: Dict,
) -> List[Tuple[str, str, int, str]]:
    """Return list of (data_path, source_name, mode, beta_literal) for cluster runs.

    Sweeps over: data_paths × modes × beta_values
    """
    data_paths = _get_data_paths(config)
    modes = sorted(int(mode) for mode in config["reconstruction"]["modes"])
    betas = config["rdp"]["beta_values"]

    combinations = []
    for data_path in data_paths:
        source_name = _get_data_source_name(data_path)
        for mode in modes:
            for beta in betas:
                combinations.append(
                    (data_path, source_name, mode, _format_literal(beta))
                )
    return combinations


def run_cluster_sweep(args, config: Dict[str, Any]) -> None:
    """Launch or stage an SGE array sweep over (data_source, mode, beta) combinations."""

    combinations = _generate_sweep_combinations(config)
    if not combinations:
        logging.error(
            "No (data_source, mode, beta) combinations available for cluster run"
        )
        return

    # Log data sources being swept
    data_paths = _get_data_paths(config)
    logging.info("Data sources (%d):", len(data_paths))
    for dp in data_paths:
        logging.info("  - %s", dp)

    cluster_cfg = config.get("cluster", {})
    job_name = cluster_cfg.get("job_name", "psf_mode_beta")
    runtime = cluster_cfg.get("runtime", "48:00:00")
    memory = cluster_cfg.get("memory", "60G")
    queue = cluster_cfg.get("queue")
    cores = int(cluster_cfg.get("cores", 1))
    gpu = bool(cluster_cfg.get("gpu", False))
    submit_jobs = bool(args.submit or cluster_cfg.get("submit", False))
    python_exec = cluster_cfg.get("python_executable", sys.executable)
    env_setup = cluster_cfg.get("env_setup_script")
    logs_subdir = cluster_cfg.get("logs_subdir", "_logs")
    qsub_extra: Sequence[str] = cluster_cfg.get("qsub_extra_args", [])

    if isinstance(qsub_extra, str):
        qsub_extra = shlex.split(qsub_extra)
    elif not isinstance(qsub_extra, (list, tuple)):
        qsub_extra = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    job_script_default = os.path.join(script_dir, "..", "cluster", "psf_sweep_job.sh")
    job_script_default = os.path.abspath(job_script_default)
    job_script = os.path.abspath(cluster_cfg.get("job_script", job_script_default))

    if not os.path.isfile(job_script):
        raise FileNotFoundError(
            f"Cluster job script not found: {job_script}. Check config['cluster']['job_script']."
        )

    config_path = os.path.abspath(args.config)
    output_root = os.path.abspath(args.output_path)
    os.makedirs(output_root, exist_ok=True)

    cluster_root = os.path.join(output_root, "_cluster")
    os.makedirs(cluster_root, exist_ok=True)

    task_file = os.path.join(cluster_root, f"{job_name}_tasks.csv")
    with open(task_file, "w", encoding="utf-8") as f:
        for data_path, source_name, mode, beta_literal in combinations:
            # CSV format: data_path,source_name,mode,beta
            f.write(f"{data_path},{source_name},{mode},{beta_literal}\n")

    overrides_file = None
    if args.override:
        overrides_file = os.path.join(cluster_root, f"{job_name}_overrides.txt")
        with open(overrides_file, "w", encoding="utf-8") as f:
            for override in args.override:
                f.write(f"{override}\n")

    if env_setup:
        env_setup = os.path.abspath(env_setup)

    logs_dir = os.path.join(cluster_root, logs_subdir)
    os.makedirs(logs_dir, exist_ok=True)

    env_exports = {
        "PSF_SCRIPT": os.path.abspath(args.script_path),
        "CONFIG_PATH": config_path,
        "OUTPUT_ROOT": output_root,
        "TASK_FILE": task_file,
        "PYTHON_EXECUTABLE": python_exec,
    }

    if overrides_file:
        env_exports["OVERRIDES_FILE"] = overrides_file
    if env_setup:
        env_exports["ENV_SETUP_SCRIPT"] = env_setup
    env_exports["CLUSTER_CORES"] = str(cores)

    def _escape_env_value(value: str) -> str:
        text = str(value)
        text = text.replace("\\", "\\\\")
        text = text.replace(",", "\\,")
        text = text.replace(" ", "\\ ")
        return text

    env_export_str = ",".join(
        f"{key}={_escape_env_value(value)}" for key, value in env_exports.items()
    )

    total_tasks = len(combinations)
    task_range = f"1-{total_tasks}"

    qsub_cmd: List[str] = [
        "qsub",
        "-t",
        task_range,
        "-N",
        job_name,
        "-l",
        f"h_rt={runtime}",
    ]

    if gpu:
        qsub_cmd.extend(["-l", "gpu=true", "-l", f"tmem={memory}"])
    else:
        qsub_cmd.extend(["-l", f"h_vmem={memory}"])

    if queue:
        qsub_cmd.extend(["-q", queue])

    if cores > 1:
        qsub_cmd.extend(["-pe", "smp", str(cores)])

    qsub_cmd.extend(["-o", logs_dir, "-e", logs_dir])
    qsub_cmd.extend(qsub_extra)
    qsub_cmd.extend(["-v", env_export_str, job_script])

    logging.info("Prepared %d SGE tasks (data_sources × modes × betas).", total_tasks)
    logging.info("Task file: %s", task_file)
    logging.info("Logs directory: %s", logs_dir)
    logging.info("Job script: %s", job_script)
    logging.info("Sample command: %s", " ".join(shlex.quote(arg) for arg in qsub_cmd))

    if submit_jobs:
        if shutil.which("qsub") is None:
            logging.error("qsub command not found in PATH; cannot submit jobs.")
            return

        logging.info("Submitting SGE array job...")
        try:
            subprocess.run(qsub_cmd, check=True)
        except subprocess.CalledProcessError as exc:
            logging.error("qsub submission failed: %s", exc)
            return

        logging.info("SGE submission successful. Monitor with qstat -u $USER")
    else:
        logging.info(
            "Cluster run staged only. Re-run with --submit or set cluster.submit=true to submit."
        )

"""
Callbacks used during PSF model comparison runs.
"""

import csv
import logging
import os
from typing import List

from sirf_simind_connection.utils import get_array


class SaveObjectiveCallback:
    """Log objective values with iteration numbers to a CSV file."""

    def __init__(self, csv_path, interval=1, start_iteration=0, log_on_armijo=False):
        self.csv_path = csv_path
        self._interval = max(int(interval), 1)
        self.start_iteration = int(start_iteration)
        self.log_on_armijo = bool(log_on_armijo)
        self.records: List[dict] = []

    def __call__(self, algorithm):
        # CIL reports iteration=-1 before the first update; skip in that case.
        if algorithm.iteration < 0:
            return

        armijo_ran = False
        if hasattr(algorithm, "step_size_rule") and hasattr(
            algorithm.step_size_rule, "armijo_ran_this_iteration"
        ):
            armijo_ran = algorithm.step_size_rule.armijo_ran_this_iteration

        if algorithm.iteration < self.start_iteration and not (
            self.log_on_armijo and armijo_ran
        ):
            return

        interval_hit = (
            algorithm.iteration - self.start_iteration
        ) % self._interval == 0
        if not interval_hit and not (self.log_on_armijo and armijo_ran):
            return

        objective_value = algorithm.f(algorithm.solution) + algorithm.g(
            algorithm.solution
        )
        self.records.append(
            {
                "iteration": algorithm.iteration,
                "objective": float(objective_value),
            }
        )
        self._flush()

    def set_interval(self, interval):
        """Update the logging interval (must be >= 1)."""
        self._interval = max(int(interval), 1)

    def set_start_iteration(self, start_iteration):
        """Delay logging until the given iteration (local to the current run)."""
        self.start_iteration = int(start_iteration)

    def load_history(self, records, interval=None):
        """
        Seed the callback with existing records so restarted runs append cleanly.
        """
        if not records:
            return
        self.records = list(records)
        try:
            last_iteration = max(int(r.get("iteration", 0)) for r in records)
        except (TypeError, ValueError):
            last_iteration = 0
        interval = interval or self._interval
        self.set_start_iteration(last_iteration + interval)

    def _flush(self):
        """Persist the collected objective history to CSV."""
        output_dir = os.path.dirname(self.csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            with open(self.csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["iteration", "objective"])
                writer.writeheader()
                writer.writerows(self.records)
        except OSError as exc:
            logging.error(
                "Failed to write objective history to %s: %s", self.csv_path, exc
            )


class SaveImageCallback:
    """Save intermediate images at a fixed interval."""

    def __init__(self, output_prefix, interval=1):
        self.output_prefix = output_prefix
        self.interval = max(int(interval), 1)
        output_dir = os.path.dirname(output_prefix)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def __call__(self, algorithm):
        iteration = algorithm.iteration
        if iteration < 0 or (iteration % self.interval) != 0:
            return

        output_path = f"{self.output_prefix}_{iteration}"

        try:
            algorithm.solution.write(output_path)
            logging.debug("Saved image at iteration %d to %s", iteration, output_path)
        except Exception as exc:  # pragma: no cover - diagnostic only
            logging.warning(
                "Failed to save image at iteration %d to %s: %s",
                iteration,
                output_path,
                exc,
            )

    def set_interval(self, interval):
        """Update the saving interval (must be >= 1)."""
        self.interval = max(int(interval), 1)


class UpdateEtaCallback:
    """
    Callback to update additive/residual terms in KL functions after SIMIND simulations.

    This callback refreshes the additive component (`eta` for classical KL, `additive`
    for the residual-aware variant) and, when available, the residual correction.
    Updates are triggered whenever the coordinator publishes a new cache version.
    """

    def __init__(
        self,
        coordinator,
        kl_data_functions,
        partition_indices,
        eta_floor=1e-8,
        debug_objective_path=None,
        image_smoothing_fwhm=None,
        restart_on_update=False,
        restart_image=None,
    ):
        self.coordinator = coordinator
        self.kl_data_functions = kl_data_functions
        self.partition_indices = partition_indices
        self.eta_floor = float(max(eta_floor, 0.0))
        self.last_cache_version = 0
        self.debug_objective_path = debug_objective_path
        self.image_smoothing_fwhm = image_smoothing_fwhm
        self.restart_on_update = bool(restart_on_update)
        self.restart_image = (
            restart_image.clone() if restart_image is not None else None
        )

    def __call__(self, algorithm):
        """Update eta if new SIMIND simulation results are available."""
        # Check if coordinator has new simulation results
        if self.coordinator.cache_version > self.last_cache_version:
            # Get full additive term from coordinator
            full_additive = self.coordinator.get_full_additive_term()

            if full_additive is not None:
                additive_for_eta = full_additive

                logging.info(
                    "UpdateEtaCallback: Updating KL at iteration %d (cache %d -> %d)",
                    algorithm.iteration,
                    self.last_cache_version,
                    self.coordinator.cache_version,
                )

                # Update eta for each subset
                for i, (kl_func, subset_indices) in enumerate(
                    zip(self.kl_data_functions, self.partition_indices)
                ):
                    additive_subset = additive_for_eta.get_subset(subset_indices)

                    if self.eta_floor > 0.0:
                        additive_subset = additive_subset.maximum(self.eta_floor)

                    if hasattr(kl_func, "eta"):
                        kl_func.eta = additive_subset
                        # CIL KL (numba backend) caches eta as numpy array once;
                        # refresh cache so objective/prox evaluations see updates.
                        if hasattr(kl_func, "eta_np"):
                            kl_func.eta_np = get_array(additive_subset)
                    else:
                        raise TypeError(
                            "UpdateEtaCallback requires KL functions with an 'eta' attribute"
                        )

                    logging.info(
                        "  Subset %d additive sum: %.2e, min: %.2e",
                        i,
                        additive_subset.sum(),
                        float(get_array(additive_subset).min()),
                    )

                # Update cache version tracker
                self.last_cache_version = self.coordinator.cache_version

                # Ensure Armijo step size update runs after the residual has been
                # applied to the additive term.
                self._trigger_armijo_after_eta_update(algorithm)

                if self.restart_on_update and self.restart_image is not None:
                    self._restart_algorithm_state(algorithm)

    def _trigger_armijo_after_eta_update(self, algorithm):
        """
        Schedule an Armijo line search so it runs with the refreshed gradient.
        """
        step_rule = getattr(algorithm, "step_size_rule", None)
        if hasattr(step_rule, "force_armijo_after_correction"):
            step_rule.force_armijo_after_correction(algorithm)
        elif hasattr(step_rule, "trigger_armijo"):
            step_rule.trigger_armijo = True

    def _restart_algorithm_state(self, algorithm):
        """Reset algorithm iterates to the stored restart image."""
        restart_img = self.restart_image
        if restart_img is None:
            return

        if hasattr(algorithm, "x_old") and algorithm.x_old is not None:
            algorithm.x_old.fill(restart_img)
        if hasattr(algorithm, "x") and algorithm.x is not None:
            algorithm.x.fill(restart_img)

        logging.info(
            "Restarted algorithm state from initial image after correction update"
        )

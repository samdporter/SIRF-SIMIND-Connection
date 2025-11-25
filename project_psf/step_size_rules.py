"""
Custom step size rules for SVRG reconstruction with coordinator-based corrections.

This module provides step size rules that adapt based on Monte Carlo correction updates
from SimindCoordinator or StirPsfCoordinator.
"""

import logging

import pandas as pd
from cil.optimisation.utilities import StepSizeRule


class LinearDecayStepSizeRule(StepSizeRule):
    """Linear decay step size: step_size / (1 + eta * k)"""

    def __init__(self, initial_step_size, decay_rate, start_iteration=0):
        super().__init__()
        self.initial_step_size = initial_step_size
        self.decay_rate = decay_rate
        self.start_iteration = start_iteration

    def get_step_size(self, algorithm):
        # Use relative iterations since the rule was (re)started
        k = max(algorithm.iteration - self.start_iteration, 0)
        return self.initial_step_size / (1 + self.decay_rate * k)


class ArmijoAfterCorrectionStepSize(StepSizeRule):
    """
    Armijo step size search triggered only after correction updates.

    This step size rule performs an Armijo line search:
    - When explicitly triggered after coordinator updates via trigger_armijo
    - When force_armijo_after_correction is called directly

    Between Armijo searches, it uses linear decay.

    The Armijo rule searches for a step size satisfying:
        f(x_new) + g(x_new) <= f(x) - tol * step_size * ||gradient||^2

    where x_new is the proximal step:
        x_new = prox_g(x - step_size * preconditioned_gradient)

    Workflow:
    1. ArmijoTriggerCallback detects coordinator updates and sets trigger_armijo
    2. When trigger flag is set, perform Armijo search starting from current_step_size
    3. Between searches, use linear decay from last Armijo step size

    Args:
        initial_step_size (float): Starting step size for Armijo search
        beta (float): Backtracking factor (0 < beta < 1), typically 0.5
        decay_rate (float): Linear decay rate for eta in denominator (1 + eta * k)
        max_iter (int): Maximum Armijo backtracking iterations
        tol (float): Armijo sufficient decrease tolerance

    Examples:
        >>> coordinator = SimindCoordinator(...)
        >>> step_size_rule = ArmijoAfterCorrectionStepSize(
        ...     initial_step_size=1.0,
        ...     beta=0.5,
        ...     decay_rate=0.01,
        ...     max_iter=20,
        ...     tol=1e-4
        ... )
        >>> armijo_callback = ArmijoTriggerCallback(step_size_rule, coordinator)
        >>> algo = ISTA(initial=x0, f=f, g=g, step_size=step_size_rule)
        >>> coordinator.algorithm = algo
        >>> algo.run(num_iterations, callbacks=[armijo_callback, ...])
    """

    def __init__(
        self,
        initial_step_size: float,
        beta: float,
        decay_rate: float,
        max_iter: int,
        tol: float,
    ):
        super().__init__()
        self.initial_step_size = initial_step_size
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.decay_rate = decay_rate

        # State tracking
        self.current_step_size = initial_step_size
        self.last_armijo_iteration = 0
        self.trigger_armijo = False  # Set by ArmijoTriggerCallback
        self.armijo_ran_this_iteration = False  # Flag for callback to read
        self.min_step_size_seen = initial_step_size
        self.cached_step_iteration = None
        self.cached_step_value = None

        # Linear decay rule for between-update iterations
        self.linear_decay = LinearDecayStepSizeRule(
            initial_step_size, decay_rate, start_iteration=0
        )

    def _armijo_search(self, algorithm, trigger_reason):
        """Run Armijo search and cache the step for the current iteration."""
        self.armijo_ran_this_iteration = True
        self.trigger_armijo = False

        logging.info(
            f"Armijo: Running line search at iteration {algorithm.iteration} "
            f"(reason: {trigger_reason})"
        )

        f_x = algorithm.f(algorithm.solution) + algorithm.g(algorithm.solution)

        if algorithm.preconditioner is not None:
            gradient = algorithm.gradient_update.copy()
            precond_grad = algorithm.preconditioner.apply(algorithm, gradient)
        else:
            precond_grad = algorithm.gradient_update

        g_norm = algorithm.gradient_update.dot(precond_grad)

        step_size = self.initial_step_size
        logging.info(f"Starting Armijo line search with initial step size: {step_size}")

        for armijo_iter in range(self.max_iter):
            x_new = algorithm.solution.copy().sapyb(1, precond_grad, -step_size)
            algorithm.g.proximal(x_new, step_size, out=x_new)
            f_x_new = algorithm.f(x_new) + algorithm.g(x_new)

            if f_x_new <= f_x - self.tol * step_size * g_norm:
                logging.info(
                    f"  Armijo accepted: step_size={step_size:.6f} "
                    f"(iter {armijo_iter}, objective={f_x_new:.6e})"
                )
                break

            step_size *= self.beta
        else:
            logging.warning(
                f"  Armijo max iterations ({self.max_iter}) reached, "
                f"using step_size={step_size:.6f}"
            )

        self.current_step_size = step_size
        self.min_step_size_seen = min(self.min_step_size_seen, step_size)
        self.last_armijo_iteration = algorithm.iteration
        self.linear_decay = LinearDecayStepSizeRule(
            step_size, self.decay_rate, start_iteration=self.last_armijo_iteration
        )

        self.cached_step_iteration = algorithm.iteration
        self.cached_step_value = step_size

        return step_size

    def force_armijo_after_correction(self, algorithm):
        """
        Immediately run Armijo after a correction update and cache the result
        for reuse in the same iteration.
        """
        return self._armijo_search(algorithm, trigger_reason="correction (immediate)")

    def get_step_size(self, algorithm):
        """
        Calculate and return the step size.

        Performs Armijo search only when trigger_armijo flag is set. Otherwise,
        uses linear decay.

        Args:
            algorithm: CIL algorithm instance (ISTA, FISTA, etc.)

        Returns:
            float: Step size for this iteration
        """
        # Reuse cached step if Armijo already ran this iteration
        if self.cached_step_iteration == algorithm.iteration:
            self.armijo_ran_this_iteration = True
            return self.cached_step_value

        # Reset flag from previous iteration
        self.armijo_ran_this_iteration = False

        if self.trigger_armijo:
            return self._armijo_search(algorithm, trigger_reason="correction update")

        # Between corrections: use linear decay
        step_size = self.linear_decay.get_step_size(algorithm)
        self.current_step_size = step_size  # Update current_step_size for callbacks
        logging.debug(
            f"Using linear decay step size at iteration {algorithm.iteration}: "
            f"{step_size:.6f}"
        )
        return step_size

    def reset(self):
        """Reset the step size rule state (for multi-stage reconstructions)."""
        self.current_step_size = self.initial_step_size
        self.last_armijo_iteration = 0
        self.trigger_armijo = False
        self.armijo_ran_this_iteration = False
        self.min_step_size_seen = self.initial_step_size
        self.cached_step_iteration = None
        self.cached_step_value = None
        self.linear_decay = LinearDecayStepSizeRule(
            self.initial_step_size, self.decay_rate, start_iteration=0
        )
        logging.info("ArmijoAfterCorrectionStepSize reset")


class ArmijoTriggerCallback:
    """
    Callback to trigger Armijo line search after coordinator updates.

    This callback monitors a coordinator's cache_version and sets a flag
    in the algorithm's step size rule to trigger Armijo search after corrections.
    It also logs all step sizes to a CSV file for tracking.

    The callback should be placed AFTER UpdateEtaCallback in the callback list
    to ensure eta updates happen before step size recomputation.

    Args:
        coordinator: SimindCoordinator or StirPsfCoordinator instance.
        csv_path (str, optional): Path to save step size history.
            If None, no CSV is saved.

    Examples:
        >>> step_size_rule = ArmijoAfterCorrectionStepSize(...)
        >>> armijo_callback = ArmijoTriggerCallback(
        ...     coordinator, csv_path="step_sizes.csv"
        ... )
        >>> eta_callback = UpdateEtaCallback(
        ...     coordinator, kl_funcs, partition_indices
        ... )
        >>> algo = ISTA(initial=x0, f=f, g=g, step_size=step_size_rule)
        >>> algo.run(iterations, callbacks=[eta_callback, armijo_callback, ...])
    """

    def __init__(self, coordinator, csv_path=None):
        self.coordinator = coordinator
        self.last_cache_version = coordinator.cache_version  # Sync with current version
        self.csv_path = csv_path
        self.step_size_history = []  # Store history in memory
        self.iteration_offset = 0

    def __call__(self, algorithm):
        """Check if coordinator updated, trigger Armijo if so, and log step size."""
        global_iteration = self.iteration_offset + algorithm.iteration

        # Get current step size - prefer reading from step_size_rule.current_step_size
        if hasattr(algorithm, "step_size_rule") and hasattr(
            algorithm.step_size_rule, "current_step_size"
        ):
            current_step_size = algorithm.step_size_rule.current_step_size
        elif hasattr(algorithm, "step_size"):
            current_step_size = algorithm.step_size
        else:
            current_step_size = None

        # Check if Armijo ran this iteration (check step_size_rule flag)
        armijo_ran = False
        if hasattr(algorithm, "step_size_rule") and hasattr(
            algorithm.step_size_rule, "armijo_ran_this_iteration"
        ):
            armijo_ran = algorithm.step_size_rule.armijo_ran_this_iteration
        elif hasattr(algorithm, "_step_size") and hasattr(
            algorithm._step_size, "armijo_ran_this_iteration"
        ):
            armijo_ran = algorithm._step_size.armijo_ran_this_iteration

        # Log step size to history
        self.step_size_history.append(
            {
                "iteration": global_iteration,
                "step_size": current_step_size,
                "cache_version": self.coordinator.cache_version,
                "armijo_ran": armijo_ran,
                "triggered_armijo_next": False,  # Will update if triggered for next iter
            }
        )

        # Check if coordinator has new corrections
        if self.coordinator.cache_version > self.last_cache_version:
            logging.info(
                f"ArmijoTriggerCallback: Setting trigger_armijo flag at iteration "
                f"{algorithm.iteration} "
                f"(cache_version: {self.last_cache_version} -> "
                f"{self.coordinator.cache_version}). "
                f"Armijo will run at iteration {algorithm.iteration + 1}"
            )
            # Access step size rule through algorithm
            if hasattr(algorithm, "step_size_rule") and hasattr(
                algorithm.step_size_rule, "trigger_armijo"
            ):
                algorithm.step_size_rule.trigger_armijo = True
                self.step_size_history[-1]["triggered_armijo_next"] = True
            elif hasattr(algorithm, "step_size") and hasattr(
                algorithm.step_size, "trigger_armijo"
            ):
                algorithm.step_size.trigger_armijo = True
                self.step_size_history[-1]["triggered_armijo_next"] = True
            else:
                logging.warning(
                    "Could not find step_size_rule with trigger_armijo attribute"
                )
            self.last_cache_version = self.coordinator.cache_version

        # Save to CSV if path provided
        if self.csv_path is not None:
            df = pd.DataFrame(self.step_size_history)
            df.to_csv(self.csv_path, index=False)
            logging.debug(f"Saved step size history to {self.csv_path}")

    def increment_iteration_offset(self, increment):
        """Advance the iteration offset for sequential algorithm runs."""
        self.iteration_offset += int(increment)


class SaveStepSizeHistoryCallback:
    """
    Callback to log step size history, specifically for ArmijoAfterCorrectionStepSize.

    Logs the step size at each iteration to a CSV file, along with metadata about
    whether the Armijo search was run and the coordinator's cache version.

    Args:
        csv_path (str): Path to save the step size history CSV file.
        coordinator (Coordinator, optional): The coordinator instance, if one is used.
    """

    def __init__(self, csv_path, coordinator=None):
        self.csv_path = csv_path
        self.coordinator = coordinator
        self.step_size_history = []
        self.iteration_offset = 0

    def __call__(self, algorithm):
        """Log step size and related metadata for the current iteration."""
        # Get current step size - prefer reading from step_size_rule.current_step_size
        # (which is now always kept up-to-date), fallback to algorithm.step_size
        if hasattr(algorithm, "step_size_rule") and hasattr(
            algorithm.step_size_rule, "current_step_size"
        ):
            current_step_size = algorithm.step_size_rule.current_step_size
        else:
            current_step_size = algorithm.step_size

        # Check if Armijo ran this iteration
        armijo_ran = False
        if hasattr(algorithm, "step_size_rule") and hasattr(
            algorithm.step_size_rule, "armijo_ran_this_iteration"
        ):
            armijo_ran = algorithm.step_size_rule.armijo_ran_this_iteration

        # Get coordinator cache version, if available
        cache_version = self.coordinator.cache_version if self.coordinator else -1

        # Check if Armijo will be triggered for the *next* iteration
        triggered_next = False
        if hasattr(algorithm, "step_size_rule") and hasattr(
            algorithm.step_size_rule, "trigger_armijo"
        ):
            triggered_next = algorithm.step_size_rule.trigger_armijo

        # Log data for this iteration
        self.step_size_history.append(
            {
                "iteration": self.iteration_offset + algorithm.iteration,
                "step_size": current_step_size,
                "cache_version": cache_version,
                "armijo_ran": armijo_ran,
                "triggered_armijo_next": triggered_next,
            }
        )

        # Save to CSV
        try:
            df = pd.DataFrame(self.step_size_history)
            df.to_csv(self.csv_path, index=False)
        except IOError as e:
            logging.error(f"Could not write to step size history file: {e}")

    def increment_iteration_offset(self, increment):
        """Advance the iteration offset when chaining multiple algorithm runs."""
        self.iteration_offset += int(increment)


class ConstantStepSize(StepSizeRule):
    """Constant step size rule (for testing/comparison)."""

    def __init__(self, step_size):
        super().__init__()
        self.step_size = step_size

    def get_step_size(self, algorithm):
        return self.step_size

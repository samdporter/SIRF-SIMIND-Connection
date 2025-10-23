"""
Coordinator Module

This module defines the Coordinator base class and its implementations:
- Coordinator: Abstract base class for coordinating accurate vs fast projections
- SimindCoordinator: Manages SIMIND Monte Carlo simulations
- StirPsfCoordinator: Manages STIR PSF-based corrections

Coordinators enable efficient iterative reconstruction by coordinating "accurate but slow"
and "fast" projection operations, computing corrections to improve reconstruction quality.
"""

import logging
from abc import ABC, abstractmethod

from .types import ScoringRoutine


# Conditional import for SIRF to avoid CI dependencies
try:
    from sirf.STIR import AcquisitionData, ImageData

    SIRF_AVAILABLE = True
except ImportError:
    AcquisitionData = type(None)
    ImageData = type(None)
    SIRF_AVAILABLE = False


class Coordinator(ABC):
    """
    Abstract base class for coordinating accurate vs fast projections.

    Coordinators manage "accurate but slow" and "fast" projection operations,
    computing corrections to improve reconstruction quality. They support both
    single-projector and subset-based reconstruction workflows.

    Subclasses must implement:
    - should_update(): Check if correction should run this iteration
    - run_accurate_projection(): Run accurate projection and update corrections
    - get_full_additive_term(): Get full cumulative additive term
    - get_subset_residual(): Get residual correction for a specific subset
    - reset_iteration_counter(): Reset iteration tracking
    """

    @abstractmethod
    def should_update(self):
        """
        Check if correction should run this iteration.

        Returns:
            bool: True if correction should be computed.
        """
        pass

    @abstractmethod
    def run_accurate_projection(self, image):
        """
        Run accurate projection and update corrections.

        This is the main coordination method that triggers the "accurate but slow"
        projection operation and computes corrections based on the coordinator's
        correction mode.

        Args:
            image (ImageData): Current image estimate.
        """
        pass

    @abstractmethod
    def get_full_additive_term(self):
        """
        Get full cumulative additive term (all views).

        Returns:
            AcquisitionData: Full cumulative additive term, or None if not yet init.
        """
        pass

    @abstractmethod
    def get_subset_residual(self, subset_indices, current_additive_subset=None):
        """
        Get residual correction for a specific subset.

        Args:
            subset_indices (list): View indices for this subset.
            current_additive_subset (AcquisitionData, optional): Current additive
                term estimate for this subset.

        Returns:
            AcquisitionData or array-like: Residual correction for this subset.
        """
        pass

    @abstractmethod
    def reset_iteration_counter(self):
        """
        Reset iteration tracking.

        Useful for multi-stage reconstructions where you want to restart
        the correction update schedule.
        """
        pass

    @property
    @abstractmethod
    def cache_version(self):
        """
        Version number incremented after each update.

        Used by callbacks (e.g., UpdateEtaCallback) to detect new corrections.

        Returns:
            int: Current cache version.
        """
        pass

    @property
    @abstractmethod
    def algorithm(self):
        """
        Reference to CIL/reconstruction algorithm for iteration tracking.

        Returns:
            Algorithm: CIL algorithm instance with iteration counter.
        """
        pass

    @algorithm.setter
    @abstractmethod
    def algorithm(self, value):
        """Set the algorithm reference."""
        pass

    @property
    def last_update_iteration(self):
        """
        Last iteration when correction was updated.

        Returns:
            int: Iteration number of last update.

        Note: Concrete implementations may override this or use a simple attribute.
        """
        return getattr(self, "_last_update_iteration", -1)

    @property
    def linear_acquisition_model(self):
        """
        Full-data linear acquisition model (no additive term).

        This model is used by the partitioner to create subset projectors
        and for computing residuals between accurate and fast projections.

        Concrete implementations should set this in __init__ or override this property.

        Returns:
            AcquisitionModel: Linear acquisition model for full data.
        """
        return getattr(self, "_linear_acquisition_model", None)


class SimindCoordinator(Coordinator):
    """
    SimindCoordinator Class

    Coordinates SIMIND Monte Carlo simulations across multiple subset projectors
    in iterative reconstruction algorithms. This class ensures efficient simulation
    by running one MPI-parallelized simulation for all views, then distributing
    subset-specific results.

    Key Features:
    - Global iteration tracking across all subsets
    - Periodic full SIMIND simulation updates
    - Result caching and distribution to subsets via get_subset()
    - Cumulative additive tracking for eta updates in CIL KL functions

    Attributes:
        simind_simulator (SimindSimulator): The shared SIMIND simulator instance.
        num_subsets (int): Number of data subsets.
        correction_update_interval (int): Subiterations between SIMIND updates.
        residual_correction (bool): Enable residual correction mode.
        update_additive (bool): Enable additive term update mode.
        linear_acquisition_model (AcquisitionModel): Full-data STIR linear model.
        stir_acquisition_model (AcquisitionModel): Full-data STIR model with additive.
        total_iterations (int, optional): Total subiterations for reconstruction.
    """

    def __init__(
        self,
        simind_simulator,
        num_subsets,
        correction_update_interval,
        residual_correction=False,
        update_additive=False,
        linear_acquisition_model=None,
        stir_acquisition_model=None,
        output_dir=None,
        total_iterations=None,
    ):
        """
        Initialize the SimindCoordinator.

        Args:
            simind_simulator (SimindSimulator): The SIMIND simulator instance.
            num_subsets (int): Number of data subsets.
            correction_update_interval (int): Subiterations between updates.
            residual_correction (bool): Enable residual correction.
            update_additive (bool): Enable additive term updates.
            linear_acquisition_model (AcquisitionModel, optional): Full-data STIR linear model (no additive).
                Required for residual correction modes. Will be set to num_subsets=1, subset_num=0.
            stir_acquisition_model (AcquisitionModel, optional): Full-data STIR model with additive.
                Required for mode_both.
            total_iterations (int, optional): Total number of subiterations for the reconstruction.
                If provided, updates will be skipped in the final correction_update_interval block
                to avoid wasting computation on corrections that won't be used.
        """
        self.simind_simulator = simind_simulator
        self.num_subsets = num_subsets
        self.correction_update_interval = correction_update_interval
        self.residual_correction = residual_correction
        self.update_additive = update_additive
        self.output_dir = output_dir
        self.total_iterations = total_iterations

        # Store full-data acquisition models
        self._linear_acquisition_model = linear_acquisition_model
        self.stir_acquisition_model = stir_acquisition_model

        # Cached full simulation results
        self.cached_b01 = None  # Full PENETRATE output (all interactions)
        self.cached_b02 = None  # Full geometric primary (for scaling)
        self.cached_linear_proj = None  # Full STIR linear projection
        self.cached_stir_full_proj = None  # Full STIR projection with additive
        self.cached_residual_full = None  # Full residual correction (all views)
        self.cached_scale_factor = None
        self._cache_version = 0  # Increments each simulation run

        # Track cumulative additive term for eta updates
        self.current_additive = None  # Full additive term (all views)
        self.initial_additive = None

        self._algorithm = None
        self._last_update_iteration = -1

        # Determine correction mode
        self.mode_residual_only = residual_correction and not update_additive
        self.mode_additive_only = update_additive and not residual_correction
        self.mode_both = update_additive and residual_correction

        # Validate required parameters for each mode
        if self.mode_residual_only and linear_acquisition_model is None:
            raise ValueError(
                "Mode A (residual_correction=True, update_additive=False) requires "
                "linear_acquisition_model to compute residuals."
            )
        if self.mode_both and linear_acquisition_model is None:
            raise ValueError(
                "Mode C (residual_correction=True, update_additive=True) requires "
                "linear_acquisition_model to compute residuals."
            )

        # Configure SIMIND simulator based on mode
        self._configure_simulator()

        logging.info(
            f"SimindCoordinator initialized: {num_subsets} subsets, "
            f"update every {correction_update_interval} subiterations"
        )
        logging.info(
            f"Correction mode: residual_only={self.mode_residual_only}, "
            f"additive_only={self.mode_additive_only}, both={self.mode_both}"
        )

    def _configure_simulator(self):
        """Configure SIMIND simulator based on correction mode."""

        # Get energy from config if available
        if hasattr(self.simind_simulator, "config"):
            config = self.simind_simulator.config

            # Set photon energy if available in config
            try:
                photon_energy = config.get_value("photon_energy")
                if photon_energy and photon_energy > 0:
                    self.simind_simulator.add_config_value(
                        "photon_energy", photon_energy
                    )
                    logging.info(f"Set photon energy from config: {photon_energy} keV")
            except (KeyError, AttributeError):
                logging.warning(
                    "Photon energy not found in config, please set manually"
                )

        # Ensure penetrate scoring routine for access to b01/b02 outputs
        self.simind_simulator.set_scoring_routine(ScoringRoutine.PENETRATE)

        if self.mode_residual_only:
            # Mode A: No penetration physics (geometric only)
            # scoring=PENETRATE but penetration physics OFF (index 53=0)
            # photon_direction=2 uses collimator hole sizes for solid angle
            self.simind_simulator.set_collimator_routine(False)
            self.simind_simulator.add_config_value(19, 2)  # photon_direction
            logging.info(
                "SIMIND configured: Mode A (scoring=PENETRATE, penetration=OFF, photon_direction=2)"
            )
        elif self.mode_additive_only or self.mode_both:
            # Mode B/C: Penetration physics enabled
            # scoring=PENETRATE and penetration physics ON (index 53=1)
            # photon_direction=3 uses phantom dimensions to cover whole camera surface
            self.simind_simulator.set_collimator_routine(True)
            self.simind_simulator.add_config_value(19, 3)  # photon_direction
            logging.info(
                "SIMIND configured: Mode B/C (scoring=PENETRATE, penetration=ON, photon_direction=3)"
            )

    def initialize_with_additive(self, initial_additive):
        """
        Initialize coordinator cache with existing additive term.

        This allows skipping the first SIMIND simulation by using the
        existing scatter estimate. Useful when starting with a good
        initial additive term from SIRF.

        Args:
            initial_additive (AcquisitionData): Initial additive term (full data, all views).
        """
        # Store initial additive term (will be updated after simulations)
        self.current_additive = initial_additive.clone()
        self.initial_additive = initial_additive.clone()

        # Mark cache as initialized (version 0 -> no updates yet)
        # Subsets will use their initial additive terms until first simulation
        self.cache_version = 0
        logging.info(
            "Coordinator initialized with existing additive term (no initial simulation)"
        )

    def should_update(self):
        """
        Check if SIMIND simulation should run this subiteration.

        Returns:
            bool: True if simulation should be triggered.
        """
        if self.correction_update_interval <= 0:
            return False

        if not (self.residual_correction or self.update_additive):
            return False

        # Don't update on the very first iteration (iteration 0)
        if self.algorithm.iteration == 0:
            return False

        # Check if enough iterations have passed since last update
        iterations_since_update = self.algorithm.iteration - self._last_update_iteration

        # Trigger when we've completed the full interval
        # Example: interval=3, last_update=-1 → update at iter 2 (2 - (-1) = 3)
        if iterations_since_update < self.correction_update_interval:
            return False

        # If total_iterations is set, check if we're too close to the end
        # Don't update if we're in the final correction_update_interval block
        if self.total_iterations is not None:
            # Calculate start of "do-not-simulate" zone (account for -1 offset)
            final_block_start = self.total_iterations - self.correction_update_interval
            if self.algorithm.iteration >= final_block_start:
                logging.info(
                    f"Skipping update at iteration {self.algorithm.iteration}: "
                    f"too close to end (total={self.total_iterations}, "
                    f"final_block_start={final_block_start})"
                )
                return False

        # Log when update is triggered
        logging.info(
            f"SimindCoordinator triggering update at iteration "
            f"{self.algorithm.iteration} "
            f"(iterations_since_update={iterations_since_update}, "
            f"interval={self.correction_update_interval})"
        )

        return True

    def run_full_simulation(self, image):
        """
        Run one full SIMIND simulation (all views) and cache results.

        This is called when should_update() returns True. The simulation is
        MPI-parallelized across cores as configured in the SIMIND simulator.

        Args:
            image (ImageData): Current image estimate.
        """
        from sirf_simind_connection.core.components import PenetrateOutputType

        if not self.algorithm:
            raise ValueError("Algorithm must be set")

        # Guard: skip if already updated in this iteration
        if self._last_update_iteration == self.algorithm.iteration:
            return

        logging.info(
            f"Running full SIMIND simulation at subiteration {self.algorithm.iteration}"
        )

        # Compute STIR linear projection upfront to align geometries
        if self.linear_acquisition_model is None:
            raise RuntimeError(
                "linear_acquisition_model not set. Pass it to coordinator.__init__"
            )

        self.linear_acquisition_model.num_subsets = 1
        self.linear_acquisition_model.subset_num = 0
        self.cached_linear_proj = self.linear_acquisition_model.forward(image)

        # Ensure SIMIND outputs share the same geometry as the STIR projector
        if getattr(self.simind_simulator, "template_sinogram", None) is None:
            try:
                self.simind_simulator.set_template_sinogram(self.cached_linear_proj)
            except AttributeError:
                logging.debug("Simulator does not support template_sinogram")

        # Update simulator source
        self.simind_simulator.set_source(image.clone())

        # Run SIMIND simulation
        self.simind_simulator.run_simulation()

        # Get SIMIND outputs based on mode
        if self.mode_residual_only:
            # Mode A: Geometric output only (GEOM_COLL_PRIMARY_ATT = b02)
            self.cached_b02 = self.simind_simulator.get_penetrate_output(
                PenetrateOutputType.GEOM_COLL_PRIMARY_ATT
            )
            self.cached_b01 = None  # Not needed for residual_only mode

        elif self.mode_additive_only or self.mode_both:
            # Mode B/C: PENETRATE outputs
            self.cached_b01 = self.simind_simulator.get_penetrate_output(
                PenetrateOutputType.ALL_INTERACTIONS
            )
            self.cached_b02 = self.simind_simulator.get_penetrate_output(
                PenetrateOutputType.GEOM_COLL_PRIMARY_ATT
            )

        # Compute scaling factor
        if self.mode_residual_only:
            # Scale b02 (geometric output) to match linear projection
            self.cached_scale_factor = self.cached_linear_proj.sum() / max(
                self.cached_b02.sum(), 1e-10
            )
        else:
            # Scale using b02 vs linear projection
            self.cached_scale_factor = self.cached_linear_proj.sum() / max(
                self.cached_b02.sum(), 1e-10
            )

        # For mode_both, also get full STIR projection with additive
        if self.mode_both:
            if self.stir_acquisition_model is None:
                raise ValueError(
                    "stir_acquisition_model required for mode_both (residual + additive). "
                    "Pass it to coordinator.__init__"
                )
            self.stir_acquisition_model.num_subsets = 1
            self.stir_acquisition_model.subset_num = 0
            self.cached_stir_full_proj = self.stir_acquisition_model.forward(image)

        # --- Update current_additive based on correction mode ---
        # This term represents the *total* additive data for the next eta update in CIL.

        if self.mode_residual_only:
            # Mode A: only residual correction. Preserve the original additive estimate.
            b02_scaled = self.cached_b02 * self.cached_scale_factor
            self.cached_residual_full = b02_scaled - self.cached_linear_proj
            if self.initial_additive is not None:
                self.current_additive = self.initial_additive.clone()
            else:
                self.current_additive = self.cached_linear_proj.get_uniform_copy(0)

        elif self.mode_additive_only:
            # Mode B: replace additive term with SIMIND scatter estimate.
            b01_scaled = self.cached_b01 * self.cached_scale_factor
            b02_scaled = self.cached_b02 * self.cached_scale_factor
            self.current_additive = b01_scaled - b02_scaled
            self.cached_residual_full = None

        elif self.mode_both:
            # Mode C: store both additive (scatter) and residual (geometric) corrections.
            b01_scaled = self.cached_b01 * self.cached_scale_factor
            b02_scaled = self.cached_b02 * self.cached_scale_factor
            self.current_additive = b01_scaled - b02_scaled
            self.cached_residual_full = b02_scaled - self.cached_linear_proj

        # --- End of update logic ---

        # Update tracking
        self._last_update_iteration = self.algorithm.iteration
        self._cache_version += 1

        # Save intermediate results if output_dir is set
        if self.output_dir:
            import os

            # Save cached projections and corrections
            if self.cached_b01:
                self.cached_b01.write(
                    os.path.join(self.output_dir, f"b01_iter_{self.cache_version}.hs")
                )
            if self.cached_b02:
                self.cached_b02.write(
                    os.path.join(self.output_dir, f"b02_iter_{self.cache_version}.hs")
                )
            if self.cached_linear_proj:
                self.cached_linear_proj.write(
                    os.path.join(
                        self.output_dir, f"linear_proj_iter_{self.cache_version}.hs"
                    )
                )
            if self.cached_stir_full_proj:
                self.cached_stir_full_proj.write(
                    os.path.join(
                        self.output_dir, f"stir_full_proj_iter_{self.cache_version}.hs"
                    )
                )
            if self.cached_residual_full is not None:
                self.cached_residual_full.write(
                    os.path.join(
                        self.output_dir,
                        f"residual_iter_{self.cache_version}.hs",
                    )
                )
            if self.current_additive:
                self.current_additive.write(
                    os.path.join(
                        self.output_dir,
                        f"current_additive_iter_{self.cache_version}.hs",
                    )
                )

        logging.info(
            f"SIMIND simulation complete: scale_factor={self.cached_scale_factor:.6f}, "
            f"cache_version={self.cache_version}, "
            + (
                f"current_additive sum={self.current_additive.sum():.2e}"
                if self.current_additive is not None
                else ""
            )
        )

    def get_subset_residual(self, subset_indices, current_additive_subset=None):
        """
        Get residual correction for a specific subset.

        Computes the full residual first, then extracts the relevant views for
        this subset using get_subset(subset_indices). This ensures ProjDataInfo
        compatibility when performing arithmetic operations.

        Args:
            subset_indices (list): View indices for this subset.
            current_additive_subset (AcquisitionData, optional): Current additive
                term estimate for this subset (required for mode_both when
                additive updates are applied).

        Returns:
            AcquisitionData: Residual correction for this subset's views.
        """
        if self.mode_residual_only:
            # Mode A: residual correction = SIMIND geometric - fast linear projection.
            if self.cached_residual_full is None:
                raise RuntimeError(
                    "No cached residual available. Call run_full_simulation() first."
                )
            residual_full = self.cached_residual_full

        elif self.mode_additive_only:
            # Mode B: no residual correction.
            residual_full = self.cached_linear_proj.get_uniform_copy(0)

        elif self.mode_both:
            # Mode C: residual correction identical to Mode A.
            if self.cached_residual_full is None:
                raise RuntimeError(
                    "No cached residual available. Call run_full_simulation() first."
                )
            residual_full = self.cached_residual_full

        else:
            # No correction
            if current_additive_subset is not None:
                return current_additive_subset.get_uniform_copy(0)
            return self.cached_linear_proj.get_uniform_copy(0)

        return residual_full.get_subset(subset_indices)

    def get_full_additive_term(self):
        """
        Get the current full additive term for updating eta in CIL KL functions.

        This term is calculated during `run_full_simulation` based on the mode:
        - **Mode A (residual_only)**: unchanged initial additive estimate.
        - **Mode B (additive_only)**: `simind_full_projection - simind_geometric`.
        - **Mode C (both)**: `simind_full_projection - simind_geometric`.

        Returns:
            AcquisitionData: The most recently computed additive term for all views,
            or None if not initialized.
        """
        # Return current_additive if it exists (either from initialization or simulation)
        return self.current_additive

    def reset_iteration_counter(self):
        """
        Reset the global iteration counter.

        Useful for multi-stage reconstructions where you want to restart
        the correction update schedule.
        """
        self._last_update_iteration = -1
        self._last_algorithm_iteration = None
        logging.info("SimindCoordinator iteration counter reset")

    def run_accurate_projection(self, image):
        """
        Alias for run_full_simulation() to conform to Coordinator interface.

        Args:
            image (ImageData): Current image estimate.
        """
        return self.run_full_simulation(image)

    # Properties required by Coordinator base class
    @property
    def cache_version(self):
        """Version number incremented after each update."""
        return self._cache_version

    @cache_version.setter
    def cache_version(self, value):
        self._cache_version = value

    @property
    def algorithm(self):
        """Reference to CIL algorithm for iteration tracking."""
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value):
        self._algorithm = value


class StirPsfCoordinator(Coordinator):
    """
    StirPsfCoordinator Class

    Coordinates STIR PSF-based corrections for iterative reconstruction.
    Uses a STIR projector with PSF modeling as the "accurate" projection
    and a fast STIR projector without PSF as the "fast" projection.

    This coordinator computes residual corrections:
        residual = STIR_PSF.forward(x) - STIR_fast.forward(x)

    to capture PSF effects without requiring Monte Carlo simulations.

    Attributes:
        stir_psf_projector (AcquisitionModel): STIR projector with PSF model.
        stir_fast_projector (AcquisitionModel): Fast STIR projector (no PSF).
        correction_update_interval (int): Subiterations between updates.
        initial_additive (AcquisitionData): Fixed additive term (scatter).
        total_iterations (int, optional): Total subiterations for reconstruction.
    """

    def __init__(
        self,
        stir_psf_projector,
        stir_fast_projector,
        correction_update_interval,
        initial_additive=None,
        output_dir=None,
        total_iterations=None,
    ):
        """
        Initialize the StirPsfCoordinator.

        Args:
            stir_psf_projector (AcquisitionModel): Full-data STIR projector
                with PSF modeling (accurate but slow).
            stir_fast_projector (AcquisitionModel): Full-data STIR projector
                without PSF (fast, for residual computation).
            correction_update_interval (int): Subiterations between updates.
            initial_additive (AcquisitionData, optional): Fixed additive term.
            output_dir (str, optional): Directory for saving intermediate results.
            total_iterations (int, optional): Total subiterations for reconstruction.
        """
        self.stir_psf_projector = stir_psf_projector
        self.stir_fast_projector = stir_fast_projector
        self.correction_update_interval = correction_update_interval
        self.output_dir = output_dir
        self.total_iterations = total_iterations

        # Store fast projector as linear_acquisition_model
        # (for compatibility with cil_partitioner)
        self._linear_acquisition_model = stir_fast_projector

        # Fixed additive term (scatter estimate, doesn't change)
        self.initial_additive = initial_additive
        self.current_additive = None

        # Cached projection results
        self.cached_psf_proj = None
        self.cached_fast_proj = None
        self._cache_version = 0

        # Iteration tracking
        self._algorithm = None
        self._last_update_iteration = -1

        logging.info(
            f"StirPsfCoordinator initialized: "
            f"update every {correction_update_interval} subiterations"
        )
        logging.info("Correction mode: PSF residual (STIR_PSF - STIR_fast)")

    def initialize_with_additive(self, initial_additive):
        """
        Initialize coordinator with fixed additive term.

        Args:
            initial_additive (AcquisitionData): Initial additive term (all views).
        """
        self.initial_additive = initial_additive.clone()
        self.current_additive = initial_additive.clone()
        self._cache_version = 0
        logging.info("StirPsfCoordinator initialized with fixed additive term")

    def should_update(self):
        """
        Check if PSF correction should run this subiteration.

        Returns:
            bool: True if correction should be computed.
        """
        if self.correction_update_interval <= 0:
            return False

        # Don't update on the very first iteration (iteration 0)
        if self.algorithm.iteration == 0:
            return False

        # Check if enough iterations have passed since last update
        iterations_since_update = self.algorithm.iteration - self._last_update_iteration

        # Trigger when we've completed the full interval
        # Example: interval=3, last_update=-1 → update at iter 2 (2 - (-1) = 3)
        if iterations_since_update < self.correction_update_interval:
            return False

        # If total_iterations is set, check if we're too close to the end
        if self.total_iterations is not None:
            final_block_start = self.total_iterations - self.correction_update_interval
            if self.algorithm.iteration >= final_block_start:
                logging.info(
                    f"Skipping update at iteration {self.algorithm.iteration}: "
                    f"too close to end (total={self.total_iterations}, "
                    f"final_block_start={final_block_start})"
                )
                return False

        # Log when update is triggered
        logging.info(
            f"StirPsfCoordinator triggering update at iteration "
            f"{self.algorithm.iteration} "
            f"(iterations_since_update={iterations_since_update}, "
            f"interval={self.correction_update_interval})"
        )

        return True

    def run_accurate_projection(self, image):
        """
        Run PSF projection and compute residual corrections.

        Computes:
            residual = STIR_PSF.forward(x) - STIR_fast.forward(x)
            current_additive = initial_additive + residual

        Args:
            image (ImageData): Current image estimate.
        """
        if not self.algorithm:
            raise ValueError("Algorithm must be set")

        # Guard: skip if already updated in this iteration
        if self._last_update_iteration == self.algorithm.iteration:
            return

        logging.info(
            f"Running STIR PSF projection at subiteration {self.algorithm.iteration}"
        )

        # Compute PSF projection (accurate)
        self.stir_psf_projector.num_subsets = 1
        self.stir_psf_projector.subset_num = 0
        self.cached_psf_proj = self.stir_psf_projector.forward(image)

        # Compute fast projection (no PSF)
        self.stir_fast_projector.num_subsets = 1
        self.stir_fast_projector.subset_num = 0
        self.cached_fast_proj = self.stir_fast_projector.forward(image)

        # Compute residual
        residual = self.cached_psf_proj - self.cached_fast_proj

        # Update cumulative additive
        # The new additive term is the initial scatter + the PSF residual.
        if self.initial_additive is not None:
            self.current_additive = self.initial_additive + residual
        else:
            # If no initial additive, the current additive is just the residual
            self.current_additive = residual

        # Update tracking
        self._last_update_iteration = self.algorithm.iteration
        self._cache_version += 1

        # Save intermediate results if output_dir is set
        if self.output_dir:
            import os

            self.cached_psf_proj.write(
                os.path.join(self.output_dir, f"psf_proj_iter_{self.cache_version}.hs")
            )
            self.cached_fast_proj.write(
                os.path.join(self.output_dir, f"fast_proj_iter_{self.cache_version}.hs")
            )
            residual.write(
                os.path.join(
                    self.output_dir,
                    f"residual_iter_{self.cache_version}.hs",
                )
            )
            self.current_additive.write(
                os.path.join(
                    self.output_dir,
                    f"current_additive_iter_{self.cache_version}.hs",
                )
            )

        logging.info(
            f"PSF projection complete: cache_version={self.cache_version}, "
            f"current_additive sum={self.current_additive.sum():.2e}"
        )

    def get_full_additive_term(self):
        """
        Get the current full additive term (all views).

        Returns:
            AcquisitionData: Full additive term, or None if not initialized.
        """
        return self.current_additive

    def get_subset_residual(self, subset_indices, current_additive_subset=None):
        """
        Get residual correction for a specific subset.

        Args:
            subset_indices (list): View indices for this subset.
            current_additive_subset (AcquisitionData, optional): Not used
                for StirPsfCoordinator (kept for interface compatibility).

        Returns:
            AcquisitionData: Residual Subset
        """
        if self.current_additive is None:
            raise RuntimeError(
                "No cached corrections. Call run_accurate_projection() first."
            )

        # This method is not used by the UpdateEtaCallback, but for correctness,
        # it should return the residual part only.
        return (self.cached_psf_proj - self.cached_fast_proj).get_subset(subset_indices)

    def reset_iteration_counter(self):
        """
        Reset the iteration counter.

        Useful for multi-stage reconstructions where you want to restart
        the correction update schedule.
        """
        self._last_update_iteration = -1
        logging.info("StirPsfCoordinator iteration counter reset")

    # Properties required by Coordinator base class
    @property
    def cache_version(self):
        """Version number incremented after each update."""
        return self._cache_version

    @cache_version.setter
    def cache_version(self, value):
        self._cache_version = value

    @property
    def algorithm(self):
        """Reference to CIL algorithm for iteration tracking."""
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value):
        self._algorithm = value

"""
SimindCoordinator Module

This module defines the SimindCoordinator class, which manages shared SIMIND
Monte Carlo simulations across multiple subset projectors in iterative reconstruction.

The coordinator ensures efficient MPI-parallelized simulations by running one full
simulation (all views) and distributing subset-specific results to multiple
SimindProjector instances.
"""

import logging

import numpy as np

from sirf_simind_connection.utils import get_array

from .types import ScoringRoutine


# Conditional import for SIRF to avoid CI dependencies
try:
    from sirf.STIR import AcquisitionData, ImageData

    SIRF_AVAILABLE = True
except ImportError:
    AcquisitionData = type(None)
    ImageData = type(None)
    SIRF_AVAILABLE = False


class _ResidualSubset:
    """Lightweight array-backed container mimicking AcquisitionData subset."""

    def __init__(self, data: np.ndarray):
        self._data = np.array(data, copy=True)

    def dimensions(self):
        return self._data.ndim

    def as_array(self):
        return self._data.copy()

    def sum(self):
        return float(self._data.sum())


class SimindCoordinator:
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
        """
        self.simind_simulator = simind_simulator
        self.num_subsets = num_subsets
        self.correction_update_interval = correction_update_interval
        self.residual_correction = residual_correction
        self.update_additive = update_additive
        self.output_dir = output_dir

        # Store full-data acquisition models
        self.linear_acquisition_model = linear_acquisition_model
        self.stir_acquisition_model = stir_acquisition_model

        # Global iteration tracking
        self.global_subiteration = 0
        self.last_update_iteration = -1

        # Cached full simulation results
        self.cached_b01 = None  # Full PENETRATE output (all interactions)
        self.cached_b02 = None  # Full geometric primary (for scaling)
        self.cached_linear_proj = None  # Full STIR linear projection
        self.cached_stir_full_proj = None  # Full STIR projection with additive
        self.cached_scale_factor = None
        self.cache_version = 0  # Increments each simulation run

        # Track cumulative additive term for eta updates
        self.cumulative_additive = None  # Full additive term (all views)

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

    def increment_iteration(self):
        """
        Increment global subiteration counter.

        Called by any SimindProjector.forward() to track total subiterations
        across all subsets.
        """
        self.global_subiteration += 1

    def initialize_with_additive(self, initial_additive):
        """
        Initialize coordinator cache with existing additive term.

        This allows skipping the first SIMIND simulation by using the
        existing scatter estimate. Useful when starting with a good
        initial additive term from SIRF.

        Args:
            initial_additive (AcquisitionData): Initial additive term (full data, all views).
        """
        # Store initial cumulative additive (will be updated after simulations)
        self.cumulative_additive = initial_additive.clone()

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

        # Check if we've hit the update interval
        iterations_since_update = self.global_subiteration - self.last_update_iteration
        return iterations_since_update >= self.correction_update_interval

    def run_full_simulation(self, image):
        """
        Run one full SIMIND simulation (all views) and cache results.

        This is called when should_update() returns True. The simulation is
        MPI-parallelized across cores as configured in the SIMIND simulator.

        Args:
            image (ImageData): Current image estimate.
        """
        from sirf_simind_connection.core.components import PenetrateOutputType

        logging.info(
            f"Running full SIMIND simulation at subiteration {self.global_subiteration}"
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

        # Update cumulative additive term based on residuals
        # This tracks the total additive estimate for eta updates
        if self.cumulative_additive is None:
            # Initialize if not set
            self.cumulative_additive = self.cached_linear_proj.get_uniform_copy(0)

        # Compute residual based on mode
        if self.mode_residual_only:
            # Mode A: Corrects PROJECTION (geometric modeling)
            # residual = b02_scaled - linear_proj (geometric SIMIND vs geometric STIR)
            b02_scaled = self.cached_b02 * self.cached_scale_factor
            residual_full = b02_scaled - self.cached_linear_proj

        elif self.mode_additive_only:
            # Mode B: Corrects ADDITIVE TERM (scatter estimate)
            # additive = b01_scaled - b02_scaled (REPLACES current)
            b01_scaled = self.cached_b01 * self.cached_scale_factor
            b02_scaled = self.cached_b02 * self.cached_scale_factor
            self.cumulative_additive = b01_scaled - b02_scaled
            residual_full = None  # No residual, just replacement

        elif self.mode_both:
            # Mode C: Corrects BOTH projection and additive
            # full_correction = (b01 - b02 - old_additive) + (b02 - linear_proj)
            #                 = b01 - old_additive - linear_proj
            # new_cumulative = old_additive + full_correction = b01 - linear_proj
            b01_scaled = self.cached_b01 * self.cached_scale_factor
            residual_full = (
                b01_scaled - self.cumulative_additive - self.cached_linear_proj
            )

        else:
            residual_full = None

        # Update cumulative additive (except mode_additive_only which replaces)
        if residual_full is not None:
            self.cumulative_additive = self.cumulative_additive + residual_full

        # Update tracking
        self.last_update_iteration = self.global_subiteration
        self.cache_version += 1

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
            if self.cumulative_additive:
                self.cumulative_additive.write(
                    os.path.join(
                        self.output_dir,
                        f"cumulative_additive_iter_{self.cache_version}.hs",
                    )
                )

        logging.info(
            f"SIMIND simulation complete: scale_factor={self.cached_scale_factor:.6f}, "
            f"cache_version={self.cache_version}, "
            f"cumulative_additive sum={self.cumulative_additive.sum():.2e}"
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
            # Mode A: Corrects PROJECTION (geometric modeling)
            # residual = b02_scaled - linear_proj
            b02_scaled = self.cached_b02 * self.cached_scale_factor
            residual_full = b02_scaled - self.cached_linear_proj

        elif self.mode_additive_only:
            if self.cached_b01 is None:
                raise RuntimeError(
                    "No cached simulation results. Call run_full_simulation() first."
                )

            # Mode B: Corrects ADDITIVE TERM (scatter estimate)
            # additive = b01_scaled - b02_scaled (scatter estimate)
            b01_scaled = self.cached_b01 * self.cached_scale_factor
            b02_scaled = self.cached_b02 * self.cached_scale_factor

            # Compute full scatter estimate
            additive_simind_full = b01_scaled - b02_scaled

            # For mode_additive_only, return the full scatter estimate
            # The subset projector will handle replacing the additive term
            residual_full = additive_simind_full

        elif self.mode_both:
            if self.cached_b01 is None or self.cached_stir_full_proj is None:
                raise RuntimeError(
                    "No cached simulation results. Call run_full_simulation() first."
                )

            # Mode C: Corrects BOTH projection and additive
            # residual = b01_scaled - stir_full_proj
            # where stir_full_proj includes old additive
            b01_scaled = self.cached_b01 * self.cached_scale_factor
            residual_full = b01_scaled - self.cached_stir_full_proj

        else:
            # No correction
            if current_additive_subset is not None:
                return current_additive_subset.get_uniform_copy(0)
            return self.cached_linear_proj.get_uniform_copy(0)

        # Extract subset views from full residual
        residual_np = get_array(residual_full)
        subset_array = residual_np[:, :, subset_indices, :]
        subset_array = subset_array[0]  # remove segment axis
        subset_array = np.transpose(subset_array, (1, 0, 2))
        return _ResidualSubset(subset_array)

    def get_full_additive_term(self):
        """
        Get the full cumulative additive term (all views) for updating eta in CIL KL functions.

        Returns the cumulative additive term that has been built up through SIMIND simulations:
        - Mode A (residual_only): cumulative_additive = sum of all (b02_scaled - linear_proj)
          Corrects the projection (geometric modeling)
        - Mode B (additive_only): cumulative_additive = b01_scaled - b02_scaled (latest)
          Corrects the additive term (scatter estimate) - REPLACES each time
        - Mode C (both): cumulative_additive = sum of all (b01 - old_additive - linear_proj)
          Simplifies to: b01 - linear_proj (corrects both projection and additive)

        Returns:
            AcquisitionData: Full cumulative additive term (all views), or None if not initialized.
        """
        # Return cumulative additive if it exists (either from initialization or simulation)
        return self.cumulative_additive

    def reset_iteration_counter(self):
        """
        Reset the global iteration counter.

        Useful for multi-stage reconstructions where you want to restart
        the correction update schedule.
        """
        self.global_subiteration = 0
        self.last_update_iteration = -1
        logging.info("SimindCoordinator iteration counter reset")

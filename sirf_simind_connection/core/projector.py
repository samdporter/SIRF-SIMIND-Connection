"""
SimindProjector Module

This module defines the SimindProjector class, which integrates the SIMIND Monte
Carlo SPECT simulator with the STIR library.
The SimindProjector class facilitates accurate forward projections, scatter
updates, and residual corrections to optimize the
Monte Carlo simulation process for SPECT imaging.
"""

# Conditional import for SIRF to avoid CI dependencies
from sirf_simind_connection.utils import get_array


try:
    from sirf.STIR import AcquisitionData, AcquisitionModel, ImageData, assert_validity

    SIRF_AVAILABLE = True
except ImportError:
    # Create dummy types for type hints when SIRF is not available
    AcquisitionData = type(None)
    AcquisitionModel = type(None)
    ImageData = type(None)
    SIRF_AVAILABLE = False

    # Fallback implementation if assert_validity is not available
    def assert_validity(obj, expected_type):
        if not isinstance(obj, expected_type):
            raise TypeError(
                f"Expected {expected_type.__name__}, got {type(obj).__name__}"
            )


class SimindProjector:
    """
    SimindProjector Class

    The SimindProjector combines the SIMIND Monte Carlo SPECT simulator and the
    STIR library to provide an AcquisitionModel-compatible interface with
    Monte Carlo-based corrections. It can be used as a drop-in replacement for
    STIR's AcquisitionModel in reconstruction algorithms.

    Supports three correction modes:
    1. Residual correction only: Corrects resolution modeling using geometric SIMIND
    2. Additive update only: Replaces scatter estimate with SIMIND (b01-b02)
    3. Both: Updates both scatter and resolution via residual (b01 - STIR projection)

    Attributes:
        simind_simulator (SimindSimulator): The SIMIND Monte Carlo simulator instance.
        stir_projector (AcquisitionModel): The STIR acquisition model instance.
        correction_update_interval (int): Interval for updating corrections in
            iterative processes.
        update_additive (bool): Replace additive term with SIMIND scatter.
        residual_correction (bool): Apply residual correction for resolution modeling.
    """

    def __init__(
        self,
        simind_simulator=None,
        stir_projector=None,
        image=None,
        acquisition_data=None,
        correction_update_interval=1,
        update_additive=False,
        residual_correction=False,
    ):
        """
        Initialize the SimindProjector with optional components.

        Args:
            simind_simulator (SimindSimulator, optional): Instance of the SIMIND
                simulator.
            stir_projector (AcquisitionModel, optional): Instance of the STIR
                acquisition model.
            image (ImageData, optional): Image data for projections.
            acquisition_data (AcquisitionData, optional): Acquisition data for
                simulations.
            correction_update_interval (int, optional): Interval for updating
                corrections (default=1).
            update_additive (bool, optional): Enables additive term update if True
                (replaces additive with SIMIND scatter b01-b02).
            residual_correction (bool, optional): Enables residual correction if True
                (adds residual to correct resolution modeling).
        """
        self._simind_simulator = simind_simulator
        self._stir_projector = stir_projector
        self._image = image
        self._acquisition_data = acquisition_data
        self._num_subsets = None
        self._subset_num = 0
        self._correction_update_interval = correction_update_interval

        # Configuration flags
        self.update_additive = update_additive
        self.residual_correction = residual_correction

        # Internal state for iteration tracking
        self._iteration_counter = 0
        self._last_update_iteration = -1

        # Cached acquisition models and terms
        self._linear_acquisition_model = None
        self._current_additive = None
        self._last_update_image = None

        # Deprecated - kept for backwards compatibility
        self._additive_correction = None
        self._additive_estimate = None

        # Templates
        self.acq_templ = None
        self.img_templ = None

    @property
    def simind_simulator(self):
        """Get or set the SIMIND simulator instance."""
        return self._simind_simulator

    @simind_simulator.setter
    def simind_simulator(self, value):
        if value is None:
            raise ValueError("simind_simulator cannot be None")
        self._simind_simulator = value

    @property
    def stir_projector(self):
        """Get or set the STIR acquisition model."""
        return self._stir_projector

    @stir_projector.setter
    def stir_projector(self, value):
        assert_validity(value, AcquisitionModel)
        self._stir_projector = value
        if self.num_subsets is not None:
            self.stir_projector.num_subsets = self.num_subsets

    @property
    def image(self):
        """Get or set the image data."""
        return self._image

    @image.setter
    def image(self, value):
        assert_validity(value, ImageData)
        self._image = value

    @property
    def acquisition_data(self):
        """Get or set the acquisition data."""
        return self._acquisition_data

    @acquisition_data.setter
    def acquisition_data(self, value):
        assert_validity(value, AcquisitionData)
        self._acquisition_data = value

    @property
    def num_subsets(self):
        """Get or set the number of subsets."""
        return self._num_subsets

    @num_subsets.setter
    def num_subsets(self, value):
        if isinstance(value, int) and value > 0:
            self._num_subsets = value
            if self.stir_projector is not None:
                self.stir_projector.num_subsets = value
        else:
            raise ValueError("num_subsets must be a positive integer")

    @property
    def subset_num(self):
        """Get or set the subset number."""
        return self._subset_num

    @subset_num.setter
    def subset_num(self, value):
        if isinstance(value, int) and value >= 0:
            self._subset_num = value
            if self.stir_projector is not None:
                self.stir_projector.subset_num = value
        else:
            raise ValueError("subset_num must be a non-negative integer")

    @property
    def correction_update_interval(self):
        """Get or set the correction update interval."""
        return self._correction_update_interval

    @correction_update_interval.setter
    def correction_update_interval(self, value):
        if isinstance(value, int) and value > 0:
            self._correction_update_interval = value
        else:
            raise ValueError("correction_update_interval must be a positive integer")

    @property
    def additive_correction(self):
        """Get the additive correction term."""
        return self._additive_correction

    @property
    def additive_estimate(self):
        """Get or set the additive estimate term."""
        return self._additive_estimate

    @additive_estimate.setter
    def additive_estimate(self, value):
        if isinstance(value, AcquisitionData):
            self._additive_estimate = value
            if self.stir_projector is not None:
                self.stir_projector.set_additive_term(value)
        else:
            raise ValueError("additive_estimate must be an AcquisitionData object")

    def forward(self, image, subset_num=None, num_subsets=None, out=None):
        """
        Perform forward projection using the STIR projector with optional
        Monte Carlo correction updates.

        This method auto-increments an internal iteration counter and triggers
        correction updates based on the correction_update_interval.

        Args:
            image (ImageData): Input image for forward projection.
            subset_num (int, optional): Subset index for the projection.
            num_subsets (int, optional): Total number of subsets.
            out (AcquisitionData, optional): Output acquisition data.

        Returns:
            AcquisitionData: Forward projected acquisition data.
        """
        # Auto-increment iteration counter
        self._iteration_counter += 1

        # Check if we should update corrections this iteration
        if self._should_update_corrections():
            self._update_corrections(image)
            self._last_update_iteration = self._iteration_counter

        # Always use STIR for forward projection (fast, with updated additive)
        return self._stir_projector.forward(image, subset_num, num_subsets, out)

    def backward(self, acquisition_data, subset_num=None, num_subsets=None, out=None):
        """
        Perform backward projection using the STIR projector.

        Args:
            acquisition_data (AcquisitionData): Input acquisition data.
            subset_num (int, optional): Subset index for the projection.
            num_subsets (int, optional): Total number of subsets.
            out (ImageData, optional): Output image data.

        Returns:
            ImageData: Backward projected image data.
        """
        return self.stir_projector.backward(
            acquisition_data, subset_num, num_subsets, out
        )

    def range_geometry(self):
        """
        Get the range geometry of the projector.

        Returns:
            AcquisitionData: Range geometry of the projector.
        """
        if self.stir_projector is None:
            raise ValueError("stir_projector cannot be None")
        return self.stir_projector.range_geometry()

    def domain_geometry(self):
        """
        Get the domain geometry of the projector.

        Returns:
            ImageData: Domain geometry of the projector.
        """
        if self.stir_projector is None:
            raise ValueError("stir_projector cannot be None")
        return self.stir_projector.domain_geometry()

    def set_up(self, acq_templ, img_templ):
        """
        Set up the projector with acquisition and image templates.

        This method initializes the projector geometry and creates a linear
        acquisition model for scaling and residual corrections.

        Args:
            acq_templ (AcquisitionData): Template for acquisition data.
            img_templ (ImageData): Template for image data.
        """
        assert_validity(acq_templ, AcquisitionData)
        assert_validity(img_templ, ImageData)

        self.acq_templ = acq_templ
        self.img_templ = img_templ

        # Set up STIR projector if not already done
        if self._stir_projector is not None:
            self._stir_projector.set_up(acq_templ, img_templ)

            # Create linear acquisition model (no additive/background)
            self._linear_acquisition_model = (
                self._stir_projector.get_linear_acquisition_model()
            )
            self._linear_acquisition_model.num_subsets = 1

            # Initialize current additive term
            self._current_additive = self._stir_projector.get_additive_term()

    def reset_iteration_counter(self):
        """
        Reset the iteration counter.

        Useful for multi-stage reconstructions where you want to restart
        the correction update schedule.
        """
        self._iteration_counter = 0
        self._last_update_iteration = -1

    def _should_update_corrections(self):
        """
        Check if corrections should be updated this iteration.

        Returns:
            bool: True if corrections should be updated.
        """
        if not (self.update_additive or self.residual_correction):
            return False

        if self._simind_simulator is None:
            return False

        if self._correction_update_interval <= 0:
            return False

        return (self._iteration_counter % self._correction_update_interval) == 0

    def _update_corrections(self, image):
        """
        Update corrections using SIMIND simulation.

        Implements three correction modes:
        1. Residual only: SIMIND geometric vs linear STIR
        2. Additive only: Replace with SIMIND scatter (b01-b02)
        3. Both: Residual between SIMIND full and STIR full projection

        Args:
            image (ImageData): Current image estimate.
        """
        from sirf_simind_connection.core.components import PenetrateOutputType

        if self._linear_acquisition_model is None:
            raise RuntimeError(
                "SimindProjector.set_up() must be called before corrections"
            )

        # Determine correction mode
        mode_residual_only = self.residual_correction and not self.update_additive
        mode_additive_only = self.update_additive and not self.residual_correction
        mode_both = self.update_additive and self.residual_correction

        # Configure SIMIND simulator
        self._simind_simulator.set_source(image.clone())

        # Mode A: Residual correction only (no penetration needed)
        if mode_residual_only:
            # Run SIMIND without collimator modeling (index 53=0)
            self._simind_simulator.set_collimator_routine(False)
            self._simind_simulator.run_simulation()

            # Get geometric output
            b01 = self._simind_simulator.get_output_total(window=1)

            # Get linear STIR projection
            linear_proj = self._linear_acquisition_model.forward(image)

            # Scale SIMIND to match STIR
            scale_factor = linear_proj.sum() / max(b01.sum(), 1e-10)
            b01_scaled = b01.clone()
            b01_scaled.fill(get_array(b01) * scale_factor)

            # Compute residual
            residual = b01_scaled - linear_proj

            # Update additive
            if self._current_additive is None:
                self._current_additive = self.acq_templ.get_uniform_copy(0)
            new_additive = self._current_additive + residual

        # Mode B: Additive update only (needs penetrate)
        elif mode_additive_only:
            # Run SIMIND with collimator modeling (index 53=1)
            self._simind_simulator.set_collimator_routine(True)
            self._simind_simulator.run_simulation()

            # Get penetrate outputs
            b01 = self._simind_simulator.get_penetrate_output(
                PenetrateOutputType.ALL_INTERACTIONS
            )
            b02 = self._simind_simulator.get_penetrate_output(
                PenetrateOutputType.GEOM_COLL_PRIMARY_ATT
            )

            # Get linear STIR projection for scaling
            linear_proj = self._linear_acquisition_model.forward(image)

            # Scale using b02 vs linear projection
            scale_factor = linear_proj.sum() / max(b02.sum(), 1e-10)
            b01_scaled = b01.clone()
            b01_scaled.fill(get_array(b01) * scale_factor)
            b02_scaled = b02.clone()
            b02_scaled.fill(get_array(b02) * scale_factor)

            # Compute additive as scatter (b01 - b02)
            additive_simind = b01_scaled - b02_scaled
            new_additive = additive_simind

        # Mode C: Both (residual implicitly updates scatter + resolution)
        elif mode_both:
            # Run SIMIND with collimator modeling (index 53=1)
            self._simind_simulator.set_collimator_routine(True)
            self._simind_simulator.run_simulation()

            # Get penetrate outputs
            b01 = self._simind_simulator.get_penetrate_output(
                PenetrateOutputType.ALL_INTERACTIONS
            )
            b02 = self._simind_simulator.get_penetrate_output(
                PenetrateOutputType.GEOM_COLL_PRIMARY_ATT
            )

            # Get linear STIR projection for scaling
            linear_proj = self._linear_acquisition_model.forward(image)

            # Scale using b02 vs linear projection
            scale_factor = linear_proj.sum() / max(b02.sum(), 1e-10)
            b01_scaled = b01.clone()
            b01_scaled.fill(get_array(b01) * scale_factor)

            # Get full STIR projection (with current additive)
            stir_full_proj = self._stir_projector.forward(image)

            # Residual implicitly updates both scatter and resolution
            residual = b01_scaled - stir_full_proj

            # Update additive
            if self._current_additive is None:
                self._current_additive = self.acq_templ.get_uniform_copy(0)
            new_additive = self._current_additive + residual

        else:
            # No correction mode enabled
            return

        # Apply maximum(0) to avoid negative values
        new_additive = new_additive.maximum(0)

        # Update STIR projector and cache
        self._stir_projector.set_additive_term(new_additive)
        self._current_additive = new_additive
        self._last_update_image = image.clone()

    def get_additive_term(self):
        """
        Get the additive term from the STIR projector.

        Returns:
            AcquisitionData: The additive term.
        """
        if self._stir_projector is None:
            if self.acq_templ is not None:
                return self.acq_templ.get_uniform_copy(0)
            raise RuntimeError("Projector not set up. Call set_up() first.")
        return self._stir_projector.get_additive_term()

    def set_additive_term(self, additive):
        """
        Set the additive term in the STIR projector.

        Args:
            additive (AcquisitionData): The additive term to set.
        """
        assert_validity(additive, AcquisitionData)
        if self._stir_projector is not None:
            self._stir_projector.set_additive_term(additive)
            self._current_additive = additive

    def get_background_term(self):
        """
        Get the background term from the STIR projector.

        Returns:
            AcquisitionData: The background term.
        """
        if self._stir_projector is None:
            if self.acq_templ is not None:
                return self.acq_templ.get_uniform_copy(0)
            raise RuntimeError("Projector not set up. Call set_up() first.")
        return self._stir_projector.get_background_term()

    def set_background_term(self, background):
        """
        Set the background term in the STIR projector.

        Args:
            background (AcquisitionData): The background term to set.
        """
        assert_validity(background, AcquisitionData)
        if self._stir_projector is not None:
            self._stir_projector.set_background_term(background)

    def get_constant_term(self):
        """
        Get the constant term (additive + background).

        Returns:
            AcquisitionData: The combined constant term.
        """
        return self.get_additive_term() + self.get_background_term()

    def get_linear_acquisition_model(self):
        """
        Get the linear acquisition model (no additive/background terms).

        Returns:
            AcquisitionModel: The linear acquisition model.
        """
        if self._linear_acquisition_model is None:
            raise RuntimeError("Linear model not initialized. Call set_up() first.")
        return self._linear_acquisition_model

    def is_linear(self):
        """
        Check if the acquisition model is linear (constant term is zero).

        Returns:
            bool: True if linear (no additive or background).
        """
        if self._stir_projector is None:
            return True
        return self._stir_projector.is_linear()

    def is_affine(self):
        """
        Check if the acquisition model is affine.

        Returns:
            bool: Always True for this model.
        """
        return True

    def direct(self, image, out=None):
        """
        Direct operator (alias for forward).

        Provided for compatibility with CIL framework.

        Args:
            image (ImageData): Input image.
            out (AcquisitionData, optional): Output acquisition data.

        Returns:
            AcquisitionData: Forward projected data.
        """
        return self.forward(
            image, subset_num=self.subset_num, num_subsets=self.num_subsets, out=out
        )

    def adjoint(self, acquisition_data, out=None):
        """
        Adjoint operator (alias for backward).

        Provided for compatibility with CIL framework.

        Args:
            acquisition_data (AcquisitionData): Input acquisition data.
            out (ImageData, optional): Output image.

        Returns:
            ImageData: Backprojected image.
        """
        return self.backward(
            acquisition_data,
            subset_num=self.subset_num,
            num_subsets=self.num_subsets,
            out=out,
        )


class SimindSubsetProjector:
    """
    SimindSubsetProjector Class

    A projector for a single subset that coordinates with a shared SimindCoordinator
    to efficiently manage SIMIND Monte Carlo simulations across multiple subsets.

    Unlike SimindProjector (which runs its own simulations), SimindSubsetProjector:
    - References a shared SimindCoordinator
    - Increments global iteration counter on each forward()
    - Triggers coordinator simulation when update interval is reached
    - Applies subset-specific residual corrections scaled by 1/num_subsets

    This design enables efficient MPI-parallelized SIMIND simulation of all views,
    with results distributed to individual subset projectors.

    Attributes:
        stir_projector (AcquisitionModel): STIR acquisition model for this subset.
        coordinator (SimindCoordinator): Shared coordinator managing SIMIND simulations.
        subset_indices (list): View indices handled by this subset.
        last_cache_version (int): Tracks which coordinator cache version was last applied.
    """

    def __init__(self, stir_projector, coordinator, subset_indices):
        """
        Initialize the SimindSubsetProjector.

        Args:
            stir_projector (AcquisitionModel): STIR acquisition model for this subset.
            coordinator (SimindCoordinator): Shared coordinator instance.
            subset_indices (list): View indices for this subset.
        """
        assert_validity(stir_projector, AcquisitionModel)

        self.stir_projector = stir_projector
        self.coordinator = coordinator
        self.subset_indices = subset_indices

        # Templates
        self.acq_templ = None
        self.img_templ = None

    def set_up(self, acq_templ, img_templ):
        """
        Set up the projector with acquisition and image templates.

        Args:
            acq_templ (AcquisitionData): Template for acquisition data (subset data).
            img_templ (ImageData): Template for image data.
        """
        assert_validity(acq_templ, AcquisitionData)
        assert_validity(img_templ, ImageData)

        self.acq_templ = acq_templ
        self.img_templ = img_templ

        # Set up STIR projector (skip if already set up by partitioner)
        try:
            self.stir_projector.set_up(acq_templ, img_templ)
        except RuntimeError as e:
            if "cannot set_up const object" in str(e):
                # Already set up - this happens when partitioner pre-sets up models
                pass
            else:
                raise

    def forward(self, image, subset_num=None, num_subsets=None, out=None):
        """
        Perform forward projection using STIR with coordinator-managed corrections.

        This method:
        1. Increments coordinator's global iteration counter
        2. Checks if coordinator should run SIMIND simulation
        3. Triggers simulation if needed
        4. Applies subset-specific residual correction if new results available
        5. Returns STIR forward projection

        Args:
            image (ImageData): Input image for forward projection.
            subset_num (int, optional): Subset index (ignored, managed internally).
            num_subsets (int, optional): Total subsets (ignored, managed internally).
            out (AcquisitionData, optional): Output acquisition data.

        Returns:
            AcquisitionData: Forward projected acquisition data.
        """

        # Check if coordinator should update
        if self.coordinator.should_update():
            # Run full SIMIND simulation (all views)
            # Coordinator has its own full-data acquisition models
            # After simulation, UpdateEtaCallback will update KL eta parameters
            self.coordinator.run_full_simulation(image)

        # Use STIR for fast forward projection (LINEAR model, no additive)
        # Additive correction is handled via eta in KL function, updated by UpdateEtaCallback
        result = self.stir_projector.forward(image, subset_num, num_subsets, out)
        # If out was provided and result is None, return out
        if result is None and out is not None:
            return out
        return result

    def backward(self, acquisition_data, subset_num=None, num_subsets=None, out=None):
        """
        Perform backward projection using STIR.

        Args:
            acquisition_data (AcquisitionData): Input acquisition data.
            subset_num (int, optional): Subset index (ignored).
            num_subsets (int, optional): Total subsets (ignored).
            out (ImageData, optional): Output image data.

        Returns:
            ImageData: Backward projected image data.
        """
        result = self.stir_projector.backward(
            acquisition_data, subset_num, num_subsets, out
        )
        # If out was provided and result is None, return out
        # (STIR modifies in-place but may not return)
        if result is None and out is not None:
            return out
        return result

    def range_geometry(self):
        """Get the range geometry of the projector."""
        return self.stir_projector.range_geometry()

    def domain_geometry(self):
        """Get the domain geometry of the projector."""
        return self.stir_projector.domain_geometry()

    def get_additive_term(self):
        """Get the additive term from the STIR projector."""
        return self.stir_projector.get_additive_term()

    def set_additive_term(self, additive):
        """Set the additive term in the STIR projector."""
        assert_validity(additive, AcquisitionData)
        self.stir_projector.set_additive_term(additive)

    def get_background_term(self):
        """Get the background term from the STIR projector."""
        return self.stir_projector.get_background_term()

    def set_background_term(self, background):
        """Set the background term in the STIR projector."""
        assert_validity(background, AcquisitionData)
        self.stir_projector.set_background_term(background)

    def is_linear(self):
        """Check if the acquisition model is linear (constant term is zero)."""
        return self.stir_projector.is_linear()

    def is_affine(self):
        """Check if the acquisition model is affine."""
        return True

    def direct(self, image, out=None):
        """
        Direct operator (alias for forward).

        Provided for compatibility with CIL framework.

        Args:
            image (ImageData): Input image.
            out (AcquisitionData, optional): Output acquisition data.

        Returns:
            AcquisitionData: Forward projected data.
        """
        return self.forward(image, out=out)

    def adjoint(self, acquisition_data, out=None):
        """
        Adjoint operator (alias for backward).

        Provided for compatibility with CIL framework.

        Args:
            acquisition_data (AcquisitionData): Input acquisition data.
            out (ImageData, optional): Output image.

        Returns:
            ImageData: Backprojected image.
        """
        return self.backward(acquisition_data, out=out)

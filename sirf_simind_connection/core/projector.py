"""
SimindProjector Module

This module defines the SimindProjector class, which integrates the SIMIND Monte Carlo SPECT simulator with the STIR library.
The SimindProjector class facilitates accurate forward projections, scatter updates, and residual corrections to optimize the
Monte Carlo simulation process for SPECT imaging.
"""

# Conditional import for SIRF to avoid CI dependencies
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

    The SimindProjector combines the SIMIND Monte Carlo SPECT simulator and the STIR library to perform accurate forward projections.
    It provides optional features for scatter updates and residual corrections, allowing efficient and accurate image reconstruction
    and acquisition data simulations.

    Attributes:
        simind_simulator (SimindSimulator): The SIMIND Monte Carlo simulator instance.
        stir_projector (AcquisitionModel): The STIR acquisition model instance.
        image (ImageData): The image data for forward and backward projection.
        acquisition_data (AcquisitionData): The acquisition data for simulations.
        correction_update_interval (int): Interval for updating corrections in iterative processes.
        update_scatter (bool): Indicates whether scatter updates are enabled.
        residual_correction (bool): Indicates whether residual correction is enabled.
    """

    def __init__(
        self,
        simind_simulator=None,
        stir_projector=None,
        image=None,
        acquisition_data=None,
        correction_update_interval=1,
        update_scatter=False,
        residual_correction=False,
    ):
        """
        Initialize the SimindProjector with optional components.

        Args:
            simind_simulator (SimindSimulator, optional): Instance of the SIMIND simulator.
            stir_projector (AcquisitionModel, optional): Instance of the STIR acquisition model.
            image (ImageData, optional): Image data for projections.
            acquisition_data (AcquisitionData, optional): Acquisition data for simulations.
            correction_update_interval (int, optional): Interval for updating corrections (default=1).
            update_scatter (bool, optional): Enables scatter update if True.
            residual_correction (bool, optional): Enables residual correction if True.
        """
        self._simind_simulator = simind_simulator
        self._stir_projector = stir_projector
        self._image = image
        self._acquisition_data = acquisition_data
        self._num_subsets = None
        self._subset_num = 0
        self._correction_update_interval = correction_update_interval

        self._additive_correction = None
        self._additive_estimate = None

        self.update_scatter = update_scatter
        self.residual_correction = residual_correction

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
        Perform forward projection using the STIR projector.

        Args:
            image (ImageData): Input image for forward projection.
            subset_num (int, optional): Subset index for the projection.
            num_subsets (int, optional): Total number of subsets.
            out (AcquisitionData, optional): Output acquisition data.

        Returns:
            AcquisitionData: Forward projected acquisition data.
        """
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

    def simulate_forward_projection(self, image, window=1):
        """
        Simulate the forward projection of the image using SIMIND and optionally update scatter or apply residual correction.

        Args:
            image (ImageData): Input image for forward projection.
            window (int, optional): Window index for SIMIND simulation (default=1).

        Raises:
            ValueError: If required components are not initialized.
        """

        if self._additive_estimate is None:
            try:
                self._additive_estimate = self.stir_projector.get_additive_term()
            except Exception:
                self._additive_estimate = self.acquisition_data.get_uniform_copy(0)

        projected_trues = None
        if self.residual_correction:
            linear_am = self.stir_projector.get_linear_acquisition_model()
            linear_am.num_subsets = 1
            projected_trues = linear_am.forward(image)
            projected_trues.write("projected_trues.hs")

        if self.update_scatter or self.residual_correction:
            self.simind_simulator.set_source(image.copy())
            self.simind_simulator.run_simulation()

            simulated_total = self.simind_simulator.get_output_total(window)
            simulated_scatter = self.simind_simulator.get_output_scatter(window)
            ratio = self.acquisition_data.max() / simulated_total.max()
            simulated_scatter = simulated_scatter.clone().fill(
                simulated_scatter.as_array() * ratio
            )
            simulated_total = simulated_total.clone().fill(
                simulated_total.as_array() * ratio
            )
            simulated_total.write("simulated_total.hs")
            simulated_scatter.write("simulated_scatter.hs")

        if self.update_scatter:
            self._additive_estimate = simulated_scatter

        if self.residual_correction:
            simulated_trues = simulated_total - simulated_scatter
            ratio = projected_trues.max() / simulated_trues.max()
            simulated_trues = simulated_trues.clone().fill(
                simulated_trues.as_array() * ratio
            )
            simulated_trues.write("simulated_trues.hs")

            self._additive_correction = simulated_trues - projected_trues

            updated_additive = (
                self._additive_estimate + self._additive_correction
            ).maximum(0)
            self.stir_projector.set_additive_term(updated_additive)

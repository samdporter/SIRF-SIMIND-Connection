"""
CIL-Compatible Partitioner for SIMIND

This module provides functions to partition SPECT data into subsets and create
CIL-compatible objective functions that integrate with SimindCoordinator for
efficient Monte Carlo corrections.
"""

import logging


try:
    from cil.optimisation.functions import KullbackLeibler, OperatorCompositionFunction
    from sirf.contrib.partitioner import partitioner

    CIL_AVAILABLE = True
except ImportError:
    CIL_AVAILABLE = False
    logging.warning("CIL or SIRF partitioner not available")

from sirf.STIR import AcquisitionData, AcquisitionModel, ImageData, assert_validity

from sirf_simind_connection.core.projector import SimindSubsetProjector


class CILAcquisitionModelAdapter:
    """Wrap a linear STIR acquisition model so it behaves like a CIL operator."""

    def __init__(self, stir_projector):
        assert_validity(stir_projector, AcquisitionModel)
        self._stir_projector = stir_projector

    def __getattr__(self, name):
        return getattr(self._stir_projector, name)

    def set_up(self, acq_templ, img_templ):
        assert_validity(acq_templ, AcquisitionData)
        assert_validity(img_templ, ImageData)
        self._stir_projector.set_up(acq_templ, img_templ)

    def forward(self, image, subset_num=None, num_subsets=None, out=None):
        result = self._stir_projector.forward(
            image, subset_num=subset_num, num_subsets=num_subsets, out=out
        )
        if result is None and out is not None:
            return out
        return result

    def backward(self, acquisition_data, subset_num=None, num_subsets=None, out=None):
        result = self._stir_projector.backward(
            acquisition_data,
            subset_num=subset_num,
            num_subsets=num_subsets,
            out=out,
        )
        if result is None and out is not None:
            return out
        return result

    def direct(self, image, out=None):
        result = self._stir_projector.direct(image, out=out)
        if result is None and out is not None:
            return out
        return result

    def adjoint(self, acquisition_data, out=None):
        result = self._stir_projector.adjoint(acquisition_data, out=out)
        if result is None and out is not None:
            return out
        return result

    def range_geometry(self):
        return self._stir_projector.range_geometry()

    def domain_geometry(self):
        return self._stir_projector.domain_geometry()


def partition_data_with_cil_objectives(
    acquisition_data,
    additive_data,
    multiplicative_factors,
    num_subsets,
    initial_image,
    create_acq_model,
    simind_coordinator=None,
    mode="staggered",
):
    """
    Partition SPECT data into subsets and create CIL KL objective functions.

    This function:
    1. Uses SIRF's partitioner to create subset acquisition models and data
    2. Creates LINEAR acquisition models (no additive term) for CIL compatibility
    3. Wraps each acquisition model in a SimindSubsetProjector (if coordinator provided)
    4. Creates CIL KullbackLeibler objectives: KL(b, eta) composed with LINEAR projector
       where eta = additive_term + epsilon (for numerical stability)

    Args:
        acquisition_data: Measured SPECT data (prompts).
        additive_data: Additive term (scatter + randoms estimate).
        multiplicative_factors: Bin efficiencies (normalization).
        num_subsets (int): Number of subsets.
        initial_image: Initial image for setup.
        create_acq_model: Factory function that returns STIR AcquisitionModel.
        simind_coordinator (SimindCoordinator, optional): Shared coordinator for
            SIMIND simulations. If None, no Monte Carlo corrections are applied.
            If provided, coordinator.linear_acquisition_model and coordinator.stir_acquisition_model
            must already be set with full-data acquisition models.
        mode (str): Partitioning mode ("staggered", "sequential", "random").

    Returns:
        tuple: (kl_objectives, projectors, partition_indices, kl_data_functions)
            - kl_objectives: List of CIL OperatorCompositionFunction(KL, projector)
            - projectors: List of SimindSubsetProjector or STIR AcquisitionModel (both LINEAR)
            - partition_indices: List of view indices for each subset
            - kl_data_functions: List of unwrapped KullbackLeibler functions for eta updates
    """
    if not CIL_AVAILABLE:
        raise ImportError(
            "CIL and SIRF partitioner required. Install with: pip install cil sirf"
        )

    logging.info(f"Partitioning data into {num_subsets} subsets with mode='{mode}'")

    # Validate coordinator has required full-data acquisition models
    if simind_coordinator is not None:
        if simind_coordinator.linear_acquisition_model is None:
            raise ValueError(
                "SimindCoordinator.linear_acquisition_model must be set before partitioning. "
                "Create a full-data linear acquisition model and pass it to coordinator.__init__"
            )
        if (
            simind_coordinator.mode_both
            and simind_coordinator.stir_acquisition_model is None
        ):
            raise ValueError(
                "SimindCoordinator.stir_acquisition_model must be set for mode_both. "
                "Create a full-data STIR acquisition model and pass it to coordinator.__init__"
            )

    # Use SIRF partitioner to get subset data and acquisition models
    # Pass initial_image so partitioner sets up the models properly
    # (SimindSubsetProjector.set_up handles re-setup gracefully via try-except)
    prompts_subsets, acquisition_models, _ = partitioner.data_partition(
        acquisition_data,
        additive_data,
        multiplicative_factors,
        num_subsets,
        mode=mode,
        initial_image=initial_image,
        create_acq_model=create_acq_model,
    )

    # Get partition indices from SIRF partitioner
    views = int(acquisition_data.dimensions()[2])  # Convert numpy.int32 to Python int
    partition_indices = partitioner.partition_indices(
        num_subsets, views, stagger=(mode == "staggered")
    )

    logging.info(f"Created {len(acquisition_models)} subset acquisition models")

    # Create projectors and CIL objectives
    kl_objectives = []
    kl_data_functions = []  # Store unwrapped KL functions for eta updates
    projectors = []

    for i, (prompts_subset, acq_model, subset_indices) in enumerate(
        zip(prompts_subsets, acquisition_models, partition_indices)
    ):
        # IMPORTANT: For CIL LinearOperator compatibility, extract additive term and linear model
        # The linear model has no additive term (required for CIL)
        additive_subset = acq_model.get_additive_term()
        linear_acq_model = acq_model.get_linear_acquisition_model()

        # Wrap LINEAR acquisition model with SimindSubsetProjector if coordinator provided
        if simind_coordinator is not None:
            projector = SimindSubsetProjector(
                stir_projector=linear_acq_model,
                coordinator=simind_coordinator,
                subset_indices=subset_indices,
            )
            # Set up projector
            projector.set_up(prompts_subset, initial_image)
        else:
            # No coordinator - wrap STIR linear acquisition model for CIL
            projector = CILAcquisitionModelAdapter(linear_acq_model)

        projectors.append(projector)

        # Create CIL KL function with eta = additive_term + epsilon
        # This is the CORRECT way to handle additive terms with CIL LinearOperators
        # eta will be updated after each SIMIND simulation (if residual_correction enabled)
        epsilon = 1e-5  # Small constant for numerical stability
        eta_subset = additive_subset + additive_subset.get_uniform_copy(epsilon)
        kl_data = KullbackLeibler(b=prompts_subset, eta=eta_subset)

        # Store unwrapped KL function for eta updates
        kl_data_functions.append(kl_data)

        # Compose with LINEAR projector: (KL ∘ A)(x) = KL(A(x), eta)
        kl_objective = OperatorCompositionFunction(kl_data, projector)

        kl_objectives.append(kl_objective)

        logging.info(
            f"Subset {i}: {len(subset_indices)} views, "
            f"data sum={prompts_subset.sum():.2e}, "
            f"eta sum={eta_subset.sum():.2e}"
        )

    logging.info("Created CIL KL objectives with LINEAR projectors and eta")

    # Return both composed objectives and unwrapped KL data functions
    return kl_objectives, projectors, partition_indices, kl_data_functions


def create_svrg_objective_with_rdp(
    kl_objectives,
    rdp_prior,
    sampler,
    snapshot_update_interval,
    initial_image,
):
    """
    Create SVRG objective function with RDP prior for ISTA reconstruction.

    Combines CIL SVRG function (sum of subset KL objectives) with SIRF RDP prior
    using CIL's function composition.

    Args:
        kl_objectives (list): List of CIL KL objective functions.
        rdp_prior: SETR RelativeDifferencePrior (or compatible Function).
        sampler: CIL Sampler for SVRG.
        snapshot_update_interval (int): SVRG snapshot update interval.
        initial_image: Initial image for setup.
        negate (bool): If True, negate objective (for maximization with ISTA).

    Returns:
        CIL Function: Combined objective function (SVRG + RDP).
    """
    from cil.optimisation.functions import SumFunction, SVRGFunction

    # Create SVRG function (handles subset sampling and variance reduction)
    svrg_func = SVRGFunction(
        kl_objectives,
        sampler,
        snapshot_update_interval=snapshot_update_interval,
    )

    # Combine SVRG with RDP prior
    total_objective = SumFunction(svrg_func, rdp_prior) if rdp_prior else svrg_func
    logging.info(
        f"Created SVRG objective with {len(kl_objectives)} subsets + RDP prior"
    )

    return total_objective

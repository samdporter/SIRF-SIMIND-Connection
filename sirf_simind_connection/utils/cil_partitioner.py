"""
CIL-Compatible Partitioner with Coordinator Support

This module provides functions to partition SPECT data into subsets and create
CIL-compatible objective functions that integrate with Coordinator instances
(SimindCoordinator or StirPsfCoordinator) for efficient corrections.
"""

import logging


try:
    from cil.optimisation.functions import OperatorCompositionFunction
    from sirf.contrib.partitioner import partitioner

    CIL_AVAILABLE = True
except ImportError:
    CIL_AVAILABLE = False
    logging.warning("CIL or SIRF partitioner not available")

try:
    from cil.optimisation.functions import KullbackLeibler

    CIL_FUNCTIONS_AVAILABLE = True
except ImportError:
    CIL_FUNCTIONS_AVAILABLE = False
    logging.warning("CIL KullbackLeibler not available; cannot build Poisson data term")

from sirf.STIR import AcquisitionData, AcquisitionModel, ImageData, assert_validity

from sirf_simind_connection.core.projector import CoordinatedSubsetProjector
from sirf_simind_connection.utils import get_array


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
    coordinator=None,
    mode="staggered",
    eta_floor=1e-5,
    count_floor=1e-8,
):
    """
    Partition SPECT data into subsets and create CIL KL objective functions.

    This function:
    1. Uses SIRF's partitioner to create subset acquisition models and data
    2. Creates LINEAR acquisition models (no additive term) for CIL compatibility
    3. Wraps each acquisition model in a CoordinatedSubsetProjector (if coordinator provided)
    4. Creates CIL KullbackLeibler objectives: KL(b, eta) composed with LINEAR projector
       where eta = additive_term + epsilon (for numerical stability)

    Args:
        acquisition_data: Measured SPECT data (prompts).
        additive_data: Additive term (scatter + randoms estimate).
        multiplicative_factors: Bin efficiencies (normalization).
        num_subsets (int): Number of subsets.
        initial_image: Initial image for setup.
        create_acq_model: Factory function that returns STIR AcquisitionModel.
        coordinator (Coordinator, optional): Shared coordinator for
            accurate projections (SimindCoordinator or StirPsfCoordinator).
            If None, no corrections are applied.
            For SimindCoordinator: linear_acquisition_model must be set.
            For mode_both: stir_acquisition_model must also be set.
            StirPsfCoordinator: Uses internal projectors, no additional setup needed.
        mode (str): Partitioning mode ("staggered", "sequential", "random").
        eta_floor (float): Minimum additive value applied via max(additive, eta_floor).
        count_floor (float): Minimum measured count used in the logarithm to avoid log(0).

    Returns:
        tuple: (
            kl_objectives,
            projectors,
            partition_indices,
            kl_data_functions,
            subset_sensitivity_max,
            subset_eta_min,
        )
            - kl_objectives: List of CIL OperatorCompositionFunction(KL, projector)
            - projectors: List of CoordinatedSubsetProjector or STIR AcquisitionModel (both LINEAR)
            - partition_indices: List of view indices for each subset
            - kl_data_functions: List of unwrapped KullbackLeibler functions for eta updates
            - subset_sensitivity_max: List of max row sums s_max per subset (for positivity cap)
            - subset_eta_min: Initial minimum additive values per subset
    """
    if not CIL_AVAILABLE or not CIL_FUNCTIONS_AVAILABLE:
        raise ImportError(
            "CIL (with KullbackLeibler) and SIRF partitioner required. "
            "Install with: pip install cil sirf"
        )

    logging.info(
        f"Partitioning data into {num_subsets} subsets with mode='{mode}' using KullbackLeibler"
    )

    # Validate coordinator has required full-data acquisition models
    if coordinator is not None:
        # All coordinators must have linear_acquisition_model (enforced by base class)
        if coordinator.linear_acquisition_model is None:
            raise ValueError(
                "Coordinator.linear_acquisition_model must be set before partitioning. "
                "Pass a full-data linear acquisition model to coordinator.__init__"
            )
        # SimindCoordinator mode_both also requires stir_acquisition_model
        if (
            hasattr(coordinator, "mode_both")
            and coordinator.mode_both
            and hasattr(coordinator, "stir_acquisition_model")
            and coordinator.stir_acquisition_model is None
        ):
            raise ValueError(
                "SimindCoordinator mode_both requires stir_acquisition_model. "
                "Pass a full-data STIR model with additive to coordinator.__init__"
            )

    # Use SIRF partitioner to get subset data and acquisition models
    # Pass initial_image so partitioner sets up the models properly
    # (CoordinatedSubsetProjector.set_up handles re-setup gracefully via try-except)
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
    subset_sensitivity_max = []
    subset_eta_min = []

    ones_image = initial_image.get_uniform_copy(1)

    for i, (prompts_subset, acq_model, subset_indices) in enumerate(
        zip(prompts_subsets, acquisition_models, partition_indices)
    ):
        # IMPORTANT: For CIL LinearOperator compatibility, extract additive term and linear model
        # The linear model has no additive term (required for CIL)
        additive_subset = acq_model.get_additive_term()
        linear_acq_model = acq_model.get_linear_acquisition_model()

        # Wrap LINEAR acquisition model with CoordinatedSubsetProjector if coordinator provided
        if coordinator is not None:
            projector = CoordinatedSubsetProjector(
                stir_projector=linear_acq_model,
                coordinator=coordinator,
                subset_indices=subset_indices,
            )
            # Set up projector
            projector.set_up(prompts_subset, initial_image)
        else:
            # No coordinator - wrap STIR linear acquisition model for CIL
            projector = CILAcquisitionModelAdapter(linear_acq_model)

        projectors.append(projector)

        # Create CIL KL function with eta floored for numerical stability
        if eta_floor > 0.0:
            eta_subset = additive_subset.maximum(eta_floor)
        else:
            eta_subset = additive_subset + additive_subset.get_uniform_copy(0)

        # Store baseline eta minimum for positivity capping
        eta_min = float(get_array(eta_subset).min())
        subset_eta_min.append(eta_min)

        # Estimate subset sensitivity maximum s_max = max(A_i 1)
        try:
            sensitivity_subset = linear_acq_model.forward(ones_image)
        except Exception:
            sensitivity_subset = linear_acq_model.forward(
                ones_image, subset_num=i, num_subsets=num_subsets
            )
        s_max = float(get_array(sensitivity_subset).max())
        subset_sensitivity_max.append(s_max)

        kl_data = KullbackLeibler(
            b=prompts_subset,
            eta=eta_subset,
        )

        # Store unwrapped KL function for eta updates
        kl_data_functions.append(kl_data)

        # Compose with LINEAR projector: (KL ∘ A)(x) = KL(A(x), eta)
        kl_objective = OperatorCompositionFunction(kl_data, projector)

        kl_objectives.append(kl_objective)

        logging.info(
            f"Subset {i}: {len(subset_indices)} views, "
            f"data sum={prompts_subset.sum():.2e}, "
            f"additive sum={eta_subset.sum():.2e}"
        )

    logging.info("Created CIL KL objectives with LINEAR projectors (standard KL)")

    # Return both composed objectives and unwrapped KL data functions
    return (
        kl_objectives,
        projectors,
        partition_indices,
        kl_data_functions,
        subset_sensitivity_max,
        subset_eta_min,
    )


def create_svrg_objective_with_rdp(
    kl_objectives,
    rdp_prior,
    sampler,
    snapshot_update_interval=None,
    initial_image=None,
    *,
    algorithm="SVRG",
):
    """
    Create a stochastic objective (SVRG or SAGA) with optional RDP prior.

    Args:
        kl_objectives (list): CIL data fidelity functions (one per subset).
        rdp_prior: SETR RelativeDifferencePrior (or compatible Function) or False/None.
        sampler: CIL Sampler instance controlling subset selection.
        snapshot_update_interval (int, optional): Interval for SVRG snapshots. Ignored for SAGA.
        initial_image: Initial image for warm-starting SAGA gradients.
        algorithm (str): Name of stochastic algorithm to use ("SVRG" or "SAGA").

    Returns:
        CIL Function: Combined stochastic objective (+ RDP prior when supplied).
    """
    from cil.optimisation.functions import SAGAFunction, SumFunction, SVRGFunction

    if not kl_objectives:
        raise ValueError("kl_objectives must contain at least one function")

    algo_name = (algorithm or "SVRG").upper()

    if algo_name == "SVRG":
        if snapshot_update_interval is None:
            raise ValueError(
                "snapshot_update_interval must be provided when using SVRG"
            )
        stochastic_func = SVRGFunction(
            kl_objectives,
            sampler,
            snapshot_update_interval=snapshot_update_interval,
        )
    elif algo_name == "SAGA":
        stochastic_func = SAGAFunction(kl_objectives, sampler)
        if initial_image is not None:
            stochastic_func.warm_start_approximate_gradients(initial_image)
    else:
        raise ValueError(f"Unsupported stochastic algorithm '{algorithm}'")

    total_objective = (
        SumFunction(stochastic_func, rdp_prior) if rdp_prior else stochastic_func
    )
    logging.info(
        "Created %s objective with %d subsets%s",
        algo_name,
        len(kl_objectives),
        " + RDP prior" if rdp_prior else "",
    )

    return total_objective


def create_full_objective_with_rdp(kl_objectives, prior):
    """
    Create a deterministic (full-gradient) objective with optional RDP prior.

    This utility mirrors :func:`create_svrg_objective_with_rdp` but directly sums
    all KL objectives, yielding a function suitable for ISTA warm-up passes
    prior to stochastic optimisation. The ``prior`` argument may be any CIL
    Function compatible with :class:`cil.optimisation.functions.SumFunction`
    (e.g. a scaled RDP prior) or falsy to disable penalisation.
    """
    if not kl_objectives:
        raise ValueError("kl_objectives must contain at least one function")

    from cil.optimisation.functions import SumFunction

    combined = kl_objectives[0]
    for func in kl_objectives[1:]:
        combined = SumFunction(combined, func)

    if prior:
        combined = SumFunction(combined, prior)

    logging.info(
        "Created full-gradient objective with %d subsets%s",
        len(kl_objectives),
        " + prior" if prior else "",
    )

    return combined

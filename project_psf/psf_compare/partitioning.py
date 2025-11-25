"""
Data partitioning helpers for CIL-based objectives.
"""

import logging


def partition_data_once_cil(
    acquisition_data,
    additive_data,
    acq_model_func,
    initial_image,
    num_subsets,
    coordinator=None,
    eta_floor=1e-5,
    count_floor=1e-8,
    attenuation_map=None,
):
    """
    Partition data ONCE and return CIL objective functions.

    Uses CIL KL objectives composed with LINEAR acquisition models (optionally wrapped
    with SimindSubsetProjector for Monte Carlo corrections).

    IMPORTANT: Returns LINEAR acquisition models and KL functions with eta parameter
    for proper CIL LinearOperator compatibility.
    """
    from sirf_simind_connection.utils.cil_partitioner import (
        partition_data_with_cil_objectives,
    )

    logging.info("Partitioning data into %d subsets (CIL mode)...", num_subsets)

    normalisation = acquisition_data.get_uniform_copy(1)

    (
        kl_objectives,
        projectors,
        partition_indices,
        kl_data_functions,
        subset_sensitivity_max,
        subset_eta_min,
    ) = partition_data_with_cil_objectives(
        acquisition_data,
        additive_data,
        normalisation,
        num_subsets,
        initial_image,
        acq_model_func,
        coordinator=coordinator,
        mode="staggered",
        eta_floor=eta_floor,
        count_floor=count_floor,
        attenuation_map=attenuation_map,
    )

    logging.info("Created %d CIL KL objectives with LINEAR models", len(kl_objectives))

    return (
        kl_objectives,
        projectors,
        kl_data_functions,
        partition_indices,
        subset_sensitivity_max,
        subset_eta_min,
    )

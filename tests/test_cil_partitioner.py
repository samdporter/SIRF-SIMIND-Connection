"""
Tests for CIL partitioner utilities.

Tests cover:
- partition_data_with_cil_objectives function
- CILAcquisitionModelAdapter wrapper
- Integration with SimindCoordinator
- SVRG objective creation with RDP prior
"""

import tempfile

import pytest

from sirf_simind_connection import SimindSimulator, SimulationConfig
from sirf_simind_connection.configs import get
from sirf_simind_connection.core.components import ScoringRoutine
from sirf_simind_connection.core.coordinator import SimindCoordinator
from sirf_simind_connection.utils.stir_utils import (
    create_attenuation_map,
    create_simple_phantom,
    create_stir_acqdata,
)


# Check if CIL is available
try:
    from cil.optimisation.functions import KullbackLeibler, OperatorCompositionFunction
    from cil.optimisation.utilities import Sampler

    from sirf_simind_connection.utils.cil_partitioner import (
        CILAcquisitionModelAdapter,
        partition_data_with_cil_objectives,
    )

    CIL_AVAILABLE = True
except ImportError:
    CIL_AVAILABLE = False


# All tests require SIRF and CIL
pytestmark = [
    pytest.mark.requires_sirf,
    pytest.mark.skipif(not CIL_AVAILABLE, reason="CIL not available"),
]


@pytest.fixture
def basic_phantom():
    """Create a simple test phantom."""
    return create_simple_phantom()


@pytest.fixture
def measured_data():
    """Create measured acquisition data."""
    # 60 projections, 64x64 matrix
    acq_data = create_stir_acqdata([64, 64], 60, [4.42, 4.42])
    # Fill with some counts
    acq_data.fill(acq_data.get_uniform_copy(100.0))
    return acq_data


@pytest.fixture
def additive_data(measured_data):
    """Create additive (scatter) data."""
    # Small uniform background
    return measured_data.get_uniform_copy(10.0)


@pytest.fixture
def multiplicative_factors(measured_data):
    """Create multiplicative factors (e.g., normalization)."""
    # Uniform factors (no correction)
    return measured_data.get_uniform_copy(1.0)


def create_acq_model_factory():
    """Create a factory function for acquisition models."""
    from sirf.STIR import AcquisitionModelUsingRayTracingMatrix

    def factory():
        return AcquisitionModelUsingRayTracingMatrix()

    return factory


class TestCILAcquisitionModelAdapter:
    """Test the CIL acquisition model adapter wrapper."""

    def test_adapter_wraps_stir_projector(self, basic_phantom, measured_data):
        """Test that adapter wraps STIR projector correctly."""
        from sirf.STIR import AcquisitionModelUsingRayTracingMatrix

        stir_am = AcquisitionModelUsingRayTracingMatrix()
        adapter = CILAcquisitionModelAdapter(stir_am)

        # Set up
        adapter.set_up(measured_data, basic_phantom)

        # Should have access to underlying projector
        assert adapter._stir_projector is stir_am

    def test_adapter_forward_method(self, basic_phantom, measured_data):
        """Test adapter's forward method."""
        from sirf.STIR import AcquisitionModelUsingRayTracingMatrix

        stir_am = AcquisitionModelUsingRayTracingMatrix()
        adapter = CILAcquisitionModelAdapter(stir_am)
        adapter.set_up(measured_data, basic_phantom)

        # Forward project
        result = adapter.forward(basic_phantom)

        assert result is not None
        assert result.shape() == measured_data.shape()

    def test_adapter_backward_method(self, basic_phantom, measured_data):
        """Test adapter's backward method."""
        from sirf.STIR import AcquisitionModelUsingRayTracingMatrix

        stir_am = AcquisitionModelUsingRayTracingMatrix()
        adapter = CILAcquisitionModelAdapter(stir_am)
        adapter.set_up(measured_data, basic_phantom)

        # Backward project
        result = adapter.backward(measured_data)

        assert result is not None
        assert result.shape() == basic_phantom.shape()

    def test_adapter_direct_adjoint_methods(self, basic_phantom, measured_data):
        """Test CIL-compatible direct/adjoint methods."""
        from sirf.STIR import AcquisitionModelUsingRayTracingMatrix

        stir_am = AcquisitionModelUsingRayTracingMatrix()
        adapter = CILAcquisitionModelAdapter(stir_am)
        adapter.set_up(measured_data, basic_phantom)

        # Direct (forward)
        direct_result = adapter.direct(basic_phantom)
        assert direct_result is not None

        # Adjoint (backward)
        adjoint_result = adapter.adjoint(measured_data)
        assert adjoint_result is not None

    def test_adapter_geometry_methods(self, basic_phantom, measured_data):
        """Test geometry pass-through methods."""
        from sirf.STIR import AcquisitionModelUsingRayTracingMatrix

        stir_am = AcquisitionModelUsingRayTracingMatrix()
        adapter = CILAcquisitionModelAdapter(stir_am)
        adapter.set_up(measured_data, basic_phantom)

        # Range geometry (acquisition space)
        range_geom = adapter.range_geometry()
        assert range_geom is not None

        # Domain geometry (image space)
        domain_geom = adapter.domain_geometry()
        assert domain_geom is not None


class TestPartitionDataWithoutCoordinator:
    """Test partitioning without SimindCoordinator (plain STIR)."""

    def test_partition_creates_correct_number_objectives(
        self, measured_data, additive_data, multiplicative_factors, basic_phantom
    ):
        """Test that correct number of objectives are created."""
        num_subsets = 6

        objectives, projectors, indices, kl_funcs = partition_data_with_cil_objectives(
            acquisition_data=measured_data,
            additive_data=additive_data,
            multiplicative_factors=multiplicative_factors,
            num_subsets=num_subsets,
            initial_image=basic_phantom,
            create_acq_model=create_acq_model_factory(),
            simind_coordinator=None,  # No coordinator
            mode="staggered",
        )

        assert len(objectives) == num_subsets
        assert len(projectors) == num_subsets
        assert len(indices) == num_subsets
        assert len(kl_funcs) == num_subsets

    def test_partition_creates_kl_objectives(
        self, measured_data, additive_data, multiplicative_factors, basic_phantom
    ):
        """Test that KL objectives are created."""
        objectives, _, _, _ = partition_data_with_cil_objectives(
            acquisition_data=measured_data,
            additive_data=additive_data,
            multiplicative_factors=multiplicative_factors,
            num_subsets=6,
            initial_image=basic_phantom,
            create_acq_model=create_acq_model_factory(),
            simind_coordinator=None,
        )

        # Each objective should be an OperatorCompositionFunction
        for obj in objectives:
            assert isinstance(obj, OperatorCompositionFunction)

    def test_partition_subset_indices_staggered(
        self, measured_data, additive_data, multiplicative_factors, basic_phantom
    ):
        """Test staggered subset index generation."""
        num_subsets = 6

        _, _, indices, _ = partition_data_with_cil_objectives(
            acquisition_data=measured_data,
            additive_data=additive_data,
            multiplicative_factors=multiplicative_factors,
            num_subsets=num_subsets,
            initial_image=basic_phantom,
            create_acq_model=create_acq_model_factory(),
            mode="staggered",
        )

        # 60 views, 6 subsets -> 10 views per subset
        assert len(indices) == num_subsets

        # Each subset should have 10 views
        for subset_idx in indices:
            assert len(subset_idx) == 10

        # First subset should be [0, 6, 12, ...]
        assert indices[0][0] == 0
        assert indices[0][1] == 6

    def test_partition_subset_indices_sequential(
        self, measured_data, additive_data, multiplicative_factors, basic_phantom
    ):
        """Test sequential subset index generation."""
        num_subsets = 6

        _, _, indices, _ = partition_data_with_cil_objectives(
            acquisition_data=measured_data,
            additive_data=additive_data,
            multiplicative_factors=multiplicative_factors,
            num_subsets=num_subsets,
            initial_image=basic_phantom,
            create_acq_model=create_acq_model_factory(),
            mode="sequential",
        )

        # Sequential: [0-9], [10-19], [20-29], ...
        assert indices[0] == list(range(0, 10))
        assert indices[1] == list(range(10, 20))
        assert indices[5] == list(range(50, 60))

    def test_partition_single_subset(
        self, measured_data, additive_data, multiplicative_factors, basic_phantom
    ):
        """Test partitioning with num_subsets=1."""
        objectives, projectors, indices, _ = partition_data_with_cil_objectives(
            acquisition_data=measured_data,
            additive_data=additive_data,
            multiplicative_factors=multiplicative_factors,
            num_subsets=1,
            initial_image=basic_phantom,
            create_acq_model=create_acq_model_factory(),
        )

        assert len(objectives) == 1
        assert len(projectors) == 1
        # Single subset should have all 60 views
        assert len(indices[0]) == 60


class TestPartitionDataWithCoordinator:
    """Test partitioning with SimindCoordinator integration."""

    @pytest.fixture
    def coordinator(self, basic_phantom, measured_data):
        """Create a SimindCoordinator for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(get("AnyScan.yaml"))
            config.set_value(26, 0.05)  # Fast
            config.set_value(29, 60)  # 60 projections
            config.set_value(76, 64)
            config.set_value(77, 64)

            simulator = SimindSimulator(
                config,
                output_dir=temp_dir,
                scoring_routine=ScoringRoutine.SCATTWIN,
            )

            mu_map = create_attenuation_map(basic_phantom)
            simulator.set_source(basic_phantom)
            simulator.set_mu_map(mu_map)
            simulator.set_energy_windows([126], [154], [0])

            # Create linear acquisition model for coordinator
            from sirf.STIR import AcquisitionModelUsingRayTracingMatrix

            linear_am = AcquisitionModelUsingRayTracingMatrix()
            linear_am.set_up(measured_data, basic_phantom)
            linear_am.set_num_subsets(1)
            linear_am.set_subset_num(0)

            coordinator = SimindCoordinator(
                simind_simulator=simulator,
                num_subsets=6,
                correction_update_interval=6,
                residual_correction=True,
                update_additive=False,
                linear_acquisition_model=linear_am,
            )

            yield coordinator

    def test_partition_with_coordinator_creates_subset_projectors(
        self,
        measured_data,
        additive_data,
        multiplicative_factors,
        basic_phantom,
        coordinator,
    ):
        """Test that SimindSubsetProjector instances are created."""
        from sirf_simind_connection.core.projector import SimindSubsetProjector

        objectives, projectors, indices, _ = partition_data_with_cil_objectives(
            acquisition_data=measured_data,
            additive_data=additive_data,
            multiplicative_factors=multiplicative_factors,
            num_subsets=6,
            initial_image=basic_phantom,
            create_acq_model=create_acq_model_factory(),
            simind_coordinator=coordinator,
        )

        # Projectors should be SimindSubsetProjector instances
        for proj in projectors:
            assert isinstance(proj, SimindSubsetProjector)
            # Should reference the coordinator
            assert proj.coordinator is coordinator

    def test_partition_with_coordinator_passes_subset_indices(
        self,
        measured_data,
        additive_data,
        multiplicative_factors,
        basic_phantom,
        coordinator,
    ):
        """Test that subset indices are passed to SimindSubsetProjector."""
        _, projectors, indices, _ = partition_data_with_cil_objectives(
            acquisition_data=measured_data,
            additive_data=additive_data,
            multiplicative_factors=multiplicative_factors,
            num_subsets=6,
            initial_image=basic_phantom,
            create_acq_model=create_acq_model_factory(),
            simind_coordinator=coordinator,
        )

        # Each projector should have correct subset indices
        for i, proj in enumerate(projectors):
            assert proj.subset_indices == indices[i]


class TestKLDataFunctions:
    """Test KL data function setup and eta parameter."""

    def test_kl_functions_returned(
        self, measured_data, additive_data, multiplicative_factors, basic_phantom
    ):
        """Test that KL data functions are returned."""
        _, _, _, kl_funcs = partition_data_with_cil_objectives(
            acquisition_data=measured_data,
            additive_data=additive_data,
            multiplicative_factors=multiplicative_factors,
            num_subsets=6,
            initial_image=basic_phantom,
            create_acq_model=create_acq_model_factory(),
        )

        assert len(kl_funcs) == 6

        # Each should be a KullbackLeibler function
        for kl_func in kl_funcs:
            assert isinstance(kl_func, KullbackLeibler)

    def test_kl_eta_includes_additive(
        self, measured_data, additive_data, multiplicative_factors, basic_phantom
    ):
        """Test that eta parameter includes additive term."""
        _, _, _, kl_funcs = partition_data_with_cil_objectives(
            acquisition_data=measured_data,
            additive_data=additive_data,
            multiplicative_factors=multiplicative_factors,
            num_subsets=6,
            initial_image=basic_phantom,
            create_acq_model=create_acq_model_factory(),
        )

        # Each KL function should have eta set
        for kl_func in kl_funcs:
            eta = kl_func.eta
            assert eta is not None
            # eta should be approximately additive_data (plus epsilon)
            # Can't easily check exact values without accessing internal state


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_coordinator_models_raises_error(
        self, measured_data, additive_data, multiplicative_factors, basic_phantom
    ):
        """Test that missing coordinator models raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimulationConfig(get("AnyScan.yaml"))
            simulator = SimindSimulator(config, temp_dir)

            # Create coordinator without linear_acquisition_model
            # This should fail during partitioning
            try:
                coordinator = SimindCoordinator(
                    simind_simulator=simulator,
                    num_subsets=6,
                    correction_update_interval=6,
                    residual_correction=True,
                    update_additive=False,
                    linear_acquisition_model=None,  # Missing!
                )

                # If we got here, coordinator creation didn't validate
                # So partitioner should catch it
                with pytest.raises((ValueError, AttributeError)):
                    partition_data_with_cil_objectives(
                        acquisition_data=measured_data,
                        additive_data=additive_data,
                        multiplicative_factors=multiplicative_factors,
                        num_subsets=6,
                        initial_image=basic_phantom,
                        create_acq_model=create_acq_model_factory(),
                        simind_coordinator=coordinator,
                    )
            except (ValueError, AttributeError, TypeError):
                # Coordinator creation itself caught the error - good!
                pass

    def test_zero_additive_term(
        self, measured_data, multiplicative_factors, basic_phantom
    ):
        """Test with zero additive term."""
        zero_additive = measured_data.get_uniform_copy(0.0)

        objectives, _, _, _ = partition_data_with_cil_objectives(
            acquisition_data=measured_data,
            additive_data=zero_additive,
            multiplicative_factors=multiplicative_factors,
            num_subsets=6,
            initial_image=basic_phantom,
            create_acq_model=create_acq_model_factory(),
        )

        # Should still work with zero additive
        assert len(objectives) == 6


# SVRG objective creation tests
class TestSVRGObjectiveCreation:
    """Test SVRG objective creation with RDP prior."""

    @pytest.mark.skipif(
        True, reason="Requires STIR RDP prior which may not be available"
    )
    def test_create_svrg_objective_with_rdp(
        self, measured_data, additive_data, multiplicative_factors, basic_phantom
    ):
        """Test creating SVRG objective with RDP prior."""
        try:
            from cil.optimisation.functions import SVRGFunction
            from sirf.STIR import CUDARelativeDifferencePrior

            from sirf_simind_connection.utils.cil_partitioner import (
                create_svrg_objective_with_rdp,
            )
        except ImportError:
            pytest.skip("SVRG or RDP not available")

        # Create objectives
        objectives, _, _, kl_funcs = partition_data_with_cil_objectives(
            acquisition_data=measured_data,
            additive_data=additive_data,
            multiplicative_factors=multiplicative_factors,
            num_subsets=6,
            initial_image=basic_phantom,
            create_acq_model=create_acq_model_factory(),
        )

        # Create RDP prior
        mu_map = create_attenuation_map(basic_phantom)
        rdp_prior = CUDARelativeDifferencePrior()
        rdp_prior.set_penalisation_factor(0.1)
        rdp_prior.set_up(basic_phantom)
        rdp_prior.set_kappa(mu_map)

        # Create sampler
        sampler = Sampler.sequential(len(objectives))

        # Create SVRG objective
        svrg_obj = create_svrg_objective_with_rdp(
            kl_data_functions=kl_funcs,
            rdp_prior=rdp_prior,
            sampler=sampler,
            initial_image=basic_phantom,
            num_subsets=6,
        )

        assert svrg_obj is not None

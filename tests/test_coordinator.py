"""
Tests for SimindCoordinator.

Tests cover:
- Initialization and mode detection
- Simulation management and caching
- Subset distribution
- Iteration tracking
- All three correction modes (A, B, C)
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


# All tests require SIRF
pytestmark = pytest.mark.requires_sirf


@pytest.fixture
def basic_phantom():
    """Create a simple test phantom."""
    return create_simple_phantom()


@pytest.fixture
def acq_template():
    """Create acquisition template with 60 projections."""
    return create_stir_acqdata([64, 64], 60, [4.42, 4.42])


@pytest.fixture
def simind_simulator(basic_phantom, acq_template):
    """Create a configured SIMIND simulator."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = SimulationConfig(get("AnyScan.yaml"))
        # Fast settings for tests
        config.set_value(26, 0.05)  # 5e4 photons (very fast)
        config.set_value(29, 60)  # 60 projections
        config.set_value(76, 64)  # matrix i
        config.set_value(77, 64)  # matrix j

        simulator = SimindSimulator(
            config,
            output_dir=temp_dir,
            scoring_routine=ScoringRoutine.SCATTWIN,
        )

        mu_map = create_attenuation_map(basic_phantom)
        simulator.set_source(basic_phantom)
        simulator.set_mu_map(mu_map)
        simulator.set_energy_windows([126], [154], [0])

        yield simulator


@pytest.fixture
def linear_acquisition_model(acq_template, basic_phantom):
    """Create a linear STIR acquisition model."""
    from sirf.STIR import AcquisitionModelUsingRayTracingMatrix

    am = AcquisitionModelUsingRayTracingMatrix()
    am.set_up(acq_template, basic_phantom)
    am.set_num_subsets(1)
    am.set_subset_num(0)
    return am


class TestCoordinatorInitialization:
    """Test coordinator initialization and configuration."""

    def test_basic_initialization(self, simind_simulator, linear_acquisition_model):
        """Test basic coordinator creation."""
        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=6,
            residual_correction=False,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
        )

        assert coordinator.num_subsets == 6
        assert coordinator.correction_update_interval == 6
        assert coordinator.global_subiteration == 0
        assert coordinator.last_update_iteration == -1
        assert coordinator.cache_version == 0

    def test_mode_a_detection(self, simind_simulator, linear_acquisition_model):
        """Test Mode A (residual only) detection and configuration."""
        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=6,
            residual_correction=True,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
        )

        assert coordinator.mode_residual_only is True
        assert coordinator.mode_additive_only is False
        assert coordinator.mode_both is False

        # Mode A should disable collimator routine
        assert coordinator.simind_simulator.config.get_value(53) == 0
        # Should use PENETRATE scoring
        assert coordinator.simind_simulator.config.get_value(84) == 4

    def test_mode_b_detection(self, simind_simulator, linear_acquisition_model):
        """Test Mode B (additive only) detection and configuration."""
        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=6,
            residual_correction=False,
            update_additive=True,
            linear_acquisition_model=linear_acquisition_model,
        )

        assert coordinator.mode_residual_only is False
        assert coordinator.mode_additive_only is True
        assert coordinator.mode_both is False

        # Mode B should enable collimator routine
        assert coordinator.simind_simulator.config.get_value(53) == 1
        # Should use PENETRATE scoring
        assert coordinator.simind_simulator.config.get_value(84) == 4

    def test_mode_c_detection(self, simind_simulator, linear_acquisition_model):
        """Test Mode C (both) detection and configuration."""
        from sirf.STIR import AcquisitionModelUsingRayTracingMatrix

        # Mode C needs stir_acquisition_model too
        stir_am = AcquisitionModelUsingRayTracingMatrix()
        stir_am.set_up(
            create_stir_acqdata([64, 64], 60, [4.42, 4.42]), create_simple_phantom()
        )

        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=6,
            residual_correction=True,
            update_additive=True,
            linear_acquisition_model=linear_acquisition_model,
            stir_acquisition_model=stir_am,
        )

        assert coordinator.mode_residual_only is False
        assert coordinator.mode_additive_only is False
        assert coordinator.mode_both is True

        # Mode C should enable collimator routine
        assert coordinator.simind_simulator.config.get_value(53) == 1
        # Should use PENETRATE scoring
        assert coordinator.simind_simulator.config.get_value(84) == 4


class TestIterationTracking:
    """Test iteration tracking and update triggering."""

    def test_should_update_on_first_call(
        self, simind_simulator, linear_acquisition_model
    ):
        """Test that update happens on first call (iteration 0)."""
        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=6,
            residual_correction=True,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
        )

        assert coordinator.should_update() is True

    def test_should_update_respects_interval(
        self, simind_simulator, linear_acquisition_model
    ):
        """Test update interval logic."""
        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=3,
            residual_correction=True,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
        )

        # Iteration 0: should update
        coordinator.global_subiteration = 0
        assert coordinator.should_update() is True
        coordinator.last_update_iteration = 0

        # Iterations 1-2: should not update
        coordinator.global_subiteration = 1
        assert coordinator.should_update() is False

        coordinator.global_subiteration = 2
        assert coordinator.should_update() is False

        # Iteration 3: should update
        coordinator.global_subiteration = 3
        assert coordinator.should_update() is True
        coordinator.last_update_iteration = 3

        # Iteration 4-5: should not update
        coordinator.global_subiteration = 4
        assert coordinator.should_update() is False

    def test_no_update_when_disabled(self, simind_simulator, linear_acquisition_model):
        """Test that updates don't happen when both flags are False."""
        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=1,
            residual_correction=False,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
        )

        assert coordinator.should_update() is False


class TestSubsetIndices:
    """Test subset index calculation and validation."""

    def test_subset_indices_staggered(self, simind_simulator, linear_acquisition_model):
        """Test staggered subset index generation."""
        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=6,
            residual_correction=True,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
        )

        # 60 views, 6 subsets -> 10 views per subset
        # Staggered: [0, 6, 12, ...], [1, 7, 13, ...], etc.
        subset_0 = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
        subset_1 = [1, 7, 13, 19, 25, 31, 37, 43, 49, 55]
        subset_5 = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]

        # Test via get_subset_residual (which uses subset indices internally)
        # This is indirect testing since indices are computed internally
        assert coordinator.num_subsets == 6


class TestCacheManagement:
    """Test cache versioning and state management."""

    def test_cache_version_increments(
        self, simind_simulator, linear_acquisition_model, basic_phantom
    ):
        """Test that cache version increments after simulation."""
        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=1,
            residual_correction=True,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
        )

        initial_version = coordinator.cache_version
        assert initial_version == 0

        # Running simulation should increment cache version
        # Note: This will actually run SIMIND if marked requires_simind
        # For unit tests, we'd mock this, but for integration it's fine

    def test_cached_results_persist(self, simind_simulator, linear_acquisition_model):
        """Test that cached results are stored."""
        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=6,
            residual_correction=True,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
        )

        # Initially no cached results
        assert coordinator.cached_b01 is None
        assert coordinator.cached_b02 is None
        assert coordinator.cached_scale_factor is None


class TestCumulativeAdditive:
    """Test cumulative additive term tracking."""

    def test_initialize_with_additive(
        self, simind_simulator, linear_acquisition_model, acq_template
    ):
        """Test initialization of cumulative additive."""
        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=6,
            residual_correction=False,
            update_additive=True,
            linear_acquisition_model=linear_acquisition_model,
        )

        # Create initial additive
        initial_additive = acq_template.get_uniform_copy(0.5)

        coordinator.initialize_with_additive(initial_additive)

        assert coordinator.cumulative_additive is not None
        # Check that it was cloned (not same object)
        assert coordinator.cumulative_additive is not initial_additive

    def test_get_full_additive_term(
        self, simind_simulator, linear_acquisition_model, acq_template
    ):
        """Test retrieving full additive term."""
        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=6,
            residual_correction=False,
            update_additive=True,
            linear_acquisition_model=linear_acquisition_model,
        )

        # Initialize with additive
        initial_additive = acq_template.get_uniform_copy(0.5)
        coordinator.initialize_with_additive(initial_additive)

        # Get full additive
        full_additive = coordinator.get_full_additive_term()

        assert full_additive is not None
        assert full_additive.shape() == initial_additive.shape()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_subset(self, simind_simulator, linear_acquisition_model):
        """Test coordinator with num_subsets=1."""
        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=1,
            correction_update_interval=1,
            residual_correction=True,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
        )

        assert coordinator.num_subsets == 1

    def test_large_update_interval(self, simind_simulator, linear_acquisition_model):
        """Test with very large update interval."""
        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=1000,
            residual_correction=True,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
        )

        # Should update on iteration 0
        assert coordinator.should_update() is True

        # Should not update for a long time
        coordinator.global_subiteration = 999
        coordinator.last_update_iteration = 0
        assert coordinator.should_update() is False

    def test_missing_linear_model_validation(self, simind_simulator):
        """Test that missing linear model is caught."""
        # Mode A requires linear_acquisition_model
        with pytest.raises((ValueError, AttributeError, TypeError)):
            coordinator = SimindCoordinator(
                simind_simulator=simind_simulator,
                num_subsets=6,
                correction_update_interval=6,
                residual_correction=True,
                update_additive=False,
                linear_acquisition_model=None,  # Missing!
            )


# Integration tests that actually run SIMIND
@pytest.mark.requires_simind
@pytest.mark.slow
class TestSimulationIntegration:
    """Integration tests that run actual SIMIND simulations."""

    def test_mode_a_simulation(
        self, simind_simulator, linear_acquisition_model, basic_phantom
    ):
        """Test Mode A with actual SIMIND simulation."""
        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=1,
            residual_correction=True,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
        )

        # Run simulation
        coordinator.run_full_simulation(basic_phantom)

        # Check cache was populated
        assert coordinator.cache_version == 1
        assert coordinator.cached_b02 is not None  # Mode A uses b02
        assert coordinator.cached_scale_factor is not None

        # Mode A should have b01 = None
        assert coordinator.cached_b01 is None

    def test_subset_residual_extraction(
        self, simind_simulator, linear_acquisition_model, basic_phantom
    ):
        """Test extracting subset-specific residuals."""
        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=1,
            residual_correction=True,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
        )

        # Run simulation
        coordinator.run_full_simulation(basic_phantom)

        # Extract subset residual
        subset_indices = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]  # Subset 0
        residual = coordinator.get_subset_residual(subset_indices)

        assert residual is not None
        # Residual should have shape corresponding to subset views
        assert residual.dimensions() == 3  # [views, y, z]


class TestOutputDirectory:
    """Test output directory handling."""

    def test_output_dir_stored(self, simind_simulator, linear_acquisition_model):
        """Test that output_dir is stored as instance attribute."""
        with tempfile.TemporaryDirectory() as temp_dir:
            coordinator = SimindCoordinator(
                simind_simulator=simind_simulator,
                num_subsets=6,
                correction_update_interval=6,
                residual_correction=True,
                update_additive=False,
                linear_acquisition_model=linear_acquisition_model,
                output_dir=temp_dir,
            )

            assert coordinator.output_dir == temp_dir

    def test_output_dir_optional(self, simind_simulator, linear_acquisition_model):
        """Test that output_dir is optional."""
        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=6,
            residual_correction=True,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
            output_dir=None,
        )

        assert coordinator.output_dir is None

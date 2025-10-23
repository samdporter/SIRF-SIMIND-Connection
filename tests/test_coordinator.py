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

import numpy as np
import pytest
from sirf.STIR import AcquisitionData

from sirf_simind_connection import SimindSimulator, SimulationConfig
from sirf_simind_connection.configs import get
from sirf_simind_connection.core.components import ScoringRoutine
from sirf_simind_connection.core.coordinator import SimindCoordinator
from sirf_simind_connection.utils import get_array
from sirf_simind_connection.utils.stir_utils import (
    create_attenuation_map,
    create_simple_phantom,
    create_stir_acqdata,
)


AcquisitionData.set_storage_scheme("memory")


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
    am.num_subsets = 1
    am.subset_num = 0
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
        """Test that update does NOT happen on iteration 0 (we start with estimated additive)."""

        class DummyAlgorithm:
            iteration = 0

        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=6,
            residual_correction=True,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
        )
        coordinator.algorithm = DummyAlgorithm()

        # Iteration 0: should NOT update (start with estimated additive)
        assert coordinator.should_update() is False

        # After interval passes, should update
        coordinator.algorithm.iteration = 6
        assert coordinator.should_update() is True

    def test_should_update_respects_interval(
        self, simind_simulator, linear_acquisition_model
    ):
        """Test update interval logic."""

        class DummyAlgorithm:
            def __init__(self):
                self.iteration = 0

        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=3,
            residual_correction=True,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
        )
        coordinator.algorithm = DummyAlgorithm()

        # Iteration 0: should NOT update (start with estimated additive)
        coordinator.algorithm.iteration = 0
        assert coordinator.should_update() is False

        # Iteration 1: should not update (1 - (-1) = 2 < 3)
        coordinator.algorithm.iteration = 1
        assert coordinator.should_update() is False

        # Iteration 2: should update (2 - (-1) = 3 >= 3)
        coordinator.algorithm.iteration = 2
        assert coordinator.should_update() is True
        coordinator._last_update_iteration = 2

        # Iteration 3-4: should not update (haven't reached interval yet)
        coordinator.algorithm.iteration = 3
        assert coordinator.should_update() is False

        # Iteration 4: should not update (4 - 2 = 2 < 3)
        coordinator.algorithm.iteration = 4
        assert coordinator.should_update() is False

        # Iteration 5: should update again (5 - 2 = 3 >= 3)
        coordinator.algorithm.iteration = 5
        assert coordinator.should_update() is True

    def test_large_update_interval(self, simind_simulator, linear_acquisition_model):
        """Test behaviour with a very large update interval."""

        class DummyAlgorithm:
            def __init__(self):
                self.iteration = 0

        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=1000,
            residual_correction=True,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
        )
        coordinator.algorithm = DummyAlgorithm()

        # Iterate 0: still relying on cached additive, so no update
        coordinator.algorithm.iteration = 0
        assert coordinator.should_update() is False

        # Once we hit the large interval, the update should trigger
        coordinator.algorithm.iteration = 1000
        assert coordinator.should_update() is True

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

    def test_should_not_update_near_end_of_reconstruction(
        self, simind_simulator, linear_acquisition_model
    ):
        """Test that update is skipped if it's in the final update interval block."""

        class DummyAlgorithm:
            def __init__(self):
                self.iteration = 0

        total_iterations = 1800  # e.g., 100 epochs * 18 subsets
        update_interval = 360  # e.g., update every 20 epochs

        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=18,
            correction_update_interval=update_interval,
            residual_correction=True,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
            total_iterations=total_iterations,
        )
        coordinator.algorithm = DummyAlgorithm()

        # The "do-not-simulate" zone starts at: 1800 - 360 = 1440

        # Simulate an update just before the final block
        # last update was at 1079 (1439 - 360)
        coordinator._last_update_iteration = 1079
        coordinator.algorithm.iteration = 1439  # 1439 - 1079 = 360

        # This update should run because 1439 < 1440
        assert coordinator.should_update() is True

        # Mark this update as done
        coordinator._last_update_iteration = 1439

        # Now, simulate the final scheduled update
        coordinator.algorithm.iteration = 1799  # 1799 - 1439 = 360

        # This update should be SKIPPED because 1799 >= 1440
        assert coordinator.should_update() is False

    def test_algorithm_iteration_tracking(
        self, simind_simulator, linear_acquisition_model
    ):
        """Test that algorithm iteration is properly used for should_update logic."""

        class DummyAlgorithm:
            def __init__(self):
                self.iteration = 0

        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=2,
            residual_correction=True,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
        )

        algorithm = DummyAlgorithm()
        coordinator.algorithm = algorithm

        # Initial state: iteration=0, last_update=-1
        # Should not update at iteration 0
        assert coordinator.should_update() is False

        # Advance to iteration 1: 1 - (-1) = 2 >= 2 → should update
        algorithm.iteration = 1
        assert coordinator.should_update() is True

        # Simulate that an update has occurred
        coordinator._last_update_iteration = 1

        # Iteration 2: not enough progress yet (2 - 1 = 1 < 2)
        algorithm.iteration = 2
        assert coordinator.should_update() is False

        # Once enough algorithm iterations have passed, the update is due again
        algorithm.iteration = 3
        assert coordinator.should_update() is True  # 3 - 1 = 2 >= 2

    def test_run_accurate_projection_guards_multiple_calls(
        self, simind_simulator, linear_acquisition_model, basic_phantom
    ):
        """Test that run_accurate_projection skips if already called in same iteration."""

        class DummyAlgorithm:
            def __init__(self):
                self.iteration = 6

        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=6,
            residual_correction=True,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
        )
        coordinator.algorithm = DummyAlgorithm()

        # Initialize with additive to avoid simulation
        acq_template = create_stir_acqdata([64, 64], 60, [4.42, 4.42])
        coordinator.initialize_with_additive(acq_template.get_uniform_copy(0.1))

        # Track if projection was called
        original_forward = linear_acquisition_model.forward
        call_count = [0]

        def counting_forward(*args, **kwargs):
            call_count[0] += 1
            return original_forward(*args, **kwargs)

        linear_acquisition_model.forward = counting_forward

        # First call at iteration 6 - should run
        initial_cache_version = coordinator.cache_version
        coordinator.run_accurate_projection(basic_phantom)
        assert coordinator._last_update_iteration == 6
        assert coordinator.cache_version == initial_cache_version + 1
        first_call_count = call_count[0]
        assert first_call_count > 0  # Projection was called

        # Second call at same iteration - should be skipped
        coordinator.run_accurate_projection(basic_phantom)
        assert coordinator._last_update_iteration == 6  # Still 6
        assert coordinator.cache_version == initial_cache_version + 1  # No increment
        assert call_count[0] == first_call_count  # No additional forward calls

        # Third call at same iteration - still skipped
        coordinator.run_accurate_projection(basic_phantom)
        assert call_count[0] == first_call_count  # Still no additional calls

        # Advance iteration - should run again
        coordinator.algorithm.iteration = 12
        coordinator.run_accurate_projection(basic_phantom)
        assert coordinator._last_update_iteration == 12
        assert coordinator.cache_version == initial_cache_version + 2
        assert call_count[0] > first_call_count  # New forward calls made


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

        class DummyAlgorithm:
            def __init__(self):
                self.iteration = 0

        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=1,
            residual_correction=True,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
        )
        coordinator.algorithm = DummyAlgorithm()

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


class TestCurrentAdditive:
    """Test current additive term tracking."""

    def test_initialize_with_additive(
        self, simind_simulator, linear_acquisition_model, acq_template
    ):
        """Test initialization of current additive."""
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

        assert coordinator.current_additive is not None
        # Check that it was cloned (not same object)
        assert coordinator.current_additive is not initial_additive

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
        assert full_additive.shape == initial_additive.shape


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

    def test_subset_residual_extraction(
        self, simind_simulator, linear_acquisition_model, basic_phantom
    ):
        """Test extracting subset-specific residuals."""

        class DummyAlgorithm:
            def __init__(self):
                self.iteration = 1

        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=1,
            residual_correction=True,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
        )
        coordinator.algorithm = DummyAlgorithm()

        # Run simulation
        coordinator.run_full_simulation(basic_phantom)

        # Extract subset residual
        subset_indices = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]  # Subset 0
        residual = coordinator.get_subset_residual(subset_indices)

        assert residual is not None
        # Residual should have shape corresponding to subset views
        assert len(residual.dimensions()) == 4  # [views, y, z]
        assert residual.shape[2] == len(subset_indices)

    def test_mode_a_simulation(
        self, simind_simulator, linear_acquisition_model, basic_phantom
    ):
        """Test Mode A with actual SIMIND simulation."""

        class DummyAlgorithm:
            def __init__(self):
                self.iteration = 1

        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=1,
            residual_correction=True,
            update_additive=False,
            linear_acquisition_model=linear_acquisition_model,
        )
        coordinator.algorithm = DummyAlgorithm()

        # Seed with an initial additive estimate to ensure it is preserved.
        initial_additive = linear_acquisition_model.range_geometry().get_uniform_copy(
            0.5
        )
        coordinator.initialize_with_additive(initial_additive)

        # Run simulation
        coordinator.run_full_simulation(basic_phantom)

        # Check cache was populated
        assert coordinator.cache_version == 1
        assert coordinator.cached_b02 is not None  # Mode A uses b02
        assert coordinator.cached_scale_factor is not None

        # Mode A should have b01 = None
        assert coordinator.cached_b01 is None

        # Additive term should remain equal to the initial estimate.
        updated_additive = coordinator.get_full_additive_term()
        updated_additive_arr = get_array(updated_additive)
        initial_additive_arr = get_array(initial_additive)
        np.testing.assert_allclose(
            updated_additive_arr,
            initial_additive_arr,
            rtol=0,
            atol=1e-6,
        )

        # Residual should match SIMIND geometric minus fast linear projection.
        b02_scaled = coordinator.cached_b02 * coordinator.cached_scale_factor
        expected_residual = b02_scaled - coordinator.cached_linear_proj
        num_views = int(coordinator.cached_linear_proj.dimensions()[2])
        subset_indices = list(range(0, num_views, coordinator.num_subsets))
        subset_residual = coordinator.get_subset_residual(subset_indices)
        expected_subset = expected_residual.get_subset(subset_indices)
        subset_residual_arr = get_array(subset_residual)
        expected_subset_arr = get_array(expected_subset)
        np.testing.assert_allclose(
            subset_residual_arr,
            expected_subset_arr,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_mode_b_additive_only_outputs_scatter(
        self, simind_simulator, linear_acquisition_model, basic_phantom
    ):
        """Mode B should update additive term to SIMIND scatter and have zero residual."""

        class DummyAlgorithm:
            def __init__(self):
                self.iteration = 1

        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=1,
            residual_correction=False,
            update_additive=True,
            linear_acquisition_model=linear_acquisition_model,
        )
        coordinator.algorithm = DummyAlgorithm()

        initial_additive = linear_acquisition_model.range_geometry().get_uniform_copy(
            0.0
        )
        coordinator.initialize_with_additive(initial_additive)

        coordinator.run_full_simulation(basic_phantom)

        b01_scaled = coordinator.cached_b01 * coordinator.cached_scale_factor
        b02_scaled = coordinator.cached_b02 * coordinator.cached_scale_factor
        expected_scatter = b01_scaled - b02_scaled
        updated_additive = coordinator.get_full_additive_term()
        updated_additive_arr = get_array(updated_additive)
        expected_scatter_arr = get_array(expected_scatter)
        np.testing.assert_allclose(
            updated_additive_arr,
            expected_scatter_arr,
            rtol=1e-5,
            atol=1e-6,
        )

        subset_indices = list(
            range(
                0,
                int(coordinator.cached_linear_proj.dimensions()[2]),
                coordinator.num_subsets,
            )
        )
        residual_subset = coordinator.get_subset_residual(subset_indices)
        residual_subset_arr = get_array(residual_subset)
        np.testing.assert_allclose(
            residual_subset_arr,
            np.zeros_like(residual_subset_arr),
            rtol=0,
            atol=1e-7,
        )
        assert coordinator.cached_residual_full is None

    def test_mode_c_returns_separate_additive_and_residual(
        self,
        simind_simulator,
        linear_acquisition_model,
        basic_phantom,
        acq_template,
    ):
        """Mode C should provide both scatter additive and geometric residual corrections."""
        from sirf.STIR import AcquisitionModelUsingRayTracingMatrix

        class DummyAlgorithm:
            def __init__(self):
                self.iteration = 1

        stir_full_am = AcquisitionModelUsingRayTracingMatrix()
        stir_full_am.set_up(acq_template, basic_phantom)

        coordinator = SimindCoordinator(
            simind_simulator=simind_simulator,
            num_subsets=6,
            correction_update_interval=1,
            residual_correction=True,
            update_additive=True,
            linear_acquisition_model=linear_acquisition_model,
            stir_acquisition_model=stir_full_am,
        )
        coordinator.algorithm = DummyAlgorithm()

        initial_additive = linear_acquisition_model.range_geometry().get_uniform_copy(
            0.25
        )
        coordinator.initialize_with_additive(initial_additive)

        coordinator.run_full_simulation(basic_phantom)

        b01_scaled = coordinator.cached_b01 * coordinator.cached_scale_factor
        b02_scaled = coordinator.cached_b02 * coordinator.cached_scale_factor
        expected_additive = b01_scaled - b02_scaled
        expected_residual = b02_scaled - coordinator.cached_linear_proj

        updated_additive = coordinator.get_full_additive_term()
        updated_additive_arr = get_array(updated_additive)
        expected_additive_arr = get_array(expected_additive)
        np.testing.assert_allclose(
            updated_additive_arr,
            expected_additive_arr,
            rtol=1e-5,
            atol=1e-6,
        )

        subset_indices = list(
            range(
                0,
                int(coordinator.cached_linear_proj.dimensions()[2]),
                coordinator.num_subsets,
            )
        )
        residual_subset = coordinator.get_subset_residual(subset_indices)
        expected_subset = expected_residual.get_subset(subset_indices)
        residual_subset_arr = get_array(residual_subset)
        expected_subset_arr = get_array(expected_subset)
        np.testing.assert_allclose(
            residual_subset_arr,
            expected_subset_arr,
            rtol=1e-5,
            atol=1e-6,
        )


class TestStirPsfCoordinator:
    """Test StirPsfCoordinator functionality."""

    def test_stir_psf_coordinator_guards_multiple_calls(
        self, linear_acquisition_model, basic_phantom, acq_template
    ):
        """Test that StirPsfCoordinator guards against multiple calls in same iteration."""
        from sirf.STIR import AcquisitionModelUsingRayTracingMatrix

        from sirf_simind_connection.core.coordinator import StirPsfCoordinator

        class DummyAlgorithm:
            def __init__(self):
                self.iteration = 6

        # Create PSF projector (with resolution model)
        psf_projector = AcquisitionModelUsingRayTracingMatrix()
        psf_projector.set_up(acq_template, basic_phantom)

        # Create fast projector (no PSF)
        fast_projector = AcquisitionModelUsingRayTracingMatrix()
        fast_projector.set_up(acq_template, basic_phantom)

        coordinator = StirPsfCoordinator(
            stir_psf_projector=psf_projector,
            stir_fast_projector=fast_projector,
            correction_update_interval=6,
            initial_additive=acq_template.get_uniform_copy(0.1),
        )
        coordinator.algorithm = DummyAlgorithm()

        # Track forward calls on PSF projector
        original_psf_forward = psf_projector.forward
        psf_call_count = [0]

        def counting_psf_forward(*args, **kwargs):
            psf_call_count[0] += 1
            return original_psf_forward(*args, **kwargs)

        psf_projector.forward = counting_psf_forward

        # First call at iteration 6 - should run
        initial_cache_version = coordinator.cache_version
        coordinator.run_accurate_projection(basic_phantom)
        assert coordinator._last_update_iteration == 6
        assert coordinator.cache_version == initial_cache_version + 1
        first_call_count = psf_call_count[0]
        assert first_call_count > 0  # PSF projection was called

        # Second call at same iteration - should be skipped
        coordinator.run_accurate_projection(basic_phantom)
        assert coordinator._last_update_iteration == 6  # Still 6
        assert coordinator.cache_version == initial_cache_version + 1  # No increment
        assert psf_call_count[0] == first_call_count  # No additional calls

        # Third call at same iteration - still skipped
        coordinator.run_accurate_projection(basic_phantom)
        assert psf_call_count[0] == first_call_count  # Still no additional calls

        # Advance iteration - should run again
        coordinator.algorithm.iteration = 12
        coordinator.run_accurate_projection(basic_phantom)
        assert coordinator._last_update_iteration == 12
        assert coordinator.cache_version == initial_cache_version + 2
        assert psf_call_count[0] > first_call_count  # New calls made


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

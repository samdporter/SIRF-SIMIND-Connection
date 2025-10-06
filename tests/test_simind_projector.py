import tempfile

import pytest

from sirf_simind_connection import SimindProjector, SimindSimulator, SimulationConfig
from sirf_simind_connection.configs import get
from sirf_simind_connection.core.components import ScoringRoutine
from sirf_simind_connection.utils.stir_utils import (
    create_attenuation_map,
    create_simple_phantom,
    create_stir_acqdata,
)


# All tests in this file require SIRF since SimindProjector uses SIRF
pytestmark = pytest.mark.requires_sirf


def test_projector_initialization():
    """Test initialization of SimindProjector."""
    projector = SimindProjector()
    assert projector is not None
    assert projector._iteration_counter == 0
    assert projector._last_update_iteration == -1
    assert not projector.update_additive
    assert not projector.residual_correction


def test_projector_set_up():
    """Test set_up method."""
    from sirf.STIR import (
        AcquisitionModelUsingRayTracingMatrix,
    )

    # Create test templates
    phantom = create_simple_phantom()
    acq_template = create_stir_acqdata([128, 128], 120, [4.42, 4.42])
    stir_am = AcquisitionModelUsingRayTracingMatrix()

    # Create projector and set up
    projector = SimindProjector(stir_projector=stir_am)
    projector.set_up(acq_template, phantom)

    assert projector.acq_templ is not None
    assert projector.img_templ is not None
    assert projector._linear_acquisition_model is not None
    assert projector._current_additive is not None


def test_iteration_tracking():
    """Test auto-iteration tracking in forward()."""
    from sirf.STIR import AcquisitionModelUsingRayTracingMatrix

    phantom = create_simple_phantom()
    acq_template = create_stir_acqdata([128, 128], 120, [4.42, 4.42])
    stir_am = AcquisitionModelUsingRayTracingMatrix()

    projector = SimindProjector(stir_projector=stir_am)
    projector.set_up(acq_template, phantom)

    # Check initial state
    assert projector._iteration_counter == 0

    # Call forward multiple times
    projector.forward(phantom)
    assert projector._iteration_counter == 1

    projector.forward(phantom)
    assert projector._iteration_counter == 2

    projector.forward(phantom)
    assert projector._iteration_counter == 3

    # Test reset
    projector.reset_iteration_counter()
    assert projector._iteration_counter == 0
    assert projector._last_update_iteration == -1


def test_should_update_corrections():
    """Test _should_update_corrections logic."""
    projector = SimindProjector(correction_update_interval=3)

    # No updates if flags are False
    assert not projector._should_update_corrections()

    # Enable residual correction
    projector.residual_correction = True

    # Still no updates without simulator
    assert not projector._should_update_corrections()

    # Add mock simulator
    projector._simind_simulator = "mock"

    # Now should update based on interval
    projector._iteration_counter = 0
    assert projector._should_update_corrections()

    projector._iteration_counter = 1
    assert not projector._should_update_corrections()

    projector._iteration_counter = 3
    assert projector._should_update_corrections()

    projector._iteration_counter = 6
    assert projector._should_update_corrections()


def test_acquisition_model_interface():
    """Test AcquisitionModel interface compatibility."""
    from sirf.STIR import AcquisitionModelUsingRayTracingMatrix

    phantom = create_simple_phantom()
    acq_template = create_stir_acqdata([128, 128], 120, [4.42, 4.42])
    stir_am = AcquisitionModelUsingRayTracingMatrix()

    projector = SimindProjector(stir_projector=stir_am)
    projector.set_up(acq_template, phantom)

    # Test interface methods
    assert projector.is_affine()
    assert hasattr(projector, "is_linear")
    assert hasattr(projector, "get_additive_term")
    assert hasattr(projector, "set_additive_term")
    assert hasattr(projector, "get_background_term")
    assert hasattr(projector, "set_background_term")
    assert hasattr(projector, "get_constant_term")
    assert hasattr(projector, "get_linear_acquisition_model")
    assert hasattr(projector, "direct")
    assert hasattr(projector, "adjoint")
    assert hasattr(projector, "range_geometry")
    assert hasattr(projector, "domain_geometry")

    # Test that linear model is created
    linear_am = projector.get_linear_acquisition_model()
    assert linear_am is not None


def test_direct_and_adjoint():
    """Test direct and adjoint operators (CIL compatibility)."""
    from sirf.STIR import AcquisitionModelUsingRayTracingMatrix

    phantom = create_simple_phantom()
    acq_template = create_stir_acqdata([128, 128], 120, [4.42, 4.42])
    stir_am = AcquisitionModelUsingRayTracingMatrix()

    projector = SimindProjector(stir_projector=stir_am)
    projector.set_up(acq_template, phantom)

    # Test direct (should call forward)
    fwd_result = projector.forward(phantom)
    direct_result = projector.direct(phantom)
    assert fwd_result is not None
    assert direct_result is not None

    # Test adjoint (should call backward)
    bwd_result = projector.backward(fwd_result)
    adj_result = projector.adjoint(fwd_result)
    assert bwd_result is not None
    assert adj_result is not None


def test_additive_and_background_terms():
    """Test getting/setting additive and background terms."""
    from sirf.STIR import AcquisitionModelUsingRayTracingMatrix

    phantom = create_simple_phantom()
    acq_template = create_stir_acqdata([128, 128], 120, [4.42, 4.42])
    stir_am = AcquisitionModelUsingRayTracingMatrix()

    projector = SimindProjector(stir_projector=stir_am)
    projector.set_up(acq_template, phantom)

    # Get initial additive term
    additive = projector.get_additive_term()
    assert additive is not None

    # Set new additive term
    new_additive = additive.get_uniform_copy(1.0)
    projector.set_additive_term(new_additive)
    retrieved = projector.get_additive_term()
    assert retrieved.max() > 0

    # Get background term
    background = projector.get_background_term()
    assert background is not None

    # Get constant term
    constant = projector.get_constant_term()
    assert constant is not None


@pytest.mark.requires_simind
def test_correction_modes_integration():
    """Test the three correction modes with actual SIMIND simulation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup
        config = SimulationConfig(get("AnyScan.yaml"))
        simulator = SimindSimulator(
            config, temp_dir, scoring_routine=ScoringRoutine.PENETRATE
        )
        phantom = create_simple_phantom()
        acq_template = create_stir_acqdata([128, 128], 120, [4.42, 4.42])
        mu_map = create_attenuation_map(phantom)

        simulator.set_source(phantom)
        simulator.set_mu_map(mu_map)
        simulator.set_energy_windows([126], [154], [0])

        from sirf.STIR import AcquisitionModelUsingRayTracingMatrix

        stir_am = AcquisitionModelUsingRayTracingMatrix()

        # Test Mode A: Residual only
        projector_a = SimindProjector(
            simind_simulator=simulator,
            stir_projector=stir_am,
            correction_update_interval=1,
            update_additive=False,
            residual_correction=True,
        )
        projector_a.set_up(acq_template, phantom)

        # Mode A should disable collimator routine
        assert hasattr(projector_a, "_update_corrections")

        # Test Mode B: Additive only
        projector_b = SimindProjector(
            simind_simulator=simulator,
            stir_projector=stir_am,
            correction_update_interval=1,
            update_additive=True,
            residual_correction=False,
        )
        projector_b.set_up(acq_template, phantom)

        # Test Mode C: Both
        projector_c = SimindProjector(
            simind_simulator=simulator,
            stir_projector=stir_am,
            correction_update_interval=1,
            update_additive=True,
            residual_correction=True,
        )
        projector_c.set_up(acq_template, phantom)


def test_set_collimator_routine():
    """Test set_collimator_routine method in SimindSimulator."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = SimulationConfig(get("AnyScan.yaml"))
        simulator = SimindSimulator(config, temp_dir)

        # Test enabling
        simulator.set_collimator_routine(True)
        assert simulator.config.get_value(53) == 1

        # Test disabling
        simulator.set_collimator_routine(False)
        assert simulator.config.get_value(53) == 0

import numpy as np
import pytest

from project_psf.step_size_rules import (
    ArmijoAfterCorrectionStepSize,
    SaveStepSizeHistoryCallback,
)


class MockDataContainer:
    """Minimal stand-in for CIL DataContainer used in step-size tests."""

    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)

    def copy(self):
        return MockDataContainer(self.data.copy())

    def sapyb(self, a, other, b, out=None):
        other_data = other.data if isinstance(other, MockDataContainer) else other
        result = a * self.data + b * other_data
        if out is not None:
            out.data[:] = result
            return out
        return MockDataContainer(result)

    def dot(self, other):
        other_data = other.data if isinstance(other, MockDataContainer) else other
        return float(np.dot(self.data, other_data))

    def fill(self, other):
        other_data = other.data if isinstance(other, MockDataContainer) else other
        self.data[:] = other_data


class QuadraticFunction:
    """Simple quadratic f(x) = 0.5 ||x||^2 with gradient = x."""

    def __call__(self, x):
        return 0.5 * x.dot(x)

    def gradient(self, x, out):
        out.fill(x)


class ZeroFunction:
    """Prox-friendly g(x) = 0."""

    def __call__(self, x):
        return 0.0

    def proximal(self, x, step, out=None):
        if out is not None:
            out.fill(x)
            return out
        return x.copy()


class RecordingPreconditioner:
    """Preconditioner that scales the gradient and records call metadata."""

    def __init__(self, scale=1.0):
        self.scale = scale
        self.called_with_copy = None
        self.call_count = 0

    def apply(self, algorithm, gradient):
        self.call_count += 1
        self.called_with_copy = gradient is not algorithm.gradient_update
        return MockDataContainer(self.scale * gradient.data)


class MockAlgorithm:
    """Minimal ISTA-like algorithm stub for exercising the step-size rule."""

    def __init__(self, initial, preconditioner=None, iteration=-1):
        self.iteration = iteration
        self.solution = initial.copy()
        self.gradient_update = initial.copy()
        self.preconditioner = preconditioner
        self.f = QuadraticFunction()
        self.g = ZeroFunction()
        self.step_size_rule = None


@pytest.fixture
def armijo_rule():
    return ArmijoAfterCorrectionStepSize(
        initial_step_size=1.0,
        beta=0.5,
        decay_rate=0.0,
        max_iter=10,
        tol=1e-4,
        update_interval=0,
        initial_armijo_iterations=0,
    )


def test_armijo_triggers_on_first_iteration_with_preconditioner(armijo_rule):
    precond = RecordingPreconditioner(scale=1.0)
    algo = MockAlgorithm(
        MockDataContainer([1.0, 1.0]), preconditioner=precond, iteration=-1
    )
    step = armijo_rule.get_step_size(algo)

    assert precond.call_count == 1
    assert precond.called_with_copy is True, (
        "Preconditioner should see a copy of the gradient"
    )
    np.testing.assert_allclose(algo.gradient_update.data, [1.0, 1.0])
    assert armijo_rule.armijo_ran_this_iteration is True
    assert step == pytest.approx(1.0)


def test_armijo_periodic_trigger_runs_on_epoch_boundary():
    rule = ArmijoAfterCorrectionStepSize(
        initial_step_size=1.0,
        beta=0.5,
        decay_rate=0.0,
        max_iter=10,
        tol=1e-4,
        update_interval=2,
        initial_armijo_iterations=0,
    )
    algo = MockAlgorithm(
        MockDataContainer([1.0, 1.0]),
        preconditioner=RecordingPreconditioner(),
        iteration=-1,
    )

    # First iteration: forced Armijo
    assert rule.get_step_size(algo) == pytest.approx(1.0)
    assert rule.armijo_ran_this_iteration

    # Next iteration (iteration == 0 -> pending 1): linear decay path
    algo.iteration = 0
    step_linear = rule.get_step_size(algo)
    assert rule.armijo_ran_this_iteration is False
    assert step_linear == pytest.approx(1.0)  # decay_rate=0 keeps it unchanged

    # Following iteration hits pending_iter % update_interval == 0 -> Armijo
    algo.iteration = 1
    step_periodic = rule.get_step_size(algo)
    assert rule.armijo_ran_this_iteration is True
    assert step_periodic == pytest.approx(1.0)


def test_armijo_warmup_forces_initial_iterations():
    rule = ArmijoAfterCorrectionStepSize(
        initial_step_size=1.0,
        beta=0.5,
        decay_rate=0.0,
        max_iter=10,
        tol=1e-4,
        update_interval=0,
        initial_armijo_iterations=2,
    )
    algo = MockAlgorithm(MockDataContainer([1.0, 1.0]), iteration=-1)

    # First iteration (pending 0)
    rule.get_step_size(algo)
    assert rule.armijo_ran_this_iteration is True

    # Second iteration (pending 1) still within warmup window
    algo.iteration = 0
    rule.get_step_size(algo)
    assert rule.armijo_ran_this_iteration is True

    # Third iteration exits warmup
    algo.iteration = 1
    rule.get_step_size(algo)
    assert rule.armijo_ran_this_iteration is False


def test_reinitialize_decay_resets_linear_decay():
    rule = ArmijoAfterCorrectionStepSize(
        initial_step_size=2.0,
        beta=0.5,
        decay_rate=1.0,
        max_iter=5,
        tol=1e-4,
        update_interval=0,
        initial_armijo_iterations=0,
    )
    algo = MockAlgorithm(MockDataContainer([1.0, 1.0]), iteration=0)
    rule.current_step_size = 2.0
    rule.reinitialize_decay(start_iteration=0)

    first_step = rule.get_step_size(algo)
    assert first_step == pytest.approx(2.0)

    algo.iteration = 1
    second_step = rule.get_step_size(algo)
    assert second_step == pytest.approx(1.0)


def test_apply_warmup_cap_sets_initial_and_current():
    rule = ArmijoAfterCorrectionStepSize(
        initial_step_size=1.0,
        beta=0.5,
        decay_rate=0.0,
        max_iter=10,
        tol=1e-4,
        update_interval=0,
        initial_armijo_iterations=0,
    )
    rule.min_step_size_seen = 0.25
    rule.current_step_size = 0.4

    capped = rule.apply_warmup_cap()
    assert capped == pytest.approx(0.5)
    assert rule.initial_step_size == pytest.approx(0.5)
    assert rule.current_step_size == pytest.approx(0.5)


def test_step_size_history_iteration_offset(tmp_path):
    csv_path = tmp_path / "history.csv"
    callback = SaveStepSizeHistoryCallback(str(csv_path))

    class DummyRule:
        def __init__(self, step_size, ran=False, trigger=False):
            self.current_step_size = step_size
            self.armijo_ran_this_iteration = ran
            self.trigger_armijo = trigger

    algo = MockAlgorithm(MockDataContainer([1.0, 1.0]), iteration=0)
    algo.step_size_rule = DummyRule(0.25)
    callback(algo)

    callback.increment_iteration_offset(5)

    algo.iteration = 0
    algo.step_size_rule = DummyRule(0.2, ran=True)
    callback(algo)

    iterations = [entry["iteration"] for entry in callback.step_size_history]
    assert iterations == [0, 5]

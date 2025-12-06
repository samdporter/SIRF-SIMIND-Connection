import numpy as np
import pytest

from project_psf.step_size_rules import (
    ArmijoAfterCorrectionStepSize,
    ArmijoPeriodicCallback,
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
    )


def test_armijo_runs_only_after_trigger(armijo_rule):
    precond = RecordingPreconditioner(scale=1.0)
    algo = MockAlgorithm(
        MockDataContainer([1.0, 1.0]), preconditioner=precond, iteration=-1
    )

    step = armijo_rule.get_step_size(algo)
    assert precond.call_count == 0
    assert armijo_rule.armijo_ran_this_iteration is False
    assert step == pytest.approx(1.0)

    armijo_rule.trigger_armijo = True
    triggered_step = armijo_rule.get_step_size(algo)

    assert precond.call_count == 1
    assert precond.called_with_copy is True
    np.testing.assert_allclose(algo.gradient_update.data, [1.0, 1.0])
    assert armijo_rule.armijo_ran_this_iteration is True
    assert triggered_step == pytest.approx(1.0)


def test_force_armijo_schedules_next_iteration_search(armijo_rule):
    precond = RecordingPreconditioner(scale=0.5)
    algo = MockAlgorithm(
        MockDataContainer([1.0, 1.0]), preconditioner=precond, iteration=3
    )

    forced_step = armijo_rule.force_armijo_after_correction(algo)
    assert armijo_rule.trigger_armijo is True
    assert armijo_rule.armijo_ran_this_iteration is False
    assert forced_step == pytest.approx(1.0)
    assert precond.call_count == 0

    scheduled_step = armijo_rule.get_step_size(algo)
    assert scheduled_step == pytest.approx(forced_step)
    assert armijo_rule.armijo_ran_this_iteration is True
    assert precond.call_count == 1


def test_reset_clears_triggers(armijo_rule):
    algo = MockAlgorithm(MockDataContainer([1.0, 1.0]), iteration=0)
    armijo_rule.trigger_armijo = True
    armijo_rule.armijo_ran_this_iteration = True
    armijo_rule.cached_step_iteration = 0
    armijo_rule.cached_step_value = 0.1

    armijo_rule.reset()
    step = armijo_rule.get_step_size(algo)

    assert step == pytest.approx(1.0)
    assert armijo_rule.trigger_armijo is False
    assert armijo_rule.armijo_ran_this_iteration is False


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


def test_armijo_periodic_callback_triggers_every_interval(armijo_rule):
    algo = MockAlgorithm(MockDataContainer([1.0, 1.0]), iteration=-1)
    algo.step_size_rule = armijo_rule
    callback = ArmijoPeriodicCallback(iteration_interval=2)

    callback(algo)
    assert armijo_rule.armijo_ran_this_iteration is False
    assert armijo_rule.trigger_armijo is False

    algo.iteration = 0
    callback(algo)
    assert armijo_rule.armijo_ran_this_iteration is False
    assert armijo_rule.trigger_armijo is False

    algo.iteration = 1
    callback(algo)
    assert armijo_rule.trigger_armijo is True
    assert armijo_rule.armijo_ran_this_iteration is False
    armijo_rule.get_step_size(algo)
    assert armijo_rule.armijo_ran_this_iteration is True
    assert armijo_rule.trigger_armijo is False

    armijo_rule.armijo_ran_this_iteration = False
    algo.iteration = 2
    callback(algo)
    assert armijo_rule.armijo_ran_this_iteration is False
    assert armijo_rule.trigger_armijo is False

    algo.iteration = 3
    callback(algo)
    assert armijo_rule.trigger_armijo is True
    assert armijo_rule.armijo_ran_this_iteration is False
    armijo_rule.get_step_size(algo)
    assert armijo_rule.armijo_ran_this_iteration is True
    assert armijo_rule.trigger_armijo is False

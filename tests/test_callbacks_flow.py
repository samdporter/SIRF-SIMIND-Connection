import csv
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parent.parent
PROJECT_PSF_DIR = ROOT / "project_psf"


def _stub_module(name, attrs=None):
    mod = ModuleType(name)
    # Mark as package to allow submodule imports
    mod.__path__ = []
    attrs = attrs or {}
    for key, val in attrs.items():
        setattr(mod, key, val)
    sys.modules[name] = mod
    return mod


def _install_stubs_if_needed(force=False):
    """
    Install lightweight stubs for heavy dependencies (cil, setr, sirf, sirf_simind_connection)
    only when real imports fail (e.g., missing shared libraries) or when forced.
    """
    if not force:
        try:
            import cil  # noqa: F401

            return  # Real dependency available; no stubs required
        except Exception:
            # Clear any partial imports
            for prefix in ("cil", "setr", "sirf", "sirf_simind_connection"):
                for key in list(sys.modules.keys()):
                    if key == prefix or key.startswith(prefix + "."):
                        sys.modules.pop(key)

    # ---- CIL stubs ----
    class _DummyStepSizeRule:
        def __init__(self, *_, **__):
            pass

    class _DummySampler:
        @staticmethod
        def random_without_replacement(_):
            return None

    cil_mod = _stub_module("cil")
    cil_opt = _stub_module("cil.optimisation")
    cil_mod.optimisation = cil_opt

    class _DummyISTA:
        pass

    cil_opt.algorithms = _stub_module(
        "cil.optimisation.algorithms", {"ISTA": _DummyISTA}
    )
    cil_opt.functions = _stub_module(
        "cil.optimisation.functions",
        {"IndicatorBox": object, "ScaledFunction": object},
    )
    cil_opt.utilities = _stub_module(
        "cil.optimisation.utilities",
        {"Sampler": _DummySampler, "StepSizeRule": _DummyStepSizeRule},
    )

    # ---- SETR stubs ----
    setr_mod = _stub_module("setr")
    setr_ce = _stub_module("setr.cil_extensions")
    setr_mod.cil_extensions = setr_ce
    setr_ce_algorithms = _stub_module(
        "setr.cil_extensions.algorithms",
        {
            "algorithms": _stub_module(
                "setr.cil_extensions.algorithms.algorithms",
                {"ista_update_step": lambda *_, **__: None},
            )
        },
    )
    setr_ce.algorithms = setr_ce_algorithms
    setr_ce.callbacks = _stub_module(
        "setr.cil_extensions.callbacks",
        {"PrintObjectiveCallback": object, "SavePreconditionerCallback": object},
    )
    setr_ce.preconditioners = _stub_module(
        "setr.cil_extensions.preconditioners",
        {
            "BSREMPreconditioner": object,
            "ImageFunctionPreconditioner": object,
            "LehmerMeanPreconditioner": object,
            "PoissonHessianPreconditioner": object,
            "PreconditionerWithInterval": object,
            "SubsetPoissonHessianPreconditioner": object,
        },
    )
    setr_mod.priors = _stub_module("setr.priors", {"RelativeDifferencePrior": object})
    setr_mod.utils = _stub_module(
        "setr.utils",
        {
            "get_spect_am": lambda *_, **__: None,
            "get_spect_data": lambda *_, **__: None,
        },
    )

    # ---- pandas stub ----
    class _DummyDataFrame:
        def __init__(self, data=None):
            self.data = data

        def to_csv(self, *_, **__):
            return None

    _stub_module("pandas", {"DataFrame": _DummyDataFrame})

    # ---- SIRF stubs ----
    sirf_mod = _stub_module("sirf")
    sirf_mod.STIR = _stub_module(
        "sirf.STIR",
        {
            "AcquisitionData": object,
            "MessageRedirector": object,
            "SeparableGaussianImageFilter": object,
        },
    )

    # ---- SIRF-SIMIND stubs ----
    simind_mod = _stub_module("sirf_simind_connection")
    simind_mod.SimindSimulator = object
    simind_mod.SimulationConfig = object
    simind_mod.configs = _stub_module(
        "sirf_simind_connection.configs", {"get": lambda *_, **__: None}
    )
    simind_core = _stub_module("sirf_simind_connection.core")
    simind_core.components = _stub_module(
        "sirf_simind_connection.core.components", {"ScoringRoutine": object}
    )
    simind_core.coordinator = _stub_module(
        "sirf_simind_connection.core.coordinator",
        {"SimindCoordinator": object, "StirPsfCoordinator": object},
    )
    simind_mod.core = simind_core
    simind_mod.utils = _stub_module(
        "sirf_simind_connection.utils",
        {
            "get_array": lambda *_, **__: None,
            "cil_partitioner": _stub_module(
                "sirf_simind_connection.utils.cil_partitioner",
                {
                    "create_full_objective_with_rdp": lambda *_, **__: None,
                    "create_svrg_objective_with_rdp": lambda *_, **__: None,
                    "partition_data_with_cil_objectives": lambda *_, **__: (
                        [],
                        [],
                        [],
                        [],
                        None,
                        None,
                    ),
                },
            ),
        },
    )


_install_stubs_if_needed()

if str(PROJECT_PSF_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_PSF_DIR))


class DummySolution:
    def __init__(self, sink):
        self._sink = sink

    def write(self, path):
        self._sink.append(path)


class DummyAlgorithm:
    def __init__(
        self,
        iteration,
        obj_val,
        g_val,
        armijo_ran=False,
        step_size=0.5,
        trigger_armijo=False,
        sink=None,
    ):
        self.iteration = iteration
        self.solution_paths = sink if sink is not None else []
        self.solution = DummySolution(self.solution_paths)
        self._obj_val = obj_val
        self._g_val = g_val
        self.step_size_rule = SimpleNamespace(
            armijo_ran_this_iteration=armijo_ran,
            current_step_size=step_size,
            trigger_armijo=trigger_armijo,
        )

    def f(self, _):
        return self._obj_val

    def g(self, _):
        return self._g_val


@pytest.fixture(scope="module")
def callbacks():
    """Import callback classes with dependency stubs to avoid heavy shared libs."""
    prefixes = ("cil", "setr", "sirf", "sirf_simind_connection")
    saved_modules = {
        name: sys.modules[name]
        for name in list(sys.modules.keys())
        if name.split(".", 1)[0] in prefixes
    }

    # Clear existing to avoid partially imported modules
    for name in list(sys.modules.keys()):
        if name.split(".", 1)[0] in prefixes:
            sys.modules.pop(name, None)

    # Install stubs unconditionally for isolation
    _install_stubs_if_needed(force=True)

    added_path = False
    if str(PROJECT_PSF_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_PSF_DIR))
        added_path = True

    import importlib

    cmp_mod = importlib.import_module("compare_psf_models")
    ssr_mod = importlib.import_module("step_size_rules")

    cb = SimpleNamespace(
        SaveObjectiveCallback=cmp_mod.SaveObjectiveCallback,
        SaveImageCallback=cmp_mod.SaveImageCallback,
        ArmijoTriggerCallback=ssr_mod.ArmijoTriggerCallback,
    )

    yield cb

    # Cleanup: remove stubbed modules and restore originals
    for name in ("compare_psf_models", "step_size_rules"):
        sys.modules.pop(name, None)

    for name in list(sys.modules.keys()):
        if name.split(".", 1)[0] in prefixes:
            sys.modules.pop(name, None)
    sys.modules.update(saved_modules)

    if added_path and str(PROJECT_PSF_DIR) in sys.path:
        sys.path.remove(str(PROJECT_PSF_DIR))


def test_warmup_armijo_logs_objective_and_images(tmp_path, callbacks):
    assert hasattr(callbacks, "SaveObjectiveCallback")
    obj_cb = callbacks.SaveObjectiveCallback(
        tmp_path / "objective.csv",
        interval=18,
        start_iteration=0,
        log_on_armijo=True,
    )
    img_cb = callbacks.SaveImageCallback(tmp_path / "image", interval=1)

    # Warm-up Armijo iterations: should log despite interval being larger than 1
    saved_paths = []
    for it in range(3):
        algo = DummyAlgorithm(
            iteration=it,
            obj_val=1.0 + it,
            g_val=0.5,
            armijo_ran=True,
            sink=saved_paths,
        )
        obj_cb(algo)
        img_cb(algo)

    assert [rec["iteration"] for rec in obj_cb.records] == [0, 1, 2]
    assert obj_cb.records[0]["objective"] == pytest.approx(1.5)
    assert str(img_cb.output_prefix)  # sanity check prefix retained
    assert saved_paths == [str(tmp_path / f"image_{it}") for it in range(3)]

    # CSV written with all three entries
    with open(tmp_path / "objective.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["iteration"]) for r in rows] == [0, 1, 2]


def test_residual_update_triggers_logging_and_armijo(tmp_path, callbacks):
    assert hasattr(callbacks, "ArmijoTriggerCallback")
    coordinator = SimpleNamespace(cache_version=0)
    armijo_cb = callbacks.ArmijoTriggerCallback(coordinator)

    obj_cb = callbacks.SaveObjectiveCallback(
        tmp_path / "objective.csv",
        interval=1,
        start_iteration=0,
        log_on_armijo=True,
    )
    img_cb = callbacks.SaveImageCallback(tmp_path / "image", interval=1)

    algo = DummyAlgorithm(iteration=10, obj_val=3.0, g_val=0.2, armijo_ran=False)

    # Objective/image saved before Armijo trigger
    obj_cb(algo)
    img_cb(algo)
    assert [rec["iteration"] for rec in obj_cb.records] == [10]
    assert algo.solution_paths[-1].endswith("_10")

    # Residual update arrives (cache_version bump), Armijo is triggered for next iter
    coordinator.cache_version = 1
    armijo_cb(algo)

    assert armijo_cb.step_size_history[-1]["triggered_armijo_next"] is True
    assert algo.step_size_rule.trigger_armijo is True

    # Objective CSV persisted with the logged entry
    with open(tmp_path / "objective.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["iteration"]) for r in rows] == [10]

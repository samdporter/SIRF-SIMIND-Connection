from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest


pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_skip,
    pytest.mark.requires_sirf,
    pytest.mark.requires_simind,
]

sirf = pytest.importorskip("sirf.STIR")


ROOT = Path(__file__).resolve().parents[1]


def _parse_interfile_header(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ":=" not in line:
            continue
        key, value = line.split(":=", 1)
        values[key.strip().lower()] = value.strip()
    return values


def _header_get(header: dict[str, str], *keys: str) -> str:
    for key in keys:
        if key in header:
            return header[key]
    raise KeyError(f"Missing Interfile key from candidates: {keys}")


def _require_paths(paths: list[Path], hint: str) -> None:
    missing = [path for path in paths if not path.exists()]
    if not missing:
        return
    missing_str = ", ".join(str(path) for path in missing)
    pytest.skip(f"{hint}. Missing files: {missing_str}")


def _load_array(obj) -> np.ndarray:
    if hasattr(obj, "asarray"):
        return obj.asarray()
    return obj.as_array()


def _case_paths(case_backend: str) -> tuple[Path, Path]:
    case_dir = ROOT / "output" / f"{case_backend}_adaptor_osem"
    source_hv = case_dir / f"{case_backend}_source.hv"
    total_hs = case_dir / f"{case_backend}_total.hs"
    return source_hv, total_hs


def _forward_projection_mse(source, measured) -> float:
    acq_model = sirf.AcquisitionModelUsingRayTracingMatrix()
    acq_model.set_num_tangential_LORs(1)
    acq_model.set_up(measured, source)
    predicted = acq_model.forward(source)

    measured_arr = _load_array(measured).astype(np.float32, copy=False)
    predicted_arr = _load_array(predicted).astype(np.float32, copy=False)
    return float(np.mean((predicted_arr - measured_arr) ** 2))


def _run_osem_reconstruction(source, measured):
    initial = source.get_uniform_copy(1.0)

    acq_model = sirf.AcquisitionModelUsingRayTracingMatrix()
    acq_model.set_num_tangential_LORs(1)
    acq_model.set_up(measured, initial)

    objective = sirf.make_Poisson_loglikelihood(measured)
    objective.set_acquisition_model(acq_model)
    objective.set_num_subsets(2)

    recon = sirf.OSMAPOSLReconstructor()
    recon.set_objective_function(objective)
    recon.set_num_subsets(2)
    recon.set_num_subiterations(2)
    recon.set_input(measured)
    recon.set_up(initial)
    recon.reconstruct(initial)
    return initial


def _normalize_2d(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    return (image - image.mean()) / (image.std() + 1e-12)


def _corr_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a * b))


def _equivalent_radius(image: np.ndarray, threshold_fraction: float = 0.35) -> float:
    vmax = float(np.max(image))
    if vmax <= 0:
        return 0.0
    area = int(np.count_nonzero(image >= vmax * threshold_fraction))
    if area <= 0:
        return 0.0
    return math.sqrt(area / math.pi)


def _flip_source_x(source):
    src_arr = _load_array(source).astype(np.float32, copy=False)
    flipped_arr = np.flip(src_arr, axis=2).copy()
    flipped = source.clone()
    flipped.fill(flipped_arr)
    return flipped


def _write_variant_header(
    base_hs: Path,
    tmp_path: Path,
    direction: str,
    start_angle: float,
) -> Path:
    base_header = _parse_interfile_header(base_hs)
    data_name = _header_get(base_header, "!name of data file", "name of data file")
    data_path = (base_hs.parent / data_name).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Projection data file not found: {data_path}")

    copied_data = tmp_path / data_path.name
    shutil.copy2(data_path, copied_data)

    output_hs = tmp_path / f"{base_hs.stem}_{direction}_{int(start_angle)}.hs"
    lines: list[str] = []
    replaced_direction = False
    replaced_start = False

    for line in base_hs.read_text().splitlines():
        lower = line.strip().lower()

        if lower.startswith("!name of data file"):
            line = f"!name of data file := {copied_data.name}"
        elif lower.startswith("!direction of rotation"):
            line = f"!direction of rotation := {direction}"
            replaced_direction = True
        elif lower.startswith("start angle"):
            line = f"start angle := {float(start_angle):.1f}"
            replaced_start = True

        lines.append(line)

    if not replaced_direction:
        raise AssertionError(f"No direction-of-rotation field found in {base_hs}")
    if not replaced_start:
        raise AssertionError(f"No start-angle field found in {base_hs}")

    output_hs.write_text("\n".join(lines) + "\n")
    return output_hs


@pytest.mark.parametrize("case_backend", ["stir", "sirf"])
def test_projection_geometry_prefers_unflipped_source(case_backend: str) -> None:
    source_hv, total_hs = _case_paths(case_backend)
    _require_paths(
        [source_hv, total_hs],
        hint=(
            f"Run examples/07{'A' if case_backend == 'stir' else 'B'}_"
            f"{case_backend}_adaptor_osem.py first"
        ),
    )

    source = sirf.ImageData(str(source_hv))
    measured = sirf.AcquisitionData(str(total_hs))

    baseline_mse = _forward_projection_mse(source, measured)
    flipped_mse = _forward_projection_mse(_flip_source_x(source), measured)

    assert baseline_mse <= flipped_mse, (
        f"{case_backend.upper()} projection geometry looks mirrored. "
        f"MSE(source)={baseline_mse:.6g}, MSE(flip_x(source))={flipped_mse:.6g}"
    )


@pytest.mark.parametrize("case_backend", ["stir", "sirf"])
def test_projection_header_direction_start_angle_baseline_is_best(
    case_backend: str, tmp_path: Path
) -> None:
    source_hv, total_hs = _case_paths(case_backend)
    _require_paths(
        [source_hv, total_hs],
        hint=(
            f"Run examples/07{'A' if case_backend == 'stir' else 'B'}_"
            f"{case_backend}_adaptor_osem.py first"
        ),
    )

    source = sirf.ImageData(str(source_hv))
    baseline_header = _parse_interfile_header(total_hs)
    baseline_direction = _header_get(
        baseline_header, "!direction of rotation", "direction of rotation"
    ).upper()
    baseline_start_angle = float(_header_get(baseline_header, "start angle"))

    candidates: list[tuple[str, float]] = []
    for direction in ("CCW", "CW"):
        for start_angle in (0.0, 90.0, 180.0, 270.0):
            candidates.append((direction, start_angle))

    scores: dict[tuple[str, float], float] = {}
    for direction, start_angle in candidates:
        variant_hs = _write_variant_header(
            base_hs=total_hs,
            tmp_path=tmp_path,
            direction=direction,
            start_angle=start_angle,
        )
        measured_variant = sirf.AcquisitionData(str(variant_hs))
        scores[(direction, start_angle)] = _forward_projection_mse(
            source, measured_variant
        )

    best_variant = min(scores.items(), key=lambda item: item[1])[0]
    baseline_variant = (baseline_direction, baseline_start_angle)
    baseline_score = scores[baseline_variant]
    best_score = scores[best_variant]

    # If geometry metadata is correct, the current header should be optimal.
    assert best_variant == baseline_variant, (
        f"{case_backend.upper()} header geometry appears suboptimal. "
        f"Baseline={baseline_variant} mse={baseline_score:.6g}, "
        f"Best={best_variant} mse={best_score:.6g}, all_scores={scores}"
    )


@pytest.mark.parametrize("case_backend", ["stir", "sirf"])
def test_mapping_sweep_finds_candidate_better_than_baseline_for_recon(
    case_backend: str, tmp_path: Path
) -> None:
    source_hv, total_hs = _case_paths(case_backend)
    _require_paths(
        [source_hv, total_hs],
        hint=(
            f"Run examples/07{'A' if case_backend == 'stir' else 'B'}_"
            f"{case_backend}_adaptor_osem.py first"
        ),
    )

    source = sirf.ImageData(str(source_hv))
    source_mid = _load_array(source)[_load_array(source).shape[0] // 2]
    source_norm = _normalize_2d(source_mid)

    baseline_header = _parse_interfile_header(total_hs)
    baseline_direction = _header_get(
        baseline_header, "!direction of rotation", "direction of rotation"
    ).upper()
    baseline_start_angle = float(_header_get(baseline_header, "start angle"))
    baseline_variant = (baseline_direction, baseline_start_angle)

    candidates: list[tuple[str, float]] = []
    for direction in ("CCW", "CW"):
        for start_angle in (0.0, 90.0, 180.0, 270.0):
            candidates.append((direction, start_angle))

    scores: dict[tuple[str, float], dict[str, float]] = {}
    for direction, start_angle in candidates:
        variant_hs = _write_variant_header(
            base_hs=total_hs,
            tmp_path=tmp_path,
            direction=direction,
            start_angle=start_angle,
        )
        measured_variant = sirf.AcquisitionData(str(variant_hs))

        recon = _run_osem_reconstruction(source, measured_variant)
        recon_mid = _load_array(recon)[_load_array(recon).shape[0] // 2]
        recon_norm = _normalize_2d(recon_mid)
        identity_corr = _corr_2d(source_norm, recon_norm)
        flip_x_corr = _corr_2d(source_norm, np.flip(recon_norm, axis=1))

        scores[(direction, start_angle)] = {
            "mse": _forward_projection_mse(source, measured_variant),
            "identity_corr": identity_corr,
            "flip_x_corr": flip_x_corr,
            "scale_ratio": _equivalent_radius(recon_mid)
            / (_equivalent_radius(source_mid) + 1e-12),
        }

    baseline = scores[baseline_variant]
    better_candidates = [
        (variant, metrics)
        for variant, metrics in scores.items()
        if metrics["mse"] < baseline["mse"]
        and metrics["identity_corr"] > baseline["identity_corr"]
    ]

    assert better_candidates, (
        f"{case_backend.upper()} baseline mapping appears suboptimal. "
        f"Baseline {baseline_variant}: {baseline}. "
        f"All scores: {scores}"
    )


@pytest.mark.parametrize("case_backend", ["stir", "sirf"])
def test_enlargement_is_model_mismatch_not_grid_scaling(case_backend: str) -> None:
    source_hv, total_hs = _case_paths(case_backend)
    _require_paths(
        [source_hv, total_hs],
        hint=(
            f"Run examples/07{'A' if case_backend == 'stir' else 'B'}_"
            f"{case_backend}_adaptor_osem.py first"
        ),
    )

    source = sirf.ImageData(str(source_hv))
    measured = sirf.AcquisitionData(str(total_hs))

    # Reconstruct actual SIMIND projections with simple ray-tracing model.
    recon_measured = _run_osem_reconstruction(source, measured)

    # Reconstruct model-consistent synthetic data (same projector as recon model).
    acq_model = sirf.AcquisitionModelUsingRayTracingMatrix()
    acq_model.set_num_tangential_LORs(1)
    acq_model.set_up(measured, source)
    ideal_proj = acq_model.forward(source)
    recon_ideal = _run_osem_reconstruction(source, ideal_proj)

    source_mid = _load_array(source)[_load_array(source).shape[0] // 2]
    measured_mid = _load_array(recon_measured)[
        _load_array(recon_measured).shape[0] // 2
    ]
    ideal_mid = _load_array(recon_ideal)[_load_array(recon_ideal).shape[0] // 2]

    source_r = _equivalent_radius(source_mid)
    measured_r = _equivalent_radius(measured_mid)
    ideal_r = _equivalent_radius(ideal_mid)
    measured_ratio = measured_r / (source_r + 1e-12)
    ideal_ratio = ideal_r / (source_r + 1e-12)

    assert ideal_ratio <= 1.08, (
        f"{case_backend.upper()} model-consistent reconstruction unexpectedly changes "
        f"scale too much: source_r={source_r:.3f}, ideal_r={ideal_r:.3f}, "
        f"ratio={ideal_ratio:.3f}"
    )
    assert measured_ratio >= ideal_ratio + 0.10, (
        f"{case_backend.upper()} enlargement does not appear significantly larger than "
        f"model-consistent baseline. measured_ratio={measured_ratio:.3f}, "
        f"ideal_ratio={ideal_ratio:.3f}"
    )

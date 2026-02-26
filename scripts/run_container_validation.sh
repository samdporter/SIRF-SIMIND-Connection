#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker/compose.yaml"

# Backward-compatible env switch.
RUN_SIMIND_TESTS="${RUN_SIMIND_TESTS:-0}"

RUN_CORE=0
RUN_STIR=0
RUN_SIRF=0
RUN_PYTOMO=0
SELECTION_SET=0
DO_BUILD=1
REQUIRE_SIMIND=0
SIMIND_PATH=""
SIMIND_CONTAINER_DIR=""
DOCKER_PLATFORM=""

COMMON_EXCLUDE="not requires_cil and not requires_setr and not ci_skip and not requires_pytomography"
CORE_SUITE_MARKERS="$COMMON_EXCLUDE and not requires_simind and not requires_sirf and not requires_stir"
STIR_SUITE_MARKERS="$COMMON_EXCLUDE and not requires_simind and not requires_sirf"
SIRF_MARKERS="$COMMON_EXCLUDE and requires_sirf and not requires_simind"
SIMIND_MARKERS="requires_simind and not requires_cil and not requires_setr"

usage() {
    cat <<'USAGE'
Usage: bash scripts/run_container_validation.sh [options]

Options:
  --only-core          Run only non-backend core tests in python container.
  --only-stir          Run only STIR-focused test suite.
  --only-sirf          Run only SIRF-focused test suite.
  --only-pytomography  Run only PyTomography-focused test suite.
  --only-osem          Select STIR + SIRF + PyTomography groups.
  --with-simind        Include SIMIND-dependent example/test checks.
  --simind-path PATH   Path to SIMIND executable on host (default: ./simind/simind).
  --docker-platform P  Override Docker target platform (e.g. linux/amd64).
  --require-simind     Fail if --with-simind is requested but SIMIND is missing.
  --no-build           Skip docker image build step.
  -h, --help           Show this help text.

With no --only-* flags, all groups are run.
USAGE
}

select_only_mode() {
    if [[ "$SELECTION_SET" -eq 0 ]]; then
        RUN_CORE=0
        RUN_STIR=0
        RUN_SIRF=0
        RUN_PYTOMO=0
        SELECTION_SET=1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only-core)
            select_only_mode
            RUN_CORE=1
            ;;
        --only-stir)
            select_only_mode
            RUN_STIR=1
            ;;
        --only-sirf)
            select_only_mode
            RUN_SIRF=1
            ;;
        --only-pytomography)
            select_only_mode
            RUN_PYTOMO=1
            ;;
        --only-osem)
            select_only_mode
            RUN_STIR=1
            RUN_SIRF=1
            RUN_PYTOMO=1
            ;;
        --with-simind)
            RUN_SIMIND_TESTS=1
            ;;
        --simind-path)
            shift
            if [[ $# -eq 0 ]]; then
                echo "--simind-path requires an argument" >&2
                usage
                exit 2
            fi
            SIMIND_PATH="$1"
            ;;
        --require-simind)
            REQUIRE_SIMIND=1
            ;;
        --docker-platform)
            shift
            if [[ $# -eq 0 ]]; then
                echo "--docker-platform requires an argument" >&2
                usage
                exit 2
            fi
            DOCKER_PLATFORM="$1"
            ;;
        --no-build)
            DO_BUILD=0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
    shift
done

if [[ "$SELECTION_SET" -eq 0 ]]; then
    RUN_CORE=1
    RUN_STIR=1
    RUN_SIRF=1
    RUN_PYTOMO=1
fi

if [[ -z "$SIMIND_PATH" ]]; then
    SIMIND_PATH="$ROOT_DIR/simind/simind"
fi
if [[ "$SIMIND_PATH" != /* ]]; then
    SIMIND_PATH="$ROOT_DIR/$SIMIND_PATH"
fi
if [[ -x "$SIMIND_PATH" ]]; then
    if [[ "$SIMIND_PATH" == "$ROOT_DIR/"* ]]; then
        SIMIND_CONTAINER_BIN="/workspace/${SIMIND_PATH#$ROOT_DIR/}"
        SIMIND_CONTAINER_DIR="$(dirname "$SIMIND_CONTAINER_BIN")"
        SIMIND_AVAILABLE=1
    else
        echo "SIMIND executable is outside repo mount ('$SIMIND_PATH'); container cannot access it."
        SIMIND_AVAILABLE=0
    fi
else
    SIMIND_AVAILABLE=0
fi

if [[ -z "$DOCKER_PLATFORM" && "$SIMIND_AVAILABLE" -eq 1 ]]; then
    if command -v file >/dev/null 2>&1; then
        file_desc="$(file -b "$SIMIND_PATH" 2>/dev/null || true)"
        if [[ "$file_desc" == *"x86-64"* ]]; then
            DOCKER_PLATFORM="linux/amd64"
        elif [[ "$file_desc" == *"aarch64"* || "$file_desc" == *"ARM64"* ]]; then
            DOCKER_PLATFORM="linux/arm64"
        fi
    fi
fi

if [[ "$RUN_SIMIND_TESTS" == "1" && "$SIMIND_AVAILABLE" -eq 0 ]]; then
    if [[ "$REQUIRE_SIMIND" -eq 1 ]]; then
        echo "SIMIND executable not found at '$SIMIND_PATH'." >&2
        echo "Cannot run SIMIND-dependent checks." >&2
        exit 1
    fi
    echo "SIMIND not found at '$SIMIND_PATH'; skipping SIMIND-dependent checks."
    RUN_SIMIND_TESTS=0
fi

run_service() {
    local service="$1"
    shift
    local command="$*"
    if [[ -n "$SIMIND_CONTAINER_DIR" ]]; then
        command="export PATH=\"$SIMIND_CONTAINER_DIR:\$PATH\"; $command"
    fi
    echo
    echo "[$service] $command"
    if [[ -n "$DOCKER_PLATFORM" ]]; then
        DOCKER_DEFAULT_PLATFORM="$DOCKER_PLATFORM" docker compose -f "$COMPOSE_FILE" run --rm "$service" bash -lc "$command"
    else
        docker compose -f "$COMPOSE_FILE" run --rm "$service" bash -lc "$command"
    fi
}

SERVICES_TO_BUILD=()
if [[ "$RUN_CORE" -eq 1 ]]; then
    SERVICES_TO_BUILD+=("python")
fi
if [[ "$RUN_STIR" -eq 1 ]]; then
    SERVICES_TO_BUILD+=("stir")
fi
if [[ "$RUN_SIRF" -eq 1 ]]; then
    SERVICES_TO_BUILD+=("sirf")
fi
if [[ "$RUN_PYTOMO" -eq 1 ]]; then
    SERVICES_TO_BUILD+=("pytomography")
fi

if [[ "$DO_BUILD" -eq 1 ]]; then
    echo "[1/4] Building Docker images for selected services..."
    if [[ -n "$DOCKER_PLATFORM" ]]; then
        echo "Using Docker platform: $DOCKER_PLATFORM"
        DOCKER_DEFAULT_PLATFORM="$DOCKER_PLATFORM" docker compose -f "$COMPOSE_FILE" build "${SERVICES_TO_BUILD[@]}"
    else
        docker compose -f "$COMPOSE_FILE" build "${SERVICES_TO_BUILD[@]}"
    fi
else
    echo "[1/4] Skipping build step (--no-build)."
fi

echo "[2/4] Running selected non-SIMIND test suites..."

if [[ "$RUN_CORE" -eq 1 ]]; then
    run_service python "python -m pytest -m \"$CORE_SUITE_MARKERS\" -q"
fi

if [[ "$RUN_STIR" -eq 1 ]]; then
    run_service stir "python -m pytest -m \"$STIR_SUITE_MARKERS\" -q"
fi

if [[ "$RUN_SIRF" -eq 1 ]]; then
    run_service sirf "python -m pytest -m \"$SIRF_MARKERS\" -q"
fi

if [[ "$RUN_PYTOMO" -eq 1 ]]; then
    run_service pytomography "python -m pytest -m \"requires_pytomography and not requires_simind and not ci_skip\" -q"
fi

echo "[3/4] Verifying per-container import isolation..."

if [[ "$RUN_CORE" -eq 1 ]]; then
    run_service python "python -m pytest tests/test_container_library_isolation.py -q"
fi
if [[ "$RUN_STIR" -eq 1 ]]; then
    run_service stir "python -m pytest tests/test_container_library_isolation.py -q"
fi
if [[ "$RUN_SIRF" -eq 1 ]]; then
    run_service sirf "python -m pytest tests/test_container_library_isolation.py -q"
fi
if [[ "$RUN_PYTOMO" -eq 1 ]]; then
    run_service pytomography "python -m pytest tests/test_container_library_isolation.py -q"
fi

echo "[4/4] Handling SIMIND-dependent examples/tests..."
if [[ "$RUN_SIMIND_TESTS" == "1" ]]; then
    if [[ "$RUN_CORE" -eq 1 ]]; then
        run_service python "python examples/01_basic_simulation.py"
        run_service python "python examples/02_runtime_switch_comparison.py"
        run_service python "python examples/03_multi_window.py"
        run_service python "python examples/05_scattwin_vs_penetrate_comparison.py"
    fi
    if [[ "$RUN_STIR" -eq 1 ]]; then
        run_service stir "python examples/07A_stir_adaptor_osem.py"
    fi
    if [[ "$RUN_SIRF" -eq 1 ]]; then
        run_service sirf "python examples/07B_sirf_adaptor_osem.py"
        run_service sirf "python -m pytest -m \"$SIMIND_MARKERS\" -q"
        run_service sirf "python -m pytest tests/test_geometry_isolation_forward_projection.py -q"
    fi
    if [[ "$RUN_PYTOMO" -eq 1 ]]; then
        run_service pytomography "python examples/07C_pytomography_adaptor_osem.py"
    fi

    # Reads generated outputs from selected examples; missing outputs are skipped.
    if [[ "$RUN_CORE" -eq 1 ]]; then
        run_service python "python -m pytest tests/test_osem_geometry_diagnostics.py -q"
    elif [[ "$RUN_PYTOMO" -eq 1 ]]; then
        run_service pytomography "python -m pytest tests/test_osem_geometry_diagnostics.py -q"
    fi
else
    echo "Skipping SIMIND-dependent checks (set RUN_SIMIND_TESTS=1 or pass --with-simind to enable)."
fi

echo
echo "[done] Selected container validation completed."

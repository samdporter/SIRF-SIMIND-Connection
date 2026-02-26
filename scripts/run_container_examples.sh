#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker/compose.yaml"

RUN_CORE=0
RUN_STIR=0
RUN_SIRF=0
RUN_PYTOMO=0
SELECTION_SET=0
DO_BUILD=1
REQUIRE_SIMIND=0
SIMIND_PATH=""
SIMIND_CONTAINER_DIR=""
SIMIND_CONTAINER_BIN=""
DOCKER_PLATFORM=""

usage() {
    cat <<'USAGE'
Usage: bash scripts/run_container_examples.sh [options]

Options:
  --only-core          Run examples that do not require SIRF/STIR/PyTomography
                       (01-06 in python container; 01/02/03/05 need SIMIND).
  --only-stir          Run STIR OSEM example (07A).
  --only-sirf          Run only SIRF OSEM example (07B).
  --only-pytomography  Run only PyTomography OSEM example (07C).
  --only-osem          Run all OSEM examples (07A, 07B, 07C).
  --simind-path PATH   Path to SIMIND executable on host (default: ./simind/simind).
  --docker-platform P  Override Docker target platform (e.g. linux/amd64).
  --require-simind     Fail if SIMIND-dependent examples are selected but SIMIND is missing.
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

SIMIND_REQUIRED_GROUPS=0
if [[ "$RUN_CORE" -eq 1 || "$RUN_STIR" -eq 1 || "$RUN_SIRF" -eq 1 || "$RUN_PYTOMO" -eq 1 ]]; then
    SIMIND_REQUIRED_GROUPS=1
fi

if [[ "$SIMIND_REQUIRED_GROUPS" -eq 1 && "$SIMIND_AVAILABLE" -eq 0 ]]; then
    if [[ "$REQUIRE_SIMIND" -eq 1 ]]; then
        echo "SIMIND executable not found at '$SIMIND_PATH'." >&2
        echo "Cannot run SIMIND-dependent examples without SIMIND." >&2
        exit 1
    fi
    echo "SIMIND not found at '$SIMIND_PATH'; skipping SIMIND-dependent examples."
    RUN_STIR=0
    RUN_SIRF=0
    RUN_PYTOMO=0
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
    echo "[1/2] Building Docker images for selected services..."
    if [[ -n "$DOCKER_PLATFORM" ]]; then
        echo "Using Docker platform: $DOCKER_PLATFORM"
        DOCKER_DEFAULT_PLATFORM="$DOCKER_PLATFORM" docker compose -f "$COMPOSE_FILE" build "${SERVICES_TO_BUILD[@]}"
    else
        docker compose -f "$COMPOSE_FILE" build "${SERVICES_TO_BUILD[@]}"
    fi
else
    echo "[1/2] Skipping build step (--no-build)."
fi

echo "[2/2] Running selected examples..."

if [[ "$RUN_CORE" -eq 1 ]]; then
    if [[ "$SIMIND_AVAILABLE" -eq 1 ]]; then
        run_service python "python examples/01_basic_simulation.py"
        run_service python "python examples/02_runtime_switch_comparison.py"
        run_service python "python examples/03_multi_window.py"
        run_service python "python examples/05_scattwin_vs_penetrate_comparison.py"
    else
        echo
        echo "[python] Skipping 01/02/03/05 (SIMIND not found at '$SIMIND_PATH')."
    fi

    run_service python "python examples/04_custom_config.py"
    run_service python "python examples/06_schneider_density_conversion.py"
fi

if [[ "$RUN_STIR" -eq 1 ]]; then
    run_service stir "python examples/07A_stir_adaptor_osem.py"
fi

if [[ "$RUN_SIRF" -eq 1 ]]; then
    run_service sirf "python examples/07B_sirf_adaptor_osem.py"
fi

if [[ "$RUN_PYTOMO" -eq 1 ]]; then
    run_service pytomography "python examples/07C_pytomography_adaptor_osem.py"
fi

echo
echo "[done] Selected container examples completed."

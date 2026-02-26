#!/bin/bash
# Run all example scripts with error checking and progress tracking

set +e  # Don't exit on error - always run all examples

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

configure_simind() {
    if command -v simind >/dev/null 2>&1; then
        return 0
    fi

    local candidates=(
        "$SCRIPT_DIR/../simind"
        "$SCRIPT_DIR/../../simind"
        "$SCRIPT_DIR/../SIMIND"
        "$SCRIPT_DIR/../../SIMIND"
    )

    for candidate in "${candidates[@]}"; do
        if [[ -x "$candidate/simind" ]]; then
            export PATH="$candidate:$PATH"
            if [[ -d "$candidate/smc_dir" ]]; then
                export SMC_DIR="${candidate%/}/smc_dir/"
            fi
            echo "Configured SIMIND from $candidate"
            return 0
        fi
    done

    return 1
}

ensure_smc_dir() {
    local smc_path="${SMC_DIR%/}"
    if [[ -n "${SMC_DIR:-}" && -d "$smc_path" ]]; then
        export SMC_DIR="${smc_path}/"
        return 0
    fi

    local simind_path
    simind_path=$(command -v simind 2>/dev/null) || return 1
    local simind_dir
    simind_dir="$(dirname "$simind_path")"

    if [[ -d "$simind_dir/smc_dir" ]]; then
        export SMC_DIR="${simind_dir%/}/smc_dir/"
        echo "Detected SMC_DIR at $SMC_DIR"
        return 0
    fi

    return 1
}

# Parse arguments
BACKEND=""

# Parse command-line options
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)
            BACKEND="$2"
            if [[ "$BACKEND" != "sirf" && "$BACKEND" != "stir" ]]; then
                echo -e "${RED}Error: Invalid backend '$BACKEND'. Must be 'sirf' or 'stir'${NC}"
                exit 1
            fi
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--backend sirf|stir]"
            echo ""
            echo "Options:"
            echo "  --backend BACKEND   Force a specific backend (sirf or stir)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Run with auto-detected backend"
            echo "  $0 --backend stir           # Force STIR Python backend"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option '$1'${NC}"
            echo "Use --help for usage information"
            exit 1
            shift
            ;;
    esac
done

# Ensure SIMIND is available before running any examples
if ! configure_simind || ! command -v simind >/dev/null 2>&1; then
    echo -e "${RED}Error: SIMIND executable not found. Please install SIMIND or add it to your PATH.${NC}"
    exit 1
fi

if ! ensure_smc_dir; then
    echo -e "${RED}Error: Unable to locate the SIMIND smc_dir directory. Set SMC_DIR or ensure simind/smc_dir exists.${NC}"
    exit 1
fi

echo -e "${GREEN}SIMIND binary: $(command -v simind)${NC}"
echo -e "${GREEN}SMC_DIR: ${SMC_DIR}${NC}"

# Track results
PASSED=()
FAILED=()
SKIPPED=()
ERROR_LOGS=()

# Examples to run
EXAMPLES=(
    "01_basic_simulation.py"
    "02_runtime_switch_comparison.py"
    "03_multi_window.py"
    "04_custom_config.py"
    "05_scattwin_vs_penetrate_comparison.py"
    "06_schneider_density_conversion.py"
)

# Function to run a single example
run_example() {
    local example=$1
    local num=$2
    local total=$3
    local log_file="${example%.py}.log"

    echo -e "\n${BLUE}[${num}/${total}] Running ${example}...${NC}"

    # Check if file exists
    if [[ ! -f "$example" ]]; then
        echo -e "${YELLOW}⚠ SKIPPED: File not found${NC}"
        SKIPPED+=("$example")
        return 0
    fi

    # Build command with optional backend argument
    local cmd="python3 $example"

    # Add backend argument if specified
    if [[ -n "$BACKEND" ]]; then
        cmd="$cmd --backend $BACKEND"
    fi

    # Run the command and capture output
    if eval "$cmd" > "$log_file" 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        PASSED+=("$example")
        rm -f "$log_file"  # Remove log on success
        return 0
    else
        local exit_code=$?
        echo -e "${RED}✗ FAILED (exit code: ${exit_code})${NC}"
        echo -e "${RED}  Error log saved to: ${log_file}${NC}"
        FAILED+=("$example")
        ERROR_LOGS+=("$log_file")

        # Show last few lines of error
        echo -e "${RED}  Last 5 lines of error:${NC}"
        tail -n 5 "$log_file" | sed 's/^/    /'

        return 1
    fi
}

# Header
echo "======================================"
echo "Running SIRF-SIMIND-Connection Examples"
echo "======================================"
echo ""
echo "Total examples: ${#EXAMPLES[@]}"
if [[ -n "$BACKEND" ]]; then
    echo "Backend: $BACKEND (forced)"
else
    echo "Backend: auto-detect"
fi
echo ""

# Change to examples directory
cd "$(dirname "$0")"

# Run all examples
TOTAL=${#EXAMPLES[@]}
COUNT=1

for example in "${EXAMPLES[@]}"; do
    run_example "$example" "$COUNT" "$TOTAL"
    COUNT=$((COUNT + 1))
done

# Summary
echo ""
echo "======================================"
echo "Summary"
echo "======================================"
echo -e "${GREEN}Passed:  ${#PASSED[@]}${NC}"
echo -e "${RED}Failed:  ${#FAILED[@]}${NC}"
echo -e "${YELLOW}Skipped: ${#SKIPPED[@]}${NC}"

if [[ ${#PASSED[@]} -gt 0 ]]; then
    echo ""
    echo -e "${GREEN}✓ Passed examples:${NC}"
    for example in "${PASSED[@]}"; do
        echo "  - $example"
    done
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo -e "${RED}✗ Failed examples:${NC}"
    for i in "${!FAILED[@]}"; do
        echo "  - ${FAILED[$i]} (log: ${ERROR_LOGS[$i]})"
    done
    echo ""
    echo -e "${RED}Error Details:${NC}"
    echo "View full logs with: cat <logfile>"
fi

if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo ""
    echo -e "${YELLOW}⚠ Skipped examples:${NC}"
    for example in "${SKIPPED[@]}"; do
        echo "  - $example"
    done
fi

# Exit with appropriate code
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo -e "${RED}Some examples failed!${NC}"
    exit 1
else
    echo ""
    echo -e "${GREEN}All examples passed!${NC}"
    exit 0
fi

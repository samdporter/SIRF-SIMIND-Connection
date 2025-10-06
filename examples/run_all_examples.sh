#!/bin/bash
# Run all example scripts with error checking and progress tracking

set +e  # Don't exit on error - always run all examples

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
DICOM_FILE=""
if [[ -n "$1" ]]; then
    DICOM_FILE="$1"
fi

# Track results
PASSED=()
FAILED=()
SKIPPED=()
ERROR_LOGS=()

# Examples to run
EXAMPLES=(
    "01_basic_simulation.py"
    "02_dicom_conversion.py"
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

    # Special handling for DICOM example (02)
    if [[ "$example" == "02_dicom_conversion.py" ]]; then
        if [[ -z "$DICOM_FILE" ]]; then
            echo -e "${YELLOW}⚠ SKIPPED: No DICOM file provided (use: ./run_all_examples.sh <dicom_file>)${NC}"
            SKIPPED+=("$example")
            return 0
        fi
        # Run with DICOM file argument
        if python "$example" "$DICOM_FILE" > "$log_file" 2>&1; then
            echo -e "${GREEN}✓ PASSED${NC}"
            PASSED+=("$example")
            rm -f "$log_file"
            return 0
        else
            local exit_code=$?
            echo -e "${RED}✗ FAILED (exit code: ${exit_code})${NC}"
            echo -e "${RED}  Error log saved to: ${log_file}${NC}"
            FAILED+=("$example")
            ERROR_LOGS+=("$log_file")
            echo -e "${RED}  Last 5 lines of error:${NC}"
            tail -n 5 "$log_file" | sed 's/^/    /'
            return 1
        fi
    fi

    # Run the example and capture output
    if python "$example" > "$log_file" 2>&1; then
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
if [[ -n "$DICOM_FILE" ]]; then
    echo "DICOM file: $DICOM_FILE"
else
    echo "Note: Example 02 will be skipped (no DICOM file provided)"
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

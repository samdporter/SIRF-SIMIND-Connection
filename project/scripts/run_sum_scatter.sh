#!/bin/bash

# source bashrc
echo "Sourcing ~/.bashrc for environment setup..."
source "$HOME/.bashrc"
echo "Bashrc sourced."

# Ensure yq is installed (for this submitter script itself)
if ! command -v yq &> /dev/null
then
    echo "Error: yq is not found. Please ensure yq is installed in ~/.local/bin/ and that ~/.local/bin is accessible on the login node."
    echo "See https://github.com/mikefarah/yq#install for installation instructions."
    exit 1
fi

# Path to the config file
CONFIG_FILE="${CONFIG_FILE}"

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: CONFIG_FILE environment variable is not set in run_sum_scatter.sh"
    exit 1
fi

# Function to read a value from the config file using yq
get_config_value() {
    yq e "$1" "${CONFIG_FILE}"
}

# --- Load parameters from environment variables (passed by qsub) ---
ITERATION="${ITERATION}"
OUTPUT_DIR="${OUTPUT_DIR}"
PYTHON="${PYTHON}"
DATA_DIR="${DATA_DIR}"
SCRIPTS_DIR="${SCRIPTS_DIR}"

echo "Summing scatter outputs for iteration ${ITERATION}..."
echo "Input directory: ${OUTPUT_DIR}"

ITERATION_MINUS_ONE=$((ITERATION - 1))

# --- Load patterns from YAML and substitute dynamic values ---
SCATTER_PATTERN=$(get_config_value '.sum_scatter.scatter_pattern')
TOTAL_PATTERN=$(get_config_value '.sum_scatter.total_pattern')
IMAGE_PATTERN=$(get_config_value '.sum_scatter.image_pattern')

# Substitute placeholders with actual iteration numbers
SCATTER_PATTERN="${SCATTER_PATTERN//%ITER%/${ITERATION}}"
TOTAL_PATTERN="${TOTAL_PATTERN//%ITER%/${ITERATION}}"
IMAGE_PATTERN="${IMAGE_PATTERN//%PREV_ITER%/${ITERATION_MINUS_ONE}}"


$PYTHON ${SCRIPTS_DIR}/sum_scatter.py \
    --input_dir="${OUTPUT_DIR}" \
    --scatter_pattern="${SCATTER_PATTERN}" \
    --total_pattern="${TOTAL_PATTERN}" \
    --image_pattern="${IMAGE_PATTERN}" \
    --output_file_prefix="${OUTPUT_DIR}/mean_iter${ITERATION}" \
    --data_dir="${DATA_DIR}"
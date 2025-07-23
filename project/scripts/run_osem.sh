#!/bin/bash

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
    echo "Error: CONFIG_FILE environment variable is not set in run_osem.sh"
    exit 1
fi

# Function to read a value from the config file using yq
get_config_value() {
    yq e "$1" "${CONFIG_FILE}"
}

# --- Load parameters from environment variables (passed by qsub) ---
ITERATION="${ITERATION:-}" # Optional
DATA_DIR="${DATA_DIR}" # This will be the patient-specific DATA_DIR
OUTPUT_DIR="${OUTPUT_DIR}"
BASE_DIR="${BASE_DIR}"
PYTHON="${PYTHON}"

# --- Load static parameters from YAML ---
INITIAL_SUBSETS=$(get_config_value '.osem.initial_subsets')
INITIAL_EPOCHS=$(get_config_value '.osem.initial_epochs')
SMOOTHING=$(get_config_value '.osem.smoothing')

if [ -z "${ITERATION:-}" ]; then
    INDEX="0"
    ADDITIVE=""
else
    INDEX="${ITERATION}"
    # For iteration i, use the mean scatter file computed in the summing job.
    ADDITIVE="--additive_path=${OUTPUT_DIR}/mean_iter${ITERATION}_scatter.hs"
fi

$PYTHON "${BASE_DIR}/scripts/project_scripts/osem.py" \
    --num_subsets="${INITIAL_SUBSETS}" \
    --num_epochs="${INITIAL_EPOCHS}" \
    --data_path="${DATA_DIR}" \
    --output_path="${OUTPUT_DIR}" \
    ${ADDITIVE} \
    --smoothing="${SMOOTHING}" \
    --index="${INDEX}"
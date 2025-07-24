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

# --- Load parameters from environment variables (passed by qsub) ---
# These are dynamic per job/iteration
ITERATION="${ITERATION}"
DATA_DIR="${DATA_DIR}" # This will be the patient-specific DATA_DIR
OUTPUT_DIR="${OUTPUT_DIR}"
BASE_DIR="${BASE_DIR}"
SCRIPTS_DIR="${SCRIPTS_DIR}"
TOTAL_ACTIVITY="${TOTAL_ACTIVITY}" # This is the dynamically set SPECT activity
PYTHON="${PYTHON}"
INITIAL_SUBSETS="${INITIAL_SUBSETS}"
INITIAL_EPOCHS="${INITIAL_EPOCHS}"
SIMIND_INPUT_SMC="${SIMIND_INPUT_SMC}" # Passed from run_pipeline.sh
CONFIG_FILE="${CONFIG_FILE}"

# Construct a unique output prefix for each array task.
OUTPUT_PREFIX="output_iter${ITERATION}_${SGE_TASK_ID}"

# Determine the input image for simulation. For iteration i, use the reconstruction from iteration i-1.
if [ "${ITERATION}" -eq 1 ]; then
    INPUT_IMAGE="${OUTPUT_DIR}/recon_osem_i${INITIAL_EPOCHS}_s${INITIAL_SUBSETS}_smoothed_0.hv"
else
    PREV_ITER=$(( ITERATION - 1 ))
    INPUT_IMAGE="${OUTPUT_DIR}/recon_osem_i${INITIAL_EPOCHS}_s${INITIAL_SUBSETS}_smoothed_${PREV_ITER}.hv"
fi

# Run the simplified simulation script - much cleaner!
$PYTHON ${BASE_DIR}/scripts/simulation.py \
    --config_file="${CONFIG_FILE}" \
    --total_activity="${TOTAL_ACTIVITY}" \
    --data_dir="${DATA_DIR}" \
    --image_path="${INPUT_IMAGE}" \
    --output_prefix="${OUTPUT_PREFIX}" \
    --output_dir="${OUTPUT_DIR}" \
    --input_smc_file_path="${SIMIND_INPUT_SMC}" \
    --simind_parent_dir="${BASE_DIR}"
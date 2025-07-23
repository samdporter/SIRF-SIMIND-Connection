#!/bin/bash
set -euo pipefail # -e: exit on error, -u: exit on unset variable, -o pipefail: catch errors in pipes

echo "Sourcing ~/.bashrc for environment setup..."
source "$HOME/.bashrc"
echo "Bashrc sourced."

# Ensure yq is installed
if ! command -v yq &> /dev/null
then
    echo "Error: yq is not installed. Please install it to parse YAML config."
    echo "See https://github.com/mikefarah/yq#install for installation instructions."
    exit 1
fi

# --- Configuration Loading ---
# Expects the path to the YAML config file as the first argument
CONFIG_FILE="$1"

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Usage: $0 <path_to_config.yaml>"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Function to read a value from the config file using yq
# Usage: get_config_value <path_to_key_in_yaml>
get_config_value() {
    yq e "$1" "${CONFIG_FILE}"
}

# Load DEFAULT values from YAML first
PYTHON=$(get_config_value '.project.python_executable')
BASE_DIR=$(get_config_value '.project.base_dir')
SCRIPTS_DIR=$(get_config_value '.project.scripts_dir')
DEFAULT_DATA_DIR_FROM_CONFIG=$(get_config_value '.data.base_input_dir')
DEFAULT_OUTPUT_SUFFIX_FROM_CONFIG=$(get_config_value '.output.suffix') # Get the default suffix
BASE_OUTPUT_DIR=$(get_config_value '.output.base_output_dir')

INITIAL_SUBSETS=$(get_config_value '.osem.initial_subsets')
INITIAL_EPOCHS=$(get_config_value '.osem.initial_epochs')

# TOTAL_ACTIVITY will be passed as an environment variable from the wrapper script
# If not passed, use the default from config
TOTAL_ACTIVITY="${TOTAL_ACTIVITY:-$(get_config_value '.simulation.total_activity')}"

PHOTON_MULTIPLIER=$(get_config_value '.simulation.photon_multiplier')
NUM_ITERATIONS=$(get_config_value '.simulation.num_iterations')
NUM_ARRAY_JOBS=$(get_config_value '.simulation.num_array_jobs')
WINDOW_LOWER=$(get_config_value '.simulation.window_lower')
WINDOW_UPPER=$(get_config_value '.simulation.window_upper')
PHOTON_ENERGY=$(get_config_value '.simulation.photon_energy')

# --- Determine actual DATA_DIR and OUTPUT_SUFFIX ---
# Allow DATA_DIR to be passed as an env var (e.g., from wrapper)
# If not, use the default from config.
DATA_DIR="${DATA_DIR:-${DEFAULT_DATA_DIR_FROM_CONFIG}}"

# CRITICAL FIX: Allow OUTPUT_SUFFIX to be passed as an env var (e.g., from wrapper)
# If not, use the default from config.
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-${DEFAULT_OUTPUT_SUFFIX_FROM_CONFIG}}"

# Allow LOG_DIR to be passed as an env var (from wrapper script)
# If not provided, default to a generic logs directory
LOG_DIR="${LOG_DIR:-$HOME/job_outputs}"

OUTPUT_DIR="${BASE_OUTPUT_DIR}/${OUTPUT_SUFFIX}"
SIMIND_INPUT_SMC="${BASE_DIR}/$(get_config_value '.data.input_smc_filename')"

# --- Pre-run checks and setup ---
# Ensure the output directory exists
mkdir -p "${OUTPUT_DIR}"
# ensure the output directory is empty - BE CAREFUL WITH THIS!
# This `rm -rf` is now safe, as OUTPUT_DIR is unique per run.
rm -rf "${OUTPUT_DIR}"/*

# Ensure the log directory exists and is writable
mkdir -p "${LOG_DIR}"
if [[ ! -w "${LOG_DIR}" ]]; then
    echo "Error: Log directory ${LOG_DIR} is not writable"
    exit 1
fi

# Ensure the script is run from SCRIPTS_DIR
if [[ "$PWD" != "$SCRIPTS_DIR" ]]; then
    echo "Error: Please run this script from ${SCRIPTS_DIR}"
    echo "Current directory is $PWD"
    exit 1
fi

# Function to extract the numeric job ID from qsub output
extract_job_id() {
    echo "$1" | awk '{print $3}' | cut -d. -f1
}

echo "Submitting initial OSEM reconstruction for data: ${DATA_DIR}, output: ${OUTPUT_DIR}"
echo "Logs will be directed to: ${LOG_DIR}"
echo "Log directory contents before starting:"
ls -la "${LOG_DIR}" || echo "Log directory is empty or inaccessible"

INIT_OUT=$(qsub \
    -N init_osem_${OUTPUT_SUFFIX} \
    -cwd -l h_rt=04:00:00,tmem=32G,h_vmem=32G,tscratch=10G \
    -j y -R y \
    -o "${LOG_DIR}" \
    -v DATA_DIR="${DATA_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",SCRIPTS_DIR="${SCRIPTS_DIR}",INITIAL_SUBSETS="${INITIAL_SUBSETS}",INITIAL_EPOCHS="${INITIAL_EPOCHS}",BASE_DIR="${BASE_DIR}",PYTHON="${PYTHON}",CONFIG_FILE="${CONFIG_FILE}",LOG_DIR="${LOG_DIR}" \
    "${SCRIPTS_DIR}/run_osem.sh")
INIT_JOB=$(extract_job_id "${INIT_OUT}")
echo "Initial OSEM job submitted with ID ${INIT_JOB}."

# Initialize dependency chain with the initial job
PREV_JOB=${INIT_JOB}

# Loop over iterations using scheduler dependencies
for i in $(seq 1 ${NUM_ITERATIONS}); do
    echo "Submitting jobs for iteration ${i}"
    
    SIM_OUT=$(qsub \
        -N sim_iter_${i}_${OUTPUT_SUFFIX} \
        -cwd -l h_rt=168:00:00,tmem=16G,h_vmem=16G,tscratch=10G \
        -t 1-${NUM_ARRAY_JOBS} \
        -j y -R y \
        -o "${LOG_DIR}" \
        -hold_jid "${PREV_JOB}" \
        -v ITERATION="${i}",DATA_DIR="${DATA_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",BASE_DIR="${BASE_DIR}",TOTAL_ACTIVITY="${TOTAL_ACTIVITY}",PYTHON="${PYTHON}",INITIAL_SUBSETS="${INITIAL_SUBSETS}",INITIAL_EPOCHS="${INITIAL_EPOCHS}",PHOTON_MULTIPLIER="${PHOTON_MULTIPLIER}",WINDOW_LOWER="${WINDOW_LOWER}",WINDOW_UPPER="${WINDOW_UPPER}",PHOTON_ENERGY="${PHOTON_ENERGY}",SIMIND_INPUT_SMC="${SIMIND_INPUT_SMC}",CONFIG_FILE="${CONFIG_FILE}",LOG_DIR="${LOG_DIR}" \
        "${SCRIPTS_DIR}/run_simulation_array.sh")
    SIM_JOB=$(extract_job_id "${SIM_OUT}")
    echo "Simulation job for iteration ${i} submitted with ID ${SIM_JOB}."
    
    SUM_OUT=$(qsub \
        -N sum_iter_${i}_${OUTPUT_SUFFIX} \
        -cwd -l h_rt=04:00:00,tmem=32G,h_vmem=32G,tscratch=10G \
        -j y -R y \
        -o "${LOG_DIR}" \
        -hold_jid "${SIM_JOB}" \
        -v ITERATION="${i}",OUTPUT_DIR="${OUTPUT_DIR}",PYTHON="${PYTHON}",DATA_DIR="${DATA_DIR}",CONFIG_FILE="${CONFIG_FILE}",LOG_DIR="${LOG_DIR}",BASE_DIR="${BASE_DIR}" \
        "${SCRIPTS_DIR}/run_sum_scatter.sh")
    SUM_JOB=$(extract_job_id "${SUM_OUT}")
    echo "Scatter summing job for iteration ${i} submitted with ID ${SUM_JOB}."
    
    OSEM_OUT=$(qsub \
        -N osem_iter_${i}_${OUTPUT_SUFFIX} \
        -cwd -l h_rt=04:00:00,tmem=32G,h_vmem=32G,tscratch=10G \
        -j y -R y \
        -o "${LOG_DIR}" \
        -hold_jid "${SUM_JOB}" \
        -v ITERATION="${i}",DATA_DIR="${DATA_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",SCRIPTS_DIR="${SCRIPTS_DIR}",INITIAL_SUBSETS="${INITIAL_SUBSETS}",INITIAL_EPOCHS="${INITIAL_EPOCHS}",BASE_DIR="${BASE_DIR}",PYTHON="${PYTHON}",CONFIG_FILE="${CONFIG_FILE}",LOG_DIR="${LOG_DIR}" \
        "${SCRIPTS_DIR}/run_osem.sh")
    OSEM_JOB=$(extract_job_id "${OSEM_OUT}")
    echo "OSEM update job for iteration ${i} submitted with ID ${OSEM_JOB}."
    
    # Set dependency for next iteration: simulation job waits on the current OSEM update
    PREV_JOB=${OSEM_JOB}
done

echo "All jobs submitted successfully."
echo "Job logs will be available in: ${LOG_DIR}"
echo "Monitor job status with: qstat -u \$USER"
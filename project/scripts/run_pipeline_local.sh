#!/bin/bash
set -euo pipefail # -e: exit on error, -u: exit on unset variable, -o pipefail: catch errors in pipes

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

# Load configuration values from YAML (new format)
PYTHON="python3"  # Fixed default
BASE_DIR=$(get_config_value '.project.simind_parent_dir')
SCRIPTS_DIR="/home/sam/working/sirf_simind_connection/project/scripts"  # Fixed path
DEFAULT_DATA_DIR_FROM_CONFIG=$(get_config_value '.project.data_dir')
DEFAULT_OUTPUT_SUFFIX_FROM_CONFIG=$(get_config_value '.output.suffix')
BASE_OUTPUT_DIR=$(get_config_value '.output.base_output_dir')
SIMIND_INPUT_SMC=${BASE_DIR}/sirf_simind_connection/configs/input.smc

INITIAL_SUBSETS=$(get_config_value '.osem.initial_subsets')
INITIAL_EPOCHS=$(get_config_value '.osem.initial_epochs')

# TOTAL_ACTIVITY can be passed as an environment variable
# If not passed, use the default from config
TOTAL_ACTIVITY="${TOTAL_ACTIVITY:-$(get_config_value '.project.total_activity')}"

PHOTON_MULTIPLIER=$(get_config_value '.simulation.photon_multiplier')
# Try pipeline settings first, fallback to defaults
NUM_ITERATIONS=$(get_config_value '.pipeline.num_iterations' 2>/dev/null || echo "2")
NUM_ARRAY_JOBS=$(get_config_value '.pipeline.num_array_jobs' 2>/dev/null || echo "1")
WINDOW_LOWER=$(get_config_value '.simulation.window_lower')
WINDOW_UPPER=$(get_config_value '.simulation.window_upper')
PHOTON_ENERGY=$(get_config_value '.simulation.photon_energy')

# --- Determine actual DATA_DIR and OUTPUT_SUFFIX ---
# Allow DATA_DIR to be passed as an env var
# If not, use the default from config
DATA_DIR="${DATA_DIR:-${DEFAULT_DATA_DIR_FROM_CONFIG}}"

# Allow OUTPUT_SUFFIX to be passed as an env var
# If not, use the default from config
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-${DEFAULT_OUTPUT_SUFFIX_FROM_CONFIG}}"

OUTPUT_DIR="${BASE_OUTPUT_DIR}/${OUTPUT_SUFFIX}"

# --- Pre-run checks and setup ---
# Ensure the script is run from SCRIPTS_DIR
if [[ "$PWD" != "$SCRIPTS_DIR" ]]; then
    echo "Error: Please run this script from ${SCRIPTS_DIR}"
    echo "Current directory is $PWD"
    exit 1
fi

# === Prep output dir ===
echo "Setting up output directory: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
# This `rm -rf` is now safe, as OUTPUT_DIR is unique per run
rm -rf "${OUTPUT_DIR:?}/"*

echo "Configuration loaded from: ${CONFIG_FILE}"
echo "Base directory: ${BASE_DIR}"
echo "Scripts directory: ${SCRIPTS_DIR}"
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "SIMIND input SMC: ${SIMIND_INPUT_SMC}"
echo "Running ${NUM_ITERATIONS} iterations with ${NUM_ARRAY_JOBS} array jobs each"

# === Initial OSEM ===
echo "Running initial OSEM reconstruction..."
env \
  DATA_DIR="${DATA_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  SCRIPTS_DIR="${SCRIPTS_DIR}" \
  INITIAL_SUBSETS="${INITIAL_SUBSETS}" \
  INITIAL_EPOCHS="${INITIAL_EPOCHS}" \
  BASE_DIR="${BASE_DIR}" \
  PYTHON="${PYTHON}" \
  CONFIG_FILE="${CONFIG_FILE}" \
  "${SCRIPTS_DIR}/run_osem.sh"

# === Iterative loop ===
for (( i=1; i<=NUM_ITERATIONS; i++ )); do
  echo "=== Iteration ${i} ==="

  # Simulation "array" (1..NUM_ARRAY_JOBS)
  for (( task=1; task<=NUM_ARRAY_JOBS; task++ )); do
    echo "  Simulation task ${task}/${NUM_ARRAY_JOBS}..."
    env \
      ITERATION="${i}" \
      TASK_ID="${task}" \
      DATA_DIR="${DATA_DIR}" \
      OUTPUT_DIR="${OUTPUT_DIR}" \
      BASE_DIR="${BASE_DIR}" \
      TOTAL_ACTIVITY="${TOTAL_ACTIVITY}" \
      PYTHON="${PYTHON}" \
      INITIAL_SUBSETS="${INITIAL_SUBSETS}" \
      INITIAL_EPOCHS="${INITIAL_EPOCHS}" \
      PHOTON_MULTIPLIER="${PHOTON_MULTIPLIER}" \
      WINDOW_LOWER="${WINDOW_LOWER}" \
      WINDOW_UPPER="${WINDOW_UPPER}" \
      PHOTON_ENERGY="${PHOTON_ENERGY}" \
      SCRIPTS_DIR="${SCRIPTS_DIR}" \
      CONFIG_FILE="${CONFIG_FILE}" \
      SIMIND_INPUT_SMC="${SIMIND_INPUT_SMC}"\
      "${SCRIPTS_DIR}/run_simulation_array.sh"
  done

  # Sum scatter
  echo "  Summing scatter for iteration ${i}..."
  env \
    ITERATION="${i}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    PYTHON="${PYTHON}" \
    DATA_DIR="${DATA_DIR}" \
    BASE_DIR="${BASE_DIR}" \
    SCRIPTS_DIR="${SCRIPTS_DIR}" \
    CONFIG_FILE="${CONFIG_FILE}" \
    "${SCRIPTS_DIR}/run_sum_scatter.sh"

  # OSEM update
  echo "  Running OSEM update for iteration ${i}..."
  env \
    ITERATION="${i}" \
    DATA_DIR="${DATA_DIR}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    SCRIPTS_DIR="${SCRIPTS_DIR}" \
    INITIAL_SUBSETS="${INITIAL_SUBSETS}" \
    INITIAL_EPOCHS="${INITIAL_EPOCHS}" \
    BASE_DIR="${BASE_DIR}" \
    PYTHON="${PYTHON}" \
    CONFIG_FILE="${CONFIG_FILE}" \
    "${SCRIPTS_DIR}/run_osem.sh"
done

echo "All iterations complete."
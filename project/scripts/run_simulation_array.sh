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

# Path to the config file (passed as an environment variable from qsub)
CONFIG_FILE="${CONFIG_FILE}"

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: CONFIG_FILE environment variable is not set in run_simulation_array.sh"
    exit 1
fi

# Function to read a value from the config file using yq
get_config_value() {
    yq e "$1" "${CONFIG_FILE}"
}

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

# --- Load static simulation parameters from YAML ---
TIME_PER_PROJECTION=$(get_config_value '.simulation.time_per_projection')
PHOTON_MULTIPLIER=$(get_config_value '.simulation.photon_multiplier')
PHOTON_ENERGY=$(get_config_value '.simulation.photon_energy')
WINDOW_LOWER=$(get_config_value '.simulation.window_lower')
WINDOW_UPPER=$(get_config_value '.simulation.window_upper')
SOURCE_TYPE=$(get_config_value '.simulation.source_type')
COLLIMATOR=$(get_config_value '.simulation.collimator')
KEV_PER_CHANNEL=$(get_config_value '.simulation.kev_per_channel')
MAX_ENERGY=$(get_config_value '.simulation.max_energy')
SCORING_ROUTINE=$(get_config_value '.simulation.scoring_routine')
COLLIMATOR_ROUTINE=$(get_config_value '.simulation.collimator_routine')
PHOTON_DIRECTION=$(get_config_value '.simulation.photon_direction')
CRYSTAL_THICKNESS=$(get_config_value '.simulation.crystal_thickness')
CRYSTAL_HALF_LENGTH_RADIUS=$(get_config_value '.simulation.crystal_half_length_radius')
CRYSTAL_HALF_WIDTH=$(get_config_value '.simulation.crystal_half_width')
HALF_LIFE=$(get_config_value '.simulation.half_life')

MU_MAP_PATH="${DATA_DIR}/$(get_config_value '.data.mu_map_filename')"
MEASURED_DATA_PATH="${DATA_DIR}/$(get_config_value '.data.measured_data_filename')"

# Construct a unique output prefix for each array task.
OUTPUT_PREFIX="output_iter${ITERATION}_${SGE_TASK_ID}"

# Determine the input image for simulation. For iteration i, use the reconstruction from iteration i-1.
if [ "${ITERATION}" -eq 1 ]; then
    INPUT_IMAGE="${OUTPUT_DIR}/recon_osem_i${INITIAL_EPOCHS}_s${INITIAL_SUBSETS}_smoothed_0.hv"
else
    PREV_ITER=$(( ITERATION - 1 ))
    INPUT_IMAGE="${OUTPUT_DIR}/recon_osem_i${INITIAL_EPOCHS}_s${INITIAL_SUBSETS}_smoothed_${PREV_ITER}.hv"
fi

$PYTHON ${BASE_DIR}/scripts/simulation.py \
    --total_activity="${TOTAL_ACTIVITY}" \
    --time_per_projection="${TIME_PER_PROJECTION}" \
    --photon_multiplier="${PHOTON_MULTIPLIER}" \
    --photopeak_energy="${PHOTON_ENERGY}" \
    --window_lower="${WINDOW_LOWER}" \
    --window_upper="${WINDOW_UPPER}" \
    --source_type="${SOURCE_TYPE}" \
    --collimator="${COLLIMATOR}" \
    --kev_per_channel="${KEV_PER_CHANNEL}" \
    --max_energy="${MAX_ENERGY}" \
    --mu_map_path="${MU_MAP_PATH}" \
    --image_path="${INPUT_IMAGE}" \
    --measured_data_path="${MEASURED_DATA_PATH}" \
    --output_prefix="${OUTPUT_PREFIX}" \
    --output_dir="${OUTPUT_DIR}" \
    --input_smc_file_path="${SIMIND_INPUT_SMC}" \
    --simind_parent_dir="${BASE_DIR}" \
    --scoring_routine="${SCORING_ROUTINE}" \
    --collimator_routine="${COLLIMATOR_ROUTINE}" \
    --photon_direction="${PHOTON_DIRECTION}" \
    --crystal_thickness="${CRYSTAL_THICKNESS}" \
    --crystal_half_length_radius="${CRYSTAL_HALF_LENGTH_RADIUS}" \
    --crystal_half_width="${CRYSTAL_HALF_WIDTH}" \
    --half_life="${HALF_LIFE}"
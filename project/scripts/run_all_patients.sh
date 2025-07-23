#!/bin/bash

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

# --- Configuration for this wrapper script ---
# Define the path to your main YAML config file
MAIN_CONFIG_FILE="/home/sporter/synergistic_Y90/MIC23_STIR_SIMIND/scripts/project_scripts/config_patient.yaml"
PIPELINE_SCRIPT="/home/sporter/synergistic_Y90/MIC23_STIR_SIMIND/scripts/project_scripts/run_pipeline.sh"

# Base directory for all job outputs
JOB_OUTPUTS_BASE="/home/sporter/job_outputs"

if [[ ! -f "$MAIN_CONFIG_FILE" ]]; then
    echo "Error: Main config file not found: ${MAIN_CONFIG_FILE}"
    exit 1
fi

# Function to read a value from the main config file using yq
get_main_config_value() {
    yq e "$1" "${MAIN_CONFIG_FILE}"
}

# Load patient processing settings from the main config
PATIENT_DATA_ROOT=$(get_main_config_value '.patient_processing.patient_data_root')
ACTIVITIES_CSV_FILENAME=$(get_main_config_value '.patient_processing.activities_csv_filename')

# --- Function to extract the single SPECT Activity ---
# This function takes the path to activities.csv as an argument
# It returns the single SPECT activity from the second column, skipping the header.
get_single_spect_activity() {
    local csv_file="$1"

    if [[ ! -f "$csv_file" ]]; then
        echo "Error: activities.csv not found at $csv_file" >&2
        return 1
    fi

    # Use awk to parse CSV, skipping the header (NR>1) and printing the second column.
    # We expect only one data row.
    awk -F',' 'NR==2 {print $2}' "$csv_file"
}

# --- Function to create and setup log directory ---
setup_log_directory() {
    local patient_id="$1"
    local log_dir="${JOB_OUTPUTS_BASE}/${patient_id}"
    
    # Create the patient-specific log directory if it doesn't exist
    if [[ ! -d "$log_dir" ]]; then
        echo "  Creating log directory: ${log_dir}" >&2  # Send to stderr to avoid capture
        mkdir -p "$log_dir"
        if [[ $? -ne 0 ]]; then
            echo "Error: Failed to create log directory: ${log_dir}" >&2
            return 1
        fi
    else
        echo "  Using existing log directory: ${log_dir}" >&2  # Send to stderr to avoid capture
    fi
    
    echo "$log_dir"  # This is the only output that should be captured
}

# --- Main loop ---
# Find all directories that start with "sirt" within PATIENT_DATA_ROOT
find "${PATIENT_DATA_ROOT}" -type d -name "sirt*" | while IFS= read -r patient_specific_data_dir; do
    echo "----------------------------------------------------"
    echo "Processing patient-specific data directory: ${patient_specific_data_dir}"

    # Extract the directory name (e.g., sirt10) for use as the output folder name
    PATIENT_ID=$(basename "${patient_specific_data_dir}")

    # Setup patient-specific log directory
    PATIENT_LOG_DIR=$(setup_log_directory "${PATIENT_ID}")
    if [[ $? -ne 0 ]]; then
        echo "Warning: Failed to setup log directory for ${PATIENT_ID}. Skipping this patient."
        continue
    fi

    # Construct the path to activities.csv
    activities_csv="${patient_specific_data_dir}/${ACTIVITIES_CSV_FILENAME}"

    # Get the single SPECT activity from the CSV
    SPECT_ACTIVITY=$(get_single_spect_activity "${activities_csv}")

    if [[ -z "$SPECT_ACTIVITY" ]]; then
        echo "Warning: No SPECT Activity found in ${activities_csv} (expected one). Skipping this patient."
        continue # Skip to the next patient directory
    fi

    echo "  Found SPECT Activity: ${SPECT_ACTIVITY}"

    # Determine a suitable unique output suffix for this patient
    # The PATIENT_ID is now directly the 'sirtXX' directory name
    # We no longer need a measurement index, as there's only one.
    RUN_SUFFIX="${PATIENT_ID}" # Using sirtXX directly as the suffix

    echo "  Running pipeline with:"
    echo "    TOTAL_ACTIVITY=${SPECT_ACTIVITY}"
    echo "    DATA_DIR=${patient_specific_data_dir}"
    echo "    OUTPUT_SUFFIX=${RUN_SUFFIX}"
    echo "    LOG_DIR=${PATIENT_LOG_DIR}"

    # Create a timestamp for the current run
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    MAIN_LOG_FILE="${PATIENT_LOG_DIR}/pipeline_${PATIENT_ID}_${TIMESTAMP}.log"

    echo "  Logs will be saved to: ${MAIN_LOG_FILE}"

    # Call run_pipeline.sh with the config file and override variables
    # These will be picked up by run_pipeline.sh and its sub-scripts.
    # Redirect both stdout and stderr to the patient-specific log file
    {
        echo "=== Pipeline execution started at $(date) ==="
        echo "Patient ID: ${PATIENT_ID}"
        echo "TOTAL_ACTIVITY: ${SPECT_ACTIVITY}"
        echo "DATA_DIR: ${patient_specific_data_dir}/SPECT"
        echo "OUTPUT_SUFFIX: ${RUN_SUFFIX}"
        echo "LOG_DIR: ${PATIENT_LOG_DIR}"
        echo "=============================================="
        echo ""
        
        CONFIG_FILE="${MAIN_CONFIG_FILE}" \
        TOTAL_ACTIVITY="${SPECT_ACTIVITY}" \
        DATA_DIR="${patient_specific_data_dir}/SPECT" \
        OUTPUT_SUFFIX="${RUN_SUFFIX}" \
        LOG_DIR="${PATIENT_LOG_DIR}" \
        bash "${PIPELINE_SCRIPT}" "${MAIN_CONFIG_FILE}"
        
        echo ""
        echo "=== Pipeline execution completed at $(date) ==="
    } 2>&1 | tee "${MAIN_LOG_FILE}"

    # Check the exit status of the pipeline
    pipeline_exit_status=${PIPESTATUS[0]}
    
    if [[ $pipeline_exit_status -eq 0 ]]; then
        echo "  ✓ Successfully finished processing ${patient_specific_data_dir}"
    else
        echo "  ✗ Pipeline failed for ${patient_specific_data_dir} (exit code: ${pipeline_exit_status})"
    fi
    
    echo "  Log saved to: ${MAIN_LOG_FILE}"
    echo "----------------------------------------------------"
done

echo "All patient data directories processed."
echo "Logs are available in: ${JOB_OUTPUTS_BASE}/"
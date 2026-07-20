#!/bin/bash

# List of arguments to run the Python script with
ARGS=(
    "kitchen_pnp/PnPCounterToSink/mg/2024-05-04-22-14-06_and_2024-05-07-07-40-17/demo_gentex_im128_randcams_500.hdf5"
    # "kitchen_pnp/PnPCounterToSink/mg/2024-05-04-22-14-06_and_2024-05-07-07-40-17/demo_gentex_im128_randcams.hdf5"
)

# Maximum number of concurrent jobs
MAX_JOBS=16

# Path to your Python script
PYTHON_SCRIPT="robocasa/scripts/libero_dataset_states_to_obs_single.py"

# Function to run the Python script with a given argument
run_script() {
    local arg=$1
    echo "Starting job with dataset: $arg"
    python $PYTHON_SCRIPT --dataset /iliad/u/jenseng/xembod/robocasa_xembod/datasets/v0.1/single_stage/$arg --generative_textures --randomize_cameras
}

# Main loop to manage parallel jobs
for arg in "${ARGS[@]}"; do
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep 1  # Wait until there's an available slot
    done
    run_script "$arg"
done

# Wait for all background jobs to complete
wait
echo "All jobs completed."
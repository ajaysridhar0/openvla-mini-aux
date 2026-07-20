#!/bin/bash
#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gpus-per-node=a40:1
#SBATCH --output=../slurm_logs/%A.out
#SBATCH --error=../slurm_logs/%A.err
#SBATCH --job-name="aux_mg_panda_sink_eval"

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "Working directory = "$SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate robocasa
cd /iliad/u/jenseng/xembod/openvla-mini-aux/
export PYTHONPATH=/iliad/u/jenseng/xembod/openvla-mini-aux

# --- CONFIGURATION ---
type="aux"
task_short="turn_on_sink"
robot="panda_og"
TASK="TurnOnSinkFaucet"

MODEL="${type}--${robot}+mg_${robot}_${task_short}"
CHECKPOINT_DIR="runs/${MODEL}/checkpoints"

# --- AUTOMATIC CHECKPOINT FINDING ---
# This finds all .pt files in the directory and stores them in an array
# We use 'basename' to just get the filename as your loop expects
mapfile -t CHECKPOINTS < <(ls "${CHECKPOINT_DIR}"/*.pt 2>/dev/null | xargs -n 1 basename)

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "Error: No checkpoints found in ${CHECKPOINT_DIR}"
    exit 1
fi

echo "Found ${#CHECKPOINTS[@]} checkpoints: ${CHECKPOINTS[*]}"

# --- TASK SETTINGS ---
AUX_TASK_TYPES=null
AUX_ABRV="act"
UNNORM_KEY="mg_${robot}_${task_short}"

if [ "$TASK" = "PnPCounterToSink" ]; then
    MAX_STEPS=600
elif [ "$TASK" = "PnPSinkToCounter" ]; then
    MAX_STEPS=650
else
    MAX_STEPS=500
fi

ACT_HORIZON=8

# --- EVALUATION LOOP ---
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CHECKPOINT_PATH="${CHECKPOINT_DIR}/${CHECKPOINT}"
    
    echo "Starting evaluation for: ${CHECKPOINT}"

    python experiments/robot/libero/run_robocasa_eval_new.py \
        --pretrained_checkpoint ${CHECKPOINT_PATH} \
        --robot PandaOmron \
        --gripper_types PandaGripper \
        --task ${TASK} \
        --unnorm_key ${UNNORM_KEY} \
        --max_steps ${MAX_STEPS} \
        --act_horizon ${ACT_HORIZON} \
        --rollout_dir "experiments/rollouts/${TASK}/PandaOGGripper/${MODEL}/STEP/${AUX_ABRV}" \
        --aux_task_types ${AUX_TASK_TYPES} &
done

wait
echo "All evaluations complete."
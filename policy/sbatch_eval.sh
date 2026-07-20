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
#SBATCH --job-name="aux_xembod_50k"

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "Working directory = "$SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate robocasa
cd /iliad/u/jenseng/xembod/openvla-mini-aux/
export PYTHONPATH=/iliad/u/jenseng/xembod/openvla-mini-aux

MODEL="aux--mg_flip_mug"
# MODEL="aux--mg_pnp_lite"
CHECKPOINT="step-050000-epoch-16-loss=0.2363.pt"
TASK="FlipMugUpright"
# TASK="PnPCounterToSink"
AUX_TASK_TYPES=null
AUX_ABRV="act"
UNNORM_KEY="mg_flip_mug"

if [ "$TASK" = "PnPCounterToSink" ]; then
    MAX_STEPS=600
elif [ "$TASK" = "PnPSinkToCounter" ]; then
    MAX_STEPS=650
else
    MAX_STEPS=500
fi

CHECKPOINT_PATH=runs/${MODEL}/checkpoints/${CHECKPOINT}
ACT_HORIZON=8


python experiments/robot/libero/run_robocasa_eval_new.py \
    --pretrained_checkpoint ${CHECKPOINT_PATH} \
    --robot Kinova3Omron \
    --task ${TASK} \
    --unnorm_key ${UNNORM_KEY} \
    --max_steps ${MAX_STEPS} \
    --act_horizon ${ACT_HORIZON} \
    --rollout_dir "experiments/rollouts/${TASK}/Kinova/${MODEL}/STEP/${AUX_ABRV}" \
    --aux_task_types ${AUX_TASK_TYPES} &

python experiments/robot/libero/run_robocasa_eval_new.py \
    --pretrained_checkpoint ${CHECKPOINT_PATH} \
    --robot UR5eOmron \
    --task ${TASK} \
    --unnorm_key ${UNNORM_KEY} \
    --max_steps ${MAX_STEPS} \
    --act_horizon ${ACT_HORIZON} \
    --rollout_dir "experiments/rollouts/${TASK}/UR5e/${MODEL}/STEP/${AUX_ABRV}" \
    --aux_task_types ${AUX_TASK_TYPES} &

python experiments/robot/libero/run_robocasa_eval_new.py \
    --pretrained_checkpoint ${CHECKPOINT_PATH} \
    --robot IIWAOmron \
    --task ${TASK} \
    --unnorm_key ${UNNORM_KEY} \
    --max_steps ${MAX_STEPS} \
    --act_horizon ${ACT_HORIZON} \
    --rollout_dir "experiments/rollouts/${TASK}/IIWA/${MODEL}/STEP/${AUX_ABRV}" \
    --aux_task_types ${AUX_TASK_TYPES} &

wait
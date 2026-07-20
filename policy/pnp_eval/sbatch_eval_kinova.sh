#!/bin/bash
#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --gpus-per-node=a5000:1
#SBATCH --output=../slurm_logs/%A.out
#SBATCH --error=../slurm_logs/%A.err
#SBATCH --job-name="base_kinova_30k_cts"

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "Working directory = "$SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate robocasa
cd /iliad/u/jenseng/xembod/openvla-mini-aux/
export PYTHONPATH=/iliad/u/jenseng/xembod/openvla-mini-aux

MODEL="base--mg_kinova_pnp_lite"
# MODEL="aux--mg_kinova_pnp_lite"
# MODEL="aux--mg_pnp"
# MODEL="base--mg_pnp"
CHECKPOINT="step-030000-epoch-28-loss=0.3381.pt"
AUX_TASK_TYPES=null
AUX_ABRV="act"
UNNORM_KEY="mg_kinova_pnp_lite"
# UNNORM_KEY="mg_pnp"

CHECKPOINT_PATH=runs/${MODEL}/checkpoints/${CHECKPOINT}
ACT_HORIZON=8

python experiments/robot/libero/run_robocasa_eval_new.py \
    --pretrained_checkpoint ${CHECKPOINT_PATH} \
    --robot Kinova3Omron \
    --task PnPCounterToSink \
    --unnorm_key ${UNNORM_KEY} \
    --max_steps 600 \
    --act_horizon ${ACT_HORIZON} \
    --rollout_dir "experiments/rollouts/PnPCounterToSink/Kinova/${MODEL}/STEP/${AUX_ABRV}" \
    --aux_task_types ${AUX_TASK_TYPES} &

# python experiments/robot/libero/run_robocasa_eval_new.py \
#     --pretrained_checkpoint ${CHECKPOINT_PATH} \
#     --robot Kinova3Omron \
#     --task PnPSinkToCounter \
#     --unnorm_key ${UNNORM_KEY} \
#     --max_steps 650 \
#     --act_horizon ${ACT_HORIZON} \
#     --rollout_dir "experiments/rollouts/PnPSinkToCounter/Kinova/${MODEL}/STEP/${AUX_ABRV}" \
#     --aux_task_types ${AUX_TASK_TYPES} &


wait
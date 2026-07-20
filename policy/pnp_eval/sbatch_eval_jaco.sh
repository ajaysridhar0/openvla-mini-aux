#!/bin/bash
#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=a5000:1
#SBATCH --output=../slurm_logs/%A.out
#SBATCH --error=../slurm_logs/%A.err
#SBATCH --job-name="aux_jaco_mg_1k"

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "Working directory = "$SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate robocasa
cd /iliad/u/jenseng/xembod/openvla-mini-aux/
export PYTHONPATH=/iliad/u/jenseng/xembod/openvla-mini-aux

# MODEL="chain--jaco+mg_pnp_lite"
MODEL="aux--jaco+mg_pnp"
# MODEL="ee--jaco+mg_pnp_lite"
# MODEL="aux_2--jaco+mg_pnp"
# MODEL="base--jaco_pnp"
# MODEL="aux--jaco_pnp"
# MODEL="aux--mg_pnp"
# MODEL="aux--mg_pnp_lite"
# MODEL="base--mg_pnp_lite"
UNNORM_KEY="mg_pnp"
# UNNORM_KEY="jaco_pnp"


CHECKPOINT="step-001000-epoch-04-loss=0.2078.pt"
AUX_TASK_TYPES=null
AUX_ABRV="act"
# AUX_TASK_TYPES="bbox->ee_pose_2D->low_level_motion"
# AUX_ABRV="chain"
CENTER_CROP=False

CHECKPOINT_PATH=runs_pnp/${MODEL}/checkpoints/${CHECKPOINT}
ACT_HORIZON=8

python experiments/robot/libero/run_robocasa_eval_new.py \
    --pretrained_checkpoint ${CHECKPOINT_PATH} \
    --robot JacoOmron \
    --task PnPCounterToSink \
    --unnorm_key ${UNNORM_KEY} \
    --max_steps 600 \
    --act_horizon ${ACT_HORIZON} \
    --center_crop ${CENTER_CROP} \
    --rollout_dir "experiments/rollouts/PnPCounterToSink/Jaco/${MODEL}/STEP/${AUX_ABRV}" \
    --aux_task_types ${AUX_TASK_TYPES} &

python experiments/robot/libero/run_robocasa_eval_new.py \
    --pretrained_checkpoint ${CHECKPOINT_PATH} \
    --robot JacoOmron \
    --task PnPSinkToCounter \
    --unnorm_key ${UNNORM_KEY} \
    --max_steps 650 \
    --act_horizon ${ACT_HORIZON} \
    --center_crop ${CENTER_CROP} \
    --rollout_dir "experiments/rollouts/PnPSinkToCounter/Jaco/${MODEL}/STEP/${AUX_ABRV}" \
    --aux_task_types ${AUX_TASK_TYPES} &


wait
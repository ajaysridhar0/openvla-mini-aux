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
#SBATCH --job-name="aux_panda_mg_3k"

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "Working directory = "$SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate robocasa
cd /iliad/u/jenseng/xembod/openvla-mini-aux/
export PYTHONPATH=/iliad/u/jenseng/xembod/openvla-mini-aux

MODEL="aux--panda+mg_pnp"
# MODEL="chain--panda+mg_pnp_lite"
# MODEL="aux--panda+mg_pnp_lite--image_aug"
# MODEL="base--panda_pnp"
# MODEL="aux--panda_pnp"
# MODEL="aux--mg_pnp"
# MODEL="base--mg_pnp"
# MODEL="aux--mg_pnp_lite"
# MODEL="base--mg_pnp_lite"

# MODEL=base--panda_pnp--image_aug
# UNNORM_KEY="mg_pnp"
UNNORM_KEY="mg_pnp"
# UNNORM_KEY="panda_pnp"

CHECKPOINT="step-003000-epoch-11-loss=0.1675.pt"
AUX_TASK_TYPES=null
# AUX_TASK_TYPES="bbox->ee_pose_2D->low_level_motion"
AUX_ABRV="act"
CENTER_CROP=False

CHECKPOINT_PATH=runs_pnp/${MODEL}/checkpoints/${CHECKPOINT}
ACT_HORIZON=8

python experiments/robot/libero/run_robocasa_eval_new.py \
    --pretrained_checkpoint ${CHECKPOINT_PATH} \
    --robot PandaOmron \
    --task PnPCounterToSink \
    --unnorm_key ${UNNORM_KEY} \
    --max_steps 600 \
    --act_horizon ${ACT_HORIZON} \
    --center_crop ${CENTER_CROP} \
    --rollout_dir "experiments/rollouts/PnPCounterToSink/Panda/${MODEL}/STEP/${AUX_ABRV}" \
    --aux_task_types ${AUX_TASK_TYPES} &

python experiments/robot/libero/run_robocasa_eval_new.py \
    --pretrained_checkpoint ${CHECKPOINT_PATH} \
    --robot PandaOmron \
    --task PnPSinkToCounter \
    --unnorm_key ${UNNORM_KEY} \
    --max_steps 650 \
    --act_horizon ${ACT_HORIZON} \
    --center_crop ${CENTER_CROP} \
    --rollout_dir "experiments/rollouts/PnPSinkToCounter/Panda/${MODEL}/STEP/${AUX_ABRV}" \
    --aux_task_types ${AUX_TASK_TYPES} &


wait
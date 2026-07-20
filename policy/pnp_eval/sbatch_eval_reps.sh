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
#SBATCH --job-name="aux_rep_panda_3k_stc"

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "Working directory = "$SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate robocasa
cd /iliad/u/jenseng/xembod/openvla-mini-aux/
export PYTHONPATH=/iliad/u/jenseng/xembod/openvla-mini-aux

# MODEL="aux--jaco+mg_pnp"
MODEL="aux--panda+mg_pnp"
ROBOT=Panda
CHECKPOINT="step-003000-epoch-11-loss=0.1675.pt"
# TASK="PnPCounterToSink"
TASK="PnPSinkToCounter"

if [ "$TASK" = "PnPCounterToSink" ]; then
    MAX_STEPS=600
else
    MAX_STEPS=650
fi

CHECKPOINT_PATH=runs_pnp/${MODEL}/checkpoints/${CHECKPOINT}
ACT_HORIZON=8


python experiments/robot/libero/run_robocasa_eval_new.py \
    --pretrained_checkpoint ${CHECKPOINT_PATH} \
    --robot "${ROBOT}Omron" \
    --task ${TASK} \
    --unnorm_key mg_pnp \
    --max_steps ${MAX_STEPS} \
    --act_horizon ${ACT_HORIZON} \
    --rollout_dir "experiments/rollouts/${TASK}/${ROBOT}/${MODEL}/STEP/lm" \
    --aux_task_types "low_level_motion" &

python experiments/robot/libero/run_robocasa_eval_new.py \
    --pretrained_checkpoint ${CHECKPOINT_PATH} \
    --robot "${ROBOT}Omron" \
    --task ${TASK} \
    --unnorm_key mg_pnp \
    --max_steps ${MAX_STEPS} \
    --act_horizon ${ACT_HORIZON} \
    --rollout_dir "experiments/rollouts/${TASK}/${ROBOT}/${MODEL}/STEP/bbox" \
    --aux_task_types "bbox" &

python experiments/robot/libero/run_robocasa_eval_new.py \
    --pretrained_checkpoint ${CHECKPOINT_PATH} \
    --robot "${ROBOT}Omron" \
    --task ${TASK} \
    --unnorm_key mg_pnp \
    --max_steps ${MAX_STEPS} \
    --act_horizon ${ACT_HORIZON} \
    --rollout_dir "experiments/rollouts/${TASK}/${ROBOT}/${MODEL}/STEP/ee" \
    --aux_task_types "ee_pose_2D" &

wait
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
#SBATCH --job-name="chain_xembod_lite_50k_stc"

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "Working directory = "$SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate robocasa
cd /iliad/u/jenseng/xembod/openvla-mini-aux/
export PYTHONPATH=/iliad/u/jenseng/xembod/openvla-mini-aux

MODEL="chain--mg_pnp_lite"
CHECKPOINT="step-050000-epoch-15-loss=0.1389.pt"
# TASK="PnPCounterToSink"
TASK="PnPSinkToCounter"
AUX_TASK_TYPES=null
AUX_ABRV="act"
UNNORM_KEY="mg_pnp_lite"

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
    --robot PandaOmron \
    --task ${TASK} \
    --unnorm_key ${UNNORM_KEY} \
    --max_steps ${MAX_STEPS} \
    --act_horizon ${ACT_HORIZON} \
    --rollout_dir "experiments/rollouts/${TASK}/Panda/${MODEL}/STEP/${AUX_ABRV}" \
    --aux_task_types ${AUX_TASK_TYPES} &

python experiments/robot/libero/run_robocasa_eval_new.py \
    --pretrained_checkpoint ${CHECKPOINT_PATH} \
    --robot JacoOmron \
    --task ${TASK} \
    --unnorm_key ${UNNORM_KEY} \
    --max_steps ${MAX_STEPS} \
    --act_horizon ${ACT_HORIZON} \
    --rollout_dir "experiments/rollouts/${TASK}/Jaco/${MODEL}/STEP/${AUX_ABRV}" \
    --aux_task_types ${AUX_TASK_TYPES} &

wait
#!/bin/bash
#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --mem=220G
#SBATCH --output=../slurm_logs/%A.out
#SBATCH --error=../slurm_logs/%A.err
#SBATCH --job-name="convert_stc_jaco"

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "Working directory = "$SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate robocasa
cd /iliad/u/jenseng/xembod/robocasa_xembod/
export MUJOCO_GL="egl"

ROBOT=jaco # either panda, jaco, iiwa, ur5e, kinova (lower case) 
DATA_PATH=/iliad/u/jenseng/xembod/robocasa_xembod/data/mg/JacoOmron/PnPSinkToCounter

CAMERA=${ROBOT}_agentview_left

python robocasa/scripts/batch_convert.py $DATA_PATH $CAMERA

wait

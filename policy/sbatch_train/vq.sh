#!/bin/bash
#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gpus-per-node=a5000:1
#SBATCH --output=../slurm_logs/%A.out
#SBATCH --error=../slurm_logs/%A.err
#SBATCH --job-name="vq_mg_panda_pnp"

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "Working directory = "$SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate robocasa
cd /iliad/u/jenseng/xembod/openvla-mini-aux/
export PYTHONPATH=/iliad/u/jenseng/xembod/openvla-mini-aux

python vla-scripts/pretrain_vq.py \
    --data_dir /iliad/u/jenseng/tensorflow_datasets/ \
    --data_mix mg_panda_pnp \
    --action_dim 7 \
    --future_action_horizon 7 \
    --vqvae_n_embed 256 \
    --epochs 2

# python vla-scripts/pretrain_vq.py \
#     --data_dir /iliad/u/jenseng/tensorflow_datasets/ \
#     --data_mix mg_jaco_flip_mug \
#     --action_dim 7 \
#     --future_action_horizon 7 \
#     --vqvae_n_embed 256 \
#     --epochs 2 &

wait
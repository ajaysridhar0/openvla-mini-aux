#!/bin/bash
#SBATCH --job-name=gpu8_test           # Job name
#SBATCH --output=gpu8_test_%j.out      # Output file
#SBATCH --error=gpu8_test_%j.err       # Error file
#SBATCH --time=00:10:00                # Maximum runtime (HH:MM:SS) - just 10 minutes for test
#SBATCH --nodes=1                      # Single node
#SBATCH --cpus-per-task=16             # CPU cores per task
#SBATCH --mem=64G                      # Total memory for the node
#SBATCH --account=models               # Account
#SBATCH --partition=hai                # HAI partition
#SBATCH --gres=gpu:h100:8              # Request 8 H100 GPUs
#SBATCH --mail-type=END,FAIL           # Email notifications
#SBATCH --mail-user=ajaysri@stanford.edu

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "Working directory = "$SLURM_SUBMIT_DIR

# Report GPU availability
echo "Testing GPU availability:"
nvidia-smi

# Simple sleep to keep the job running for a few seconds
echo "Job started, sleeping for 5 seconds..."
sleep 5
echo "Job completed successfully."
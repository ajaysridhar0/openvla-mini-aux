#!/bin/bash
#SBATCH --job-name=cotrain_base_sawyer_500_mg           # Job name
#SBATCH --output=/afs/.ir/users/a/j/ajaysri/openvla-mini-aux/slurm_jobs/slurmout/cotrain_base_sawyer_500_mg_%j.out      # Output file
#SBATCH --error=/afs/.ir/users/a/j/ajaysri/openvla-mini-aux/slurm_jobs/slurmout/cotrain_base_sawyer_500_mg_%j.err       # Error file
#SBATCH --time=24:00:00                # Maximum runtime (HH:MM:SS) - just 10 minutes for test
#SBATCH --nodes=1                      # Single node
#SBATCH --cpus-per-task=16             # CPU cores per task
#SBATCH --mem=64G                      # Total memory for the node
#SBATCH --account=models               # Account
#SBATCH --partition=hai                # HAI partition
#SBATCH --gres=gpu:h100:4              # Request 8 H100 GPUs
#SBATCH --mail-type=END,FAIL           # Email notifications
#SBATCH --mail-user=ajaysri@stanford.edu

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "Working directory = "$SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate openvla
cd /afs/.ir/users/a/j/ajaysri/openvla-mini-aux

python -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/train.py \
    --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-xembod-robocasa-full \
    --vla.base_vlm $BASE_VLM_DIR \
    --vla.data_mix good_saywer_500_mg_everything \
    --data_root_dir $XEMBOD_DATA_DIR \
    --vla.action_tokenizer extra_action_tokenizer \
    --vla.expected_world_size 4 \
    --vla.global_batch_size 96 \
    --vla.per_device_batch_size 24 \
    --vla.lr_scheduler_type=constant \
    --image_aug True \
    --vla.max_steps 200000 \
    --wandb_entity="ajaysridhar" \
    --wandb_project="prismatic-aux" \
    --run_id_note sawyer--image_aug \
    --run_id="cotrain_base_sawyer_500_mg" 
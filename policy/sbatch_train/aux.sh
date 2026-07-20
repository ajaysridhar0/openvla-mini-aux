#!/bin/bash
#SBATCH --account=iliad
#SBATCH --partition=sc-loprio
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --gpus-per-node=h200:4
#SBATCH --output=../slurm_logs/%A.out
#SBATCH --error=../slurm_logs/%A.err
#SBATCH --job-name="vla_mg_panda_pnp_aux"

task="pnp"
robot="panda"

if [[ "$task" == "pnp" ]]; then
  max_steps=30000
else
  max_steps=10000
fi

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "Working directory = "$SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate robocasa
cd /iliad/u/jenseng/xembod/openvla-mini-aux/
export PYTHONPATH=/iliad/u/jenseng/xembod/openvla-mini-aux

torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/train.py \
    --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-xembod-robocasa-full \
    --vla.base_vlm /iliad/u/jenseng/xembod/openvla-mini-aux/runs/prism-qwen25-extra-dinosiglip-224px+0_5b+stage-finetune+x7/ \
    --vla.data_mix mg_${robot}_${task} \
    --data_root_dir /iliad/u/jenseng/tensorflow_datasets \
    --vla.action_tokenizer mg_${robot}_${task}_vq_extra_action_tokenizer \
    --vla.expected_world_size 4 \
    --vla.global_batch_size 256 \
    --vla.per_device_batch_size 64 \
    --vla.lr_scheduler_type "constant" \
    --vla.max_steps ${max_steps} \
    --vla.use_wrist_image False \
    --vla.image_sequence_len 1 \
    --wandb_entity="jensen_team" \
    --run_id_note mg_${robot}_${task} \
    --run_id="aux" \
    --vla.transform_types="bbox->,low_level_motion->,ee_pose_2D->,action" \
    --save_interval 1250 \
    --resume_from_last True \

wait
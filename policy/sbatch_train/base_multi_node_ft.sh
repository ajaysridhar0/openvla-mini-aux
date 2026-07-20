#!/bin/bash
#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --time=8:00:00
#SBATCH --nodes=2                   # Increased to 2 nodes
#SBATCH --ntasks-per-node=1         # One task per node (torchrun handles the rest)
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gpus-per-node=l40s:4
#SBATCH --output=../slurm_logs/%A.out
#SBATCH --error=../slurm_logs/%A.err
#SBATCH --job-name="vla_mg_jaco_mug_ft_base"

task="flip_mug"
robot="jaco"

key="${robot}_${task}"
value="mg_${robot}_${task}"

if [[ "$task" == "pnp" ]]; then
  save_interval=1000
  max_steps=3000
else
  save_interval=500
  max_steps=2000
fi

# --- Multi-Node Environment Setup ---
# Get the name of the first node to act as the master/rendezvous point
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Master Node: $head_node at $head_node_ip"
echo "SLURM_JOBID="$SLURM_JOBID

source ~/.bashrc
conda activate robocasa
cd /iliad/u/jenseng/xembod/openvla-mini-aux/
export PYTHONPATH=/iliad/u/jenseng/xembod/openvla-mini-aux

export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # Set to 1 if your cluster DOES NOT have InfiniBand

# Use srun to launch torchrun on every node
srun torchrun \
    --nnodes 2 \
    --nproc-per-node 4 \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint "$head_node_ip:29500" \
    vla-scripts/train.py \
    --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-xembod-robocasa-full \
    --vla.base_vlm /iliad/u/jenseng/xembod/openvla-mini-aux/runs/prism-qwen25-extra-dinosiglip-224px+0_5b+stage-finetune+x7/ \
    --vla.data_mix ${robot}+mg_${robot}_${task} \
    --data_root_dir /iliad/u/jenseng/tensorflow_datasets \
    --vla.action_tokenizer mg_${robot}_${task}_vq_extra_action_tokenizer \
    --vla.expected_world_size 8 \
    --vla.global_batch_size 192 \
    --vla.per_device_batch_size 24 \
    --vla.lr_scheduler_type "constant" \
    --vla.max_steps ${max_steps} \
    --vla.use_wrist_image False \
    --vla.image_sequence_len 1 \
    --wandb_entity="jensen_team" \
    --run_id="base" \
    --run_id_note ${robot}+mg_${robot}_${task} \
    --save_interval ${save_interval} \
    --resume_from_last True \
    --dataset_statistics_map "{\"$key\": \"$value\"}" \
    --pretrained_checkpoint "/iliad/u/jenseng/xembod/openvla-mini-aux/runs/base--mg_${robot}_${task}/checkpoints/step-010000-epoch-10-loss=0.5329.pt" \
    --is_resume False \ 

wait
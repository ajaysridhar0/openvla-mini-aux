task="pnp"
robot="panda_og"

key="${robot}_${task}"
value="mg_${robot}_${task}"

if [[ "$task" == "pnp" ]]; then
  max_steps=30000
else
  max_steps=10000
fi

conda activate openvla
cd /home/jenseng/openvla-mini-aux/
export PYTHONPATH=/home/jenseng/openvla-mini-aux/

torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/train.py \
    --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-xembod-robocasa-full \
    --vla.base_vlm /home/jenseng/openvla-mini-aux/runs/prism-qwen25-extra-dinosiglip-224px+0_5b+stage-finetune+x7/ \
    --vla.data_mix mg_${robot}_${task} \
    --data_root_dir /home/jenseng/datasets \
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
    --save_interval 5000 \
    --resume_from_last True \
    --vla.transform_types="bbox->,low_level_motion->,ee_pose_2D->,action" \
 
wait
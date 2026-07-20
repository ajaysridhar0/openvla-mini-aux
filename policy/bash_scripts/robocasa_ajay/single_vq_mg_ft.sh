ROBOT=sawyer
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
    --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-xembod-robocasa-full \
    --vla.base_vlm /workspace/openvla-mini-aux/runs/prism-qwen25-extra-dinosiglip-224px+0_5b+stage-finetune+x7/ \
    --vla.data_mix ${ROBOT}_robocasa \
    --data_root_dir /datasets/jensen/xembod_data_human_rlds_final_filter_noop_actions/ \
    --vla.action_tokenizer robocasa_vq_extra_action_tokenizer \
    --vla.expected_world_size 8 \
    --vla.global_batch_size 128 \
    --vla.per_device_batch_size 16 \
    --vla.lr_scheduler_type "constant" \
    --vla.max_steps 20000 \
    --image_aug True \
    --pretrained_checkpoint "/workspace/openvla-mini-aux/runs/base_noop_filter_vq_mg_cotrain--sawyer--image_aug/checkpoints/step-050000-epoch-117-loss=0.1912.pt" \
    --is_resume False \
    --wandb_entity="jensen_team" \
    --run_id_note ${ROBOT} \
    --run_id="base_noop_filter_vq_mg_ft_50k"
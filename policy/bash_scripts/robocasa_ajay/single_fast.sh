ROBOT=sawyer
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
    --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-xembod-robocasa-full \
    --vla.base_vlm /workspace/openvla-mini-aux/runs/prism-qwen25-extra-dinosiglip-224px+0_5b+stage-finetune+x7/ \
    --vla.data_mix ${ROBOT}_robocasa \
    --data_root_dir /datasets/jensen/xembod_data_human_rlds_final_filter_noop_actions/ \
    --vla.action_tokenizer robocasa_fast_tokenizer_noop_filter \
    --vla.expected_world_size 8 \
    --vla.global_batch_size 128 \
    --vla.per_device_batch_size 16 \
    --vla.lr_scheduler_type=linear-warmup+cosine-decay \
    --vla.warmup_ratio 0.02 \
    --vla.max_steps 50000 \
    --vla.normalize_data False \
    --wandb_entity="jensen_team" \
    --run_id_note ${ROBOT} \
    --run_id="base_noop_filter_fast"
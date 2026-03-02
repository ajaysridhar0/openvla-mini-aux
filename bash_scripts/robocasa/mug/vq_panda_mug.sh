CUDA_VISIBLE_DEVICES=1 python vla-scripts/pretrain_vq.py \
    --data_dir /datasets/jensen/jensen_xembod_human_data \
    --data_mix panda_og_flip_mug \
    --action_dim 7 \
    --future_action_horizon 7 \
    --vqvae_n_embed 256 \
    --epochs 10 \
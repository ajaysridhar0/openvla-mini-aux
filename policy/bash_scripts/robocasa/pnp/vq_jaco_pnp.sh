python vla-scripts/pretrain_vq.py \
    --data_dir /datasets/jensen/jensen_xembod_human_data \
    --data_mix jaco_pnp \
    --action_dim 7 \
    --future_action_horizon 7 \
    --vqvae_n_embed 256 \
    --epochs 10 \
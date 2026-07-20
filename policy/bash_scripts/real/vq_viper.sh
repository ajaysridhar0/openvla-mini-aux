python vla-scripts/pretrain_vq.py \
    --data_dir /datasets/jensen/jensen_xembod_human_data \
    --data_mix viper_real_vary \
    --action_dim 7 \
    --future_action_horizon 7 \
    --vqvae_n_embed 256 \
    --epochs 50 \
    --dataset_statistics_map '{"viper_pnpsinktocounter_aux": "viper_pnpcountertosink_vary_aux"}' \

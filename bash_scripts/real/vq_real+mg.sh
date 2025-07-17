python vla-scripts/pretrain_vq.py \
    --data_dir /datasets/jensen/jensen_xembod_human_data \
    --data_mix panda_viper_real_mg \
    --action_dim 7 \
    --future_action_horizon 7 \
    --vqvae_n_embed 256 \
    --epochs 30 \
    --load_dir /workspace/openvla-mini-aux/vq/mg_pnp/checkpoints/model.pt \
    --dataset_statistics_map '{"viper_pnpcountertosink_aux": "viper_pnpsinktocounter_aux", "franka_pnpcountertosink_aux": "franka_pnpsinktocounter_aux"}' \

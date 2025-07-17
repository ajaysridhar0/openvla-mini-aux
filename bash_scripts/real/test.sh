python vla-scripts/pretrain_vq.py \
    --data_dir /datasets/jensen/jensen_xembod_human_data \
    --data_mix viper_real \
    --action_dim 7 \
    --future_action_horizon 7 \
    --vqvae_n_embed 256 \
    --epochs 2 \
    --load_dir /workspace/openvla-mini-aux/vq/pretrain_vq+mx-panda_viper_real_mg+fach-7+ng-7+nemb-256+nlatent-512/checkpoints/model.pt \
    --dataset_statistics_map '{"viper_pnpcountertosink": "viper_pnpsinktocounter", "franka_pnpcontertosink": "franka_pnpsinktocounter"}' \

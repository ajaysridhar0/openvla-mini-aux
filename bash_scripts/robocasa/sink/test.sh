CUDA_VISIBLE_DEVICES=1 python vla-scripts/pretrain_vq.py \
    --data_dir /datasets/jensen/jensen_xembod_human_data \
    --data_mix jaco_pnp_test \
    --action_dim 7 \
    --future_action_horizon 7 \
    --vqvae_n_embed 256 \
    --epochs 2 \
    --load_dir /workspace/openvla-mini-aux/vq/mg_pnp_lite/checkpoints/model.pt \
    --dataset_statistics_map '{"jaco_pnp": "mg_pnp_lite"}' \

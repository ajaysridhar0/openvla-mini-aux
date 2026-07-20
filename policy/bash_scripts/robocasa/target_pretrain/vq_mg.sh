python vla-scripts/pretrain_vq.py \
    --data_dir /iliad/u/jenseng/tensorflow_datasets/ \
    --data_mix mg_panda_flip_mug \
    --action_dim 7 \
    --future_action_horizon 7 \
    --vqvae_n_embed 256 \
    --epochs 2 \
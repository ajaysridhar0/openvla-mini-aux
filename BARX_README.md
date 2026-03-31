# BARX: Cross-Embodiment Transfer via Behavior-Aligned Representations

This repository contains the BARX code release for simulation training and finetuning on RoboCasa-style tasks.
It is built on top of the MiniVLA / OpenVLA-Mini codebase and focuses on three paper tasks:

- `PnP Counter to Sink` / `PnP Sink to Counter`
- `Turn On Sink Faucet`
- `Flip Mug Upright`

This README covers training with our simulation data from RoboCasa-X, our cross-embodiment benchmark.

## RoboCasa-X Benchmark

RoboCasa-X is a simulation benchmark for studying cross-embodiment transfer in robot manipulation. It is built on
[RoboCasa](https://github.com/robocasa/robocasa), a platform with realistic and diverse kitchen scenes, and uses
[MimicGen](https://mimicgen.github.io/) to scale demonstration data across embodiments.

### Tasks

RoboCasa-X includes four manipulation tasks that capture significant variation in kitchen layouts, textures, object
types, and object poses:

| Task | Description |
| --- | --- |
| PnP Counter to Sink | Pick an object from the counter and place it in the sink |
| PnP Sink to Counter | Pick an object from the sink and place it on the counter |
| Turn On Sink Faucet | Reach and turn on the kitchen faucet |
| Flip Mug Upright | Flip an inverted mug to an upright position |

PnP Counter to Sink and PnP Sink to Counter share a single model during training.

### Embodiments

The benchmark separates robots into **source** embodiments (used to generate large-scale prior data) and **target**
embodiments (used to evaluate cross-embodiment transfer with limited data).

| Role | Robot | Gripper |
| --- | --- | --- |
| Source | IIWA | Robotiq 2F-140 |
| Source | Kinova3 | Robotiq 2F-85 |
| Source | UR5e | Robotiq 2F-85 |
| Target | Panda | Robotiq 2F-85 |
| Target | Panda-OG | Franka Hand (original) |
| Target | Jaco | Robotiq 2F-85 |

Source data is generated with MimicGen from a small number of human demonstrations. Target data consists of 50 human
demonstrations per task per robot. Camera pose is randomized across all embodiments to promote sim-to-real transfer.

### Dataset Scales

| Name | Description | Demos per task |
| --- | --- | --- |
| XP-3K | Cross-embodiment prior, 1000 demos per source robot | 3000 |
| XP-900 | Cross-embodiment prior, 300 demos per source robot | 900 |
| SP-900 | Same-embodiment prior, 900 demos of the target robot | 900 |
| Target | Target-robot demonstrations only (no prior) | 50 |

The cross-embodiment transfer workflow is: (1) pretrain a policy on source-robot data (XP-3K or XP-900),
(2) co-fine-tune on the target-robot demonstrations plus down-weighted prior data. Target-only and same-embodiment
(SP-900) baselines skip or replace the cross-embodiment prior, respectively.

## Installation

These setup steps are the BARX-relevant parts of the main repo README, plus the VQ dependency BARX needs for action
tokenization.

The repository uses:

- Python `3.10`
- PyTorch `2.2.0`
- torchvision `0.17.0`
- transformers `4.40.1`
- tokenizers `0.19.1`
- timm `0.9.10`
- flash-attn `2.5.5`

Create the environment and install the repo:

```bash
conda create -n openvla python=3.10 -y
conda activate openvla

# Update this command if your compute platform needs a different PyTorch install.
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

pip install -e .
```

Install Flash Attention 2 for training:

```bash
pip install packaging ninja
ninja --version; echo $?   # should print 0
pip install "flash-attn==2.5.5" --no-build-isolation
```

If Flash Attention gives you trouble, try:

```bash
pip cache remove flash_attn
```

BARX also uses VQ-backed action tokenizers. Install the VQ-Bet dependency by following:

- https://github.com/jayLEE0301/vq_bet_official

One working setup is:

```bash
git clone https://github.com/jayLEE0301/vq_bet_official.git
cd vq_bet_official
pip install -r requirements.txt
pip install -e .
cd ..
```

The training code expects RoboCasa-X VQ tokenizer assets under `vq/` relative to this repository.

Training also expects a Hugging Face token (see [Environment Variables](#environment-variables) for setup).

Run the training commands from the repository root if you rely on the default `.hf_token` path and `vq/` relative
asset paths.

## Required Assets

### Base VLM

All BARX runs start from a pre-trained VLM checkpoint. The base VLM is publicly available on Hugging Face:

[`ajaysri/prism-qwen25-extra-dinosiglip-224px-0_5b-stage-finetune-x7`](https://huggingface.co/ajaysri/prism-qwen25-extra-dinosiglip-224px-0_5b-stage-finetune-x7)

Download it with:

```bash
huggingface-cli download ajaysri/prism-qwen25-extra-dinosiglip-224px-0_5b-stage-finetune-x7 \
    --local-dir prism-qwen25-extra-dinosiglip-224px+0_5b+stage-finetune+x7
```

Note: the first BARX training run may also download backbone weights from `timm` / `transformers` if they are not
already cached locally. In practice this means the first run may fetch:

- DINOv2 ViT-L backbone weights
- SigLIP SO400M backbone weights
- Qwen2.5-0.5B model/tokenizer files

Subsequent runs reuse the local cache.

### Datasets

Released paper datasets live in the Hugging Face RoboCasa-X collection:

- https://huggingface.co/collections/ajaysri/robocasa-x

This README uses the released RoboCasa-X names directly for `--vla.data_mix`, `--vla.action_tokenizer`,
`--dataset_statistics_map`, and `--non_action_datasets`.

Released dataset repo names follow this scheme:

- `robocasa-x-xp3k-<task>` for cross-embodiment `XP-3K`
- `robocasa-x-xp900-<task>` for cross-embodiment `XP-900`
- `robocasa-x-sp900-<robot>-<task>` for same-embodiment `SP-900`

For finetune mixtures, RoboCasa-X uses two additional alias patterns:

- `robocasa-x-xp{3k,900}-<robot>-<task>-mix` for cross-embodiment finetuning
- `robocasa-x-sp900-<robot>-<task>-mix` for same-embodiment finetuning

For target-only baselines, use:

- `robocasa-x-target-<robot>-<task>`

The training data should be available in TFDS format under a single root dir. See the
[Task And Mixture Mapping](#task-and-mixture-mapping) tables for every concrete dataset name.

### VQ Assets

Released VQ tokenizers live in the Hugging Face RoboCasa-X VQ Action Tokenizers collection:

- https://huggingface.co/collections/ajaysri/robocasa-x-vq-action-tokenizers

You usually do not choose the VQ tokenizer independently. Choose it from the training setup:

1. Prior training: use the tokenizer that matches the prior dataset exactly.
2. Finetuning from a prior: keep using the prior tokenizer, even though `--vla.data_mix` is now a `*-mix` dataset.
3. Target-only baselines: use the tokenizer that matches the target dataset exactly.

Examples:

| Training setup | `--vla.data_mix` | `--vla.action_tokenizer` |
| --- | --- | --- |
| Cross-emb prior | `robocasa-x-xp900-pnp` | `robocasa-x-xp900-pnp-vq-extra-action-tokenizer` |
| Cross-emb finetune from XP-3K prior | `robocasa-x-xp3k-panda-pnp-mix` | `robocasa-x-xp3k-pnp-vq-extra-action-tokenizer` |
| Same-emb finetune from SP-900 prior | `robocasa-x-sp900-panda-pnp-mix` | `robocasa-x-sp900-panda-pnp-vq-extra-action-tokenizer` |
| Target-only baseline | `robocasa-x-target-panda-pnp` | `robocasa-x-target-panda-pnp-vq-extra-action-tokenizer` |

Download each tokenizer repo into the matching local RoboCasa-X directory under `vq/`.
The local directory name is the tokenizer's dataset slug with hyphens replaced by underscores.

Examples:

| Hugging Face tokenizer repo | Local directory expected by the code |
| --- | --- |
| `ajaysri/robocasa-x-xp3k-pnp-vq-extra-action-tokenizer` | `vq/robocasa_x_xp3k_pnp` |
| `ajaysri/robocasa-x-xp900-pnp-vq-extra-action-tokenizer` | `vq/robocasa_x_xp900_pnp` |
| `ajaysri/robocasa-x-sp900-panda-pnp-vq-extra-action-tokenizer` | `vq/robocasa_x_sp900_panda_pnp` |
| `ajaysri/robocasa-x-target-panda-pnp-vq-extra-action-tokenizer` | `vq/robocasa_x_target_panda_pnp` |

For example:

```bash
huggingface-cli download ajaysri/robocasa-x-xp900-pnp-vq-extra-action-tokenizer \
    --repo-type model \
    --local-dir vq/robocasa_x_xp900_pnp
```

At minimum, BARX training expects these prior-tokenizer directories under `vq/`:

- `vq/robocasa_x_xp3k_pnp`, `vq/robocasa_x_xp900_pnp`
- `vq/robocasa_x_xp3k_turn_on_sink_faucet`, `vq/robocasa_x_xp900_turn_on_sink_faucet`
- `vq/robocasa_x_xp3k_flip_mug_upright`, `vq/robocasa_x_xp900_flip_mug_upright`

Depending on which runs you reproduce, you may also need the target-only and same-embodiment directories:

- `vq/robocasa_x_target_<robot>_pnp`
- `vq/robocasa_x_target_<robot>_turn_on_sink_faucet`
- `vq/robocasa_x_target_<robot>_flip_mug_upright`
- `vq/robocasa_x_sp900_<robot>_pnp`
- `vq/robocasa_x_sp900_<robot>_turn_on_sink_faucet`
- `vq/robocasa_x_sp900_<robot>_flip_mug_upright`

## Environment Variables

```bash
export BARX_DATA_ROOT=/path/to/rlds_datasets
export BARX_BASE_VLM=/path/to/prism-qwen25-extra-dinosiglip-224px+0_5b+stage-finetune+x7
export HF_TOKEN=hf_...
export WANDB_ENTITY=your_wandb_entity

# Useful for fresh-user smoke tests.
export WANDB_MODE=disabled

# Set to the number of GPUs available on your machine.
# Adjust global_batch_size in the training commands accordingly.
export NUM_GPUS=8
```

All commands below assume `--hf_token HF_TOKEN`.

## BARX Training Setup

BARX uses `vla-scripts/train.py` for both prior training and target-robot finetuning.
Run outputs are written to `runs/<run_id>--<run_id_note>/`; for example, a Joint Reps prior on
XP-900 PnP lands in `runs/aux--robocasa-x-xp900-pnp/`. The finetuning command references
checkpoints from this path.

High-level workflow:

1. Train a prior on a cross-embodiment or same-embodiment mixture.
2. Finetune a target robot from that prior.
3. For target-only baselines, train directly on the target dataset without a prior checkpoint.

All BARX examples below use:

- `--vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-xembod-robocasa-full`
- `--vla.use_wrist_image False`
- `--vla.image_sequence_len 1`
- constant LR schedule

## Method Mapping

| Paper method | `run_id` | `--vla.transform_types` |
| --- | --- | --- |
| No Reps | `base` | `"action"` |
| Joint Reps | `aux` | `"bbox->,low_level_motion->,ee_pose_2D->,action"` |
| Bounding Box | `bbox` | `"bbox->,action"` |
| EE Trace | `ee` | `"ee_pose_2D->,action"` |
| Language Motions | `lm` | `"low_level_motion->,action"` |
| ECoT | `chain` | `"bbox->low_level_motion->ee_pose_2D,action"` |

## Aux Transform Syntax

`--vla.transform_types` is a comma-separated list of transform modes. Each comma-separated entry is sampled as its own
mode during training.

- `action` means direct action supervision only.
- `bbox`, `obj_pose`, `low_level_motion`, and `ee_pose_2D` mean aux-only supervision for that target.
- Any transform containing `->` is a chained aux-to-action sample. The aux tasks are supervised left-to-right in the
  order you write them, and action supervision is appended automatically at the end.
- You can choose any ordered subset of the supported aux tasks. For example, `bbox->obj_pose` and `obj_pose->bbox`
  are different chains.
- For a single-step chain, use a trailing arrow: `bbox->` means "predict bbox, then predict action". Plain `bbox`
  means aux-only and does not include the action target.
- Do not write `action` inside a chain. `bbox->ee_pose_2D` already means `bbox`, then `ee_pose_2D`, then action.
- `--vla.transform_weights` can weight the comma-separated entries. Its weights must align with the transform list and
  sum to `1.0`.

Examples:

- `"action"`: direct action prediction only.
- `"bbox"`: bbox-only aux supervision.
- `"bbox->"`: bbox followed by action.
- `"bbox->,action"`: mixture of bbox-then-action samples and pure action samples.
- `"bbox->obj_pose->ee_pose_2D"`: custom left-to-right chain ending in action.
- `"bbox->ee_pose_2D,low_level_motion->,action"`: two custom chained modes plus a pure action mode.
- `--vla.transform_types "bbox->obj_pose->ee_pose_2D,action" --vla.transform_weights "0.7,0.3"`: spend 70% of
  samples on that custom chain and 30% on pure action supervision.

## Task And Mixture Mapping

BARX training is two-phase: **pretrain** a prior on a large multi-robot mixture, then **finetune** on a
target-robot mixture. Target-only baselines skip the pretraining step entirely.
The tables below map paper settings to the CLI values used in each phase.

In all tables below, `<robot>` is one of `panda`, `panda-og`, or `jaco`.

### Pretraining Mixtures

Use these with the [Prior Training Command](#prior-training-command). Each row is a standalone prior run.
The action tokenizer for any `data_mix` is always `<data_mix>-vq-extra-action-tokenizer`.

| Paradigm | Task | `data_mix` |
| --- | --- | --- |
| Cross-emb XP-3K | PnP | `robocasa-x-xp3k-pnp` |
| Cross-emb XP-3K | Turn On Sink Faucet | `robocasa-x-xp3k-turn-on-sink-faucet` |
| Cross-emb XP-3K | Flip Mug Upright | `robocasa-x-xp3k-flip-mug-upright` |
| Cross-emb XP-900 | PnP | `robocasa-x-xp900-pnp` |
| Cross-emb XP-900 | Turn On Sink Faucet | `robocasa-x-xp900-turn-on-sink-faucet` |
| Cross-emb XP-900 | Flip Mug Upright | `robocasa-x-xp900-flip-mug-upright` |
| Same-emb SP-900 | PnP | `robocasa-x-sp900-<robot>-pnp` |
| Same-emb SP-900 | Turn On Sink Faucet | `robocasa-x-sp900-<robot>-turn-on-sink-faucet` |
| Same-emb SP-900 | Flip Mug Upright | `robocasa-x-sp900-<robot>-flip-mug-upright` |

### Finetuning Mixtures

Use these with the [Finetuning Command](#finetuning-command). Each finetune continues from a
pretrained prior checkpoint.

Each finetune mixture includes the **prior dataset** (down-weighted) alongside the **target robot dataset**
(weight 1.0). XP-3K finetunes weight the prior at 1%; XP-900 and SP-900 finetunes weight it at 5%.
The action tokenizer carries over from the prior. Set `--dataset_statistics_map` to
`{"robocasa-x-target-<robot>-<task>": "<prior_data_mix>"}` (maps the target dataset to the prior's normalization stats).

| Paradigm | Task | `data_mix` |
| --- | --- | --- |
| Cross-emb XP-3K | PnP | `robocasa-x-xp3k-<robot>-pnp-mix` |
| Cross-emb XP-3K | Turn On Sink Faucet | `robocasa-x-xp3k-<robot>-turn-on-sink-faucet-mix` |
| Cross-emb XP-3K | Flip Mug Upright | `robocasa-x-xp3k-<robot>-flip-mug-upright-mix` |
| Cross-emb XP-900 | PnP | `robocasa-x-xp900-<robot>-pnp-mix` |
| Cross-emb XP-900 | Turn On Sink Faucet | `robocasa-x-xp900-<robot>-turn-on-sink-faucet-mix` |
| Cross-emb XP-900 | Flip Mug Upright | `robocasa-x-xp900-<robot>-flip-mug-upright-mix` |
| Same-emb SP-900 | PnP | `robocasa-x-sp900-<robot>-pnp-mix` |
| Same-emb SP-900 | Turn On Sink Faucet | `robocasa-x-sp900-<robot>-turn-on-sink-faucet-mix` |
| Same-emb SP-900 | Flip Mug Upright | `robocasa-x-sp900-<robot>-flip-mug-upright-mix` |

### Target-Only Baselines

Train directly on the target dataset with no prior. Omit `--pretrained_checkpoint` and `--dataset_statistics_map`.
The action tokenizer follows the same `<data_mix>-vq-extra-action-tokenizer` pattern.

| Task | `data_mix` | `max_steps` | `save_interval` |
| --- | --- | --- | --- |
| PnP | `robocasa-x-target-<robot>-pnp` | 3000 | 1000 |
| Turn On Sink Faucet | `robocasa-x-target-<robot>-turn-on-sink-faucet` | 2000 | 500 |
| Flip Mug Upright | `robocasa-x-target-<robot>-flip-mug-upright` | 2000 | 500 |

## Fresh-User Smoke Test

The fastest clean bootstrap test is a target-only Panda PnP run on one GPU with freshly downloaded assets.

```bash
export NUM_GPUS=1
export TARGET_DATA_MIX=robocasa-x-target-panda-pnp
export ACTION_TOKENIZER=robocasa-x-target-panda-pnp-vq-extra-action-tokenizer
export RUN_ID=base
export RUN_NOTE=robocasa-x-target-panda-pnp-smoke

torchrun --standalone --nnodes 1 --nproc-per-node "$NUM_GPUS" vla-scripts/train.py \
    --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-xembod-robocasa-full \
    --vla.base_vlm "$BARX_BASE_VLM" \
    --vla.data_mix "$TARGET_DATA_MIX" \
    --data_root_dir "$BARX_DATA_ROOT" \
    --vla.action_tokenizer "$ACTION_TOKENIZER" \
    --vla.expected_world_size 1 \
    --vla.global_batch_size 1 \
    --vla.per_device_batch_size 1 \
    --vla.lr_scheduler_type constant \
    --vla.max_steps 1 \
    --vla.use_wrist_image False \
    --vla.image_sequence_len 1 \
    --run_id_note "$RUN_NOTE" \
    --run_id "$RUN_ID" \
    --is_resume False \
    --vla.transform_types action \
    --hf_token HF_TOKEN \
    --save_interval 1
```

This smoke test should create a run directory, write dataset statistics, and save a step-1 checkpoint without relying
on any preexisting local BARX assets.

## Prior Training Command

The example below trains a Joint Reps (`aux`) prior on `XP-900 PnP`. Swap the first block of
env vars to match your target task, scale, and method from the tables above.

```bash
export PRIOR_DATA_MIX=robocasa-x-xp900-pnp
export ACTION_TOKENIZER=robocasa-x-xp900-pnp-vq-extra-action-tokenizer
export RUN_ID=aux
export RUN_NOTE=robocasa-x-xp900-pnp
export TRANSFORM_TYPES='bbox->,low_level_motion->,ee_pose_2D->,action'
export MAX_STEPS=100000
export SAVE_INTERVAL=10000

torchrun --standalone --nnodes 1 --nproc-per-node "$NUM_GPUS" vla-scripts/train.py \
    --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-xembod-robocasa-full \
    --vla.base_vlm "$BARX_BASE_VLM" \
    --vla.data_mix "$PRIOR_DATA_MIX" \
    --data_root_dir "$BARX_DATA_ROOT" \
    --vla.action_tokenizer "$ACTION_TOKENIZER" \
    --vla.expected_world_size "$NUM_GPUS" \
    --vla.global_batch_size 128 \
    --vla.per_device_batch_size 16 \
    --vla.lr_scheduler_type constant \
    --vla.max_steps "$MAX_STEPS" \
    --vla.use_wrist_image False \
    --vla.image_sequence_len 1 \
    --wandb_entity "$WANDB_ENTITY" \
    --run_id_note "$RUN_NOTE" \
    --run_id "$RUN_ID" \
    --vla.transform_types "$TRANSFORM_TYPES" \
    --is_resume False \
    --hf_token HF_TOKEN \
    --save_interval "$SAVE_INTERVAL"
```

For an exact paper rerun, match `max_steps`, `save_interval`, and any resume fields to the released prior checkpoint
configuration you are recreating.

## Finetuning Command

The example below finetunes Panda PnP Joint Reps from an `XP-3K` prior checkpoint. Swap the first
block of env vars to match your target task, robot, scale, and method.

```bash
export TARGET_DATA_MIX=robocasa-x-xp3k-panda-pnp-mix
export ACTION_TOKENIZER=robocasa-x-xp3k-pnp-vq-extra-action-tokenizer
export PRETRAIN_CKPT=/path/to/aux--robocasa-x-xp3k-pnp/checkpoints/<step-file>.pt
export RUN_ID=aux
export RUN_NOTE=robocasa-x-xp3k-panda-pnp-mix
export TRANSFORM_TYPES='bbox->,low_level_motion->,ee_pose_2D->,action'
export MAX_STEPS=3000
export SAVE_INTERVAL=1000
export DATASET_STATS_JSON='{"robocasa-x-target-panda-pnp":"robocasa-x-xp3k-pnp"}'

torchrun --standalone --nnodes 1 --nproc-per-node "$NUM_GPUS" vla-scripts/train.py \
    --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-xembod-robocasa-full \
    --vla.base_vlm "$BARX_BASE_VLM" \
    --vla.data_mix "$TARGET_DATA_MIX" \
    --data_root_dir "$BARX_DATA_ROOT" \
    --vla.action_tokenizer "$ACTION_TOKENIZER" \
    --vla.expected_world_size "$NUM_GPUS" \
    --vla.global_batch_size 16 \
    --vla.per_device_batch_size 16 \
    --vla.lr_scheduler_type constant \
    --vla.max_steps "$MAX_STEPS" \
    --vla.use_wrist_image False \
    --vla.image_sequence_len 1 \
    --wandb_entity "$WANDB_ENTITY" \
    --run_id_note "$RUN_NOTE" \
    --run_id "$RUN_ID" \
    --pretrained_checkpoint "$PRETRAIN_CKPT" \
    --is_resume False \
    --dataset_statistics_map "$DATASET_STATS_JSON" \
    --vla.transform_types "$TRANSFORM_TYPES" \
    --vla.warmup_ratio 0.02 \
    --hf_token HF_TOKEN \
    --save_interval "$SAVE_INTERVAL"
```


<!-- ## Troubleshooting

If TFDS loading breaks, the main repo README recommends:

```bash
pip install tensorflow-datasets==4.9.3
```

If `dlimp` incompatibilities show up, the main repo README recommends:

```bash
pip install --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla
```

If a BARX VQ tokenizer fails to load, check:

- the `vq/` symlink or directory exists
- the expected `vq/<name>/config.json` exists
- the expected `vq/<name>/checkpoints/model.pt` exists -->

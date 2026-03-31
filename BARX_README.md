# BARX: Cross-Embodiment Transfer via Behavior-Aligned Representations

This repository contains the BARX code release for simulation training and finetuning on RoboCasa-style tasks.
It is built on top of the MiniVLA / OpenVLA-Mini codebase and focuses on three paper tasks:

- `PnP Counter to Sink` / `PnP Sink to Counter`
- `Turn On Sink Faucet`
- `Flip Mug Upright`

This README covers training with our simulation data from RoboCasa-X, our cross-embodiment benchmark.

## RoboCasa-X Benchmark

[RoboCasa-X](https://huggingface.co/collections/ajaysri/robocasa-x) is a simulation benchmark for cross-embodiment
transfer built on [RoboCasa](https://github.com/robocasa/robocasa) and
[MimicGen](https://mimicgen.github.io/). It covers four kitchen tasks (`PnP Counter to Sink`, `PnP Sink to Counter`, `Turn On Sink Faucet`,
`Flip Mug Upright`), three source robots (IIWA, Kinova3, UR5e), and three target robots (Panda, Panda-OG, Jaco).
The two PnP tasks share a single model during training.

Dataset scales: **XP-900** (900 source demos), **XP-3K** (3000 source demos), **SP-900** (900 same-embodiment demos),
and **Target** (50 target-robot demos). The workflow is: (1) pretrain on source data, (2) co-fine-tune on
target demos plus down-weighted prior data. Target-only and SP-900 baselines skip or replace the cross-embodiment prior.

## Installation

Create the environment and install the repo:

```bash
conda create -n openvla python=3.10 -y
conda activate openvla

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

pip install -e .
```

Install Flash Attention 2 for training:

```bash
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

Install the VQ-Bet dependency that BARX needs for action tokenization:

```bash
git clone https://github.com/jayLEE0301/vq_bet_official.git
cd vq_bet_official
pip install -r requirements.txt
pip install -e .
cd ..
```

The training code expects RoboCasa-X VQ tokenizer assets under `vq/` relative to this repository and a
Hugging Face token (see [Environment Variables](#environment-variables)). Run training commands from the repo root.

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

### Released Pretraining Checkpoints (skip prior training)

We release prior checkpoints so you can **go straight to finetuning**. All checkpoints are in the
[BARX Pretraining Models](https://huggingface.co/collections/ajaysri/barx-pretraining-models-joint-reps-and-no-reps)
collection. Checkpoint names follow the pattern `barx-<run_id>-<scale>-<task>`, where `run_id` is `aux`
(Joint Reps) or `base` (No Reps), scale is `xp900` or `xp3k`, and task is `pnp`, `turn-on-sink-faucet`,
or `flip-mug-upright`.

Download example:

```bash
huggingface-cli download ajaysri/barx-aux-xp3k-pnp \
    --repo-type model \
    --local-dir runs/aux--robocasa-x-xp3k-pnp/checkpoints

export PRETRAIN_CKPT=runs/aux--robocasa-x-xp3k-pnp/checkpoints/step-090000-epoch-38-loss=0.2513.pt
```

> **Tip:** To reproduce finetuning results only, download a released checkpoint and skip to the
> [Finetuning Command](#finetuning-command).

### Datasets

Released paper datasets live in the Hugging Face RoboCasa-X collection:

- https://huggingface.co/collections/ajaysri/robocasa-x

This README uses the released RoboCasa-X names directly for `--vla.data_mix`, `--vla.action_tokenizer`,
`--dataset_statistics_map`, and `--non_action_datasets`.

Released dataset repo names follow this scheme:

- `robocasa-x-xp900-<task>` for cross-embodiment `XP-900`
- `robocasa-x-xp3k-<task>` for cross-embodiment `XP-3K`
- `robocasa-x-sp900-<robot>-<task>` for same-embodiment `SP-900`

For finetune mixtures, RoboCasa-X uses two additional alias patterns:

- `robocasa-x-xp{900,3k}-<robot>-<task>-mix` for cross-embodiment finetuning
- `robocasa-x-sp900-<robot>-<task>-mix` for same-embodiment finetuning

For target-only baselines, use:

- `robocasa-x-target-<robot>-<task>`

The training data should be available in TFDS format under a single root dir. See the
[Task And Mixture Mapping](#task-and-mixture-mapping) tables for every concrete dataset name.

### VQ Assets

Released VQ tokenizers: [RoboCasa-X VQ Action Tokenizers](https://huggingface.co/collections/ajaysri/robocasa-x-vq-action-tokenizers).
To train your own, see the [OpenVLA-Mini repo](https://github.com/Stanford-ILIAD/openvla-mini).

The action tokenizer for any `data_mix` is always `<data_mix>-vq-extra-action-tokenizer`. When finetuning from a
prior, keep the **prior's** tokenizer (not the finetune mix's).

Download each tokenizer into `vq/<data_mix_with_underscores>`:

```bash
huggingface-cli download ajaysri/robocasa-x-xp900-pnp-vq-extra-action-tokenizer \
    --repo-type model \
    --local-dir vq/robocasa_x_xp900_pnp
```

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

`--vla.transform_types` is a comma-separated list of transform modes sampled during training.

- `action` — direct action supervision only.
- `bbox`, `obj_pose`, `low_level_motion`, `ee_pose_2D` — aux-only supervision (no action target).
- A trailing `->` adds action after the aux chain: `bbox->` = predict bbox then action.
- Multi-step chains: `bbox->ee_pose_2D` = bbox, then ee_pose_2D, then action (left-to-right). Do not write `action`
  inside a chain.
- `--vla.transform_weights` can weight entries (must sum to 1.0).

Examples:

- `"action"`: action only.
- `"bbox->"`: bbox then action.
- `"bbox->,action"`: mixture of bbox-then-action and pure action samples.
- `"bbox->obj_pose->ee_pose_2D,action" --vla.transform_weights "0.7,0.3"`: 70% chain, 30% action.

## Task And Mixture Mapping

All `data_mix` names follow a consistent pattern. `<task>` is `pnp`, `turn-on-sink-faucet`, or `flip-mug-upright`.
`<robot>` is `panda`, `panda-og`, or `jaco`.

| Phase | Pattern | Example |
| --- | --- | --- |
| Pretrain (cross-emb) | `robocasa-x-{xp900,xp3k}-<task>` | `robocasa-x-xp900-pnp` |
| Pretrain (same-emb) | `robocasa-x-sp900-<robot>-<task>` | `robocasa-x-sp900-panda-pnp` |
| Finetune | append `-<robot>-<task>-mix` to prior pattern | `robocasa-x-xp3k-panda-pnp-mix` |
| Target-only | `robocasa-x-target-<robot>-<task>` | `robocasa-x-target-panda-pnp` |

**Finetuning notes:** each finetune mixture includes the prior dataset (down-weighted) plus the target dataset
(weight 1.0). XP-900 and SP-900 finetunes weight the prior at 5%; XP-3K at 1%. The action tokenizer carries over
from the prior. Set `--dataset_statistics_map '{"robocasa-x-target-<robot>-<task>": "<prior_data_mix>"}'`.

**Target-only notes:** omit `--pretrained_checkpoint` and `--dataset_statistics_map`. Use `max_steps` 3000 /
`save_interval` 1000 for PnP; 2000 / 500 for Turn On Sink Faucet and Flip Mug Upright.

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

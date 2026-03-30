# BARX Training

BARX (Cross-Embodiment Transfer via Behavior-Aligned Representations) is built on top of the
[MiniVLA codebase](https://github.com/Stanford-ILIAD/openvla-mini) and focuses on three paper tasks:

- `PnP Counter to Sink` / `PnP Sink to Counter`
- `Turn On Sink Faucet`
- `Flip Mug Upright`

This document covers the BARX simulation training runs used in the paper:

- prior training on the simulation mixtures
- target-robot finetuning from those priors
- no-prior target-only training baselines

## Prerequisites

This repository uses:

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

BARX uses VQ-backed action tokenizers. Install the VQ dependency and make sure the corresponding `vq/` assets are present locally:

```bash
git clone https://github.com/jayLEE0301/vq_bet_official.git
cd vq_bet_official
pip install -r requirements.txt
pip install -e .
cd ..
```

## Base VLM

All BARX runs start from a pre-trained VLM checkpoint. The base VLM is publicly available on Hugging Face:

[`ajaysri/prism-qwen25-extra-dinosiglip-224px-0_5b-stage-finetune-x7`](https://huggingface.co/ajaysri/prism-qwen25-extra-dinosiglip-224px-0_5b-stage-finetune-x7)

Download it with:

```bash
huggingface-cli download ajaysri/prism-qwen25-extra-dinosiglip-224px-0_5b-stage-finetune-x7 \
    --local-dir prism-qwen25-extra-dinosiglip-224px+0_5b+stage-finetune+x7
```

## Environment

```bash
export BARX_DATA_ROOT=/path/to/rlds_datasets
export BARX_BASE_VLM=/path/to/prism-qwen25-extra-dinosiglip-224px+0_5b+stage-finetune+x7
export BARX_PRIOR_ROOT=/path/to/barx_runs/priors
export BARX_FT_ROOT=/path/to/barx_runs/finetunes
export NUM_GPUS=8
export PER_DEVICE_BATCH_SIZE=32
export GLOBAL_BATCH_SIZE=$((NUM_GPUS * PER_DEVICE_BATCH_SIZE))
export WANDB_ENTITY=your_wandb_entity
```

## Required Datasets

Released paper datasets live in the Hugging Face BARX Paper Datasets collection:

- https://huggingface.co/collections/ajaysri/barx-paper-datasets-69ca39478824752feea55460

The collection uses paper-facing dataset names. The BARX release also adds paper-facing aliases for
`--vla.data_mix`, `--vla.action_tokenizer`, `--dataset_statistics_map`, and `--non_action_datasets`.
This README uses those aliases throughout. Historical BARX `mg_*` names remain supported for backwards compatibility.

Each released dataset has two related names:

- Hugging Face repo slug: paper-facing and hyphenated, for example `barx-xp900-flip-mug-upright`
- TFDS builder id inside the repo: underscore-safe, for example `barx_xp900_flip_mug_upright`

Use the hyphenated BARX aliases in the training commands below. If you download a dataset repo and point BARX at the
raw TFDS directory directly, the underlying builder id is the underscore version.

Released dataset repo names follow this scheme:

- `barx-xp3k-<task>` for cross-embodiment `XP-3K`
- `barx-xp900-<task>` for cross-embodiment `XP-900`
- `barx-sp900-<robot>-<task>` for same-embodiment `SP-900`

For finetune mixtures, BARX uses two additional alias patterns:

- `barx-xp{3k,900}-<robot>-<task>-mix` for cross-embodiment finetuning
- `barx-sp900-<robot>-<task>-mix` for same-embodiment finetuning

For target-only baselines, use:

- `barx-target-<robot>-<task>`

Cross-embodiment prior datasets:

- `barx-xp3k-pnp`, `barx-xp900-pnp`
- `barx-xp3k-turn-on-sink-faucet`, `barx-xp900-turn-on-sink-faucet`
- `barx-xp3k-flip-mug-upright`, `barx-xp900-flip-mug-upright`

Target datasets:

- `barx-target-panda-pnp`, `barx-target-panda-og-pnp`, `barx-target-jaco-pnp`
- `barx-target-panda-turn-on-sink-faucet`, `barx-target-panda-og-turn-on-sink-faucet`, `barx-target-jaco-turn-on-sink-faucet`
- `barx-target-panda-flip-mug-upright`, `barx-target-panda-og-flip-mug-upright`, `barx-target-jaco-flip-mug-upright`

Same-embodiment prior datasets:

- `barx-sp900-panda-pnp`, `barx-sp900-panda-og-pnp`, `barx-sp900-jaco-pnp`
- `barx-sp900-panda-turn-on-sink-faucet`, `barx-sp900-panda-og-turn-on-sink-faucet`, `barx-sp900-jaco-turn-on-sink-faucet`
- `barx-sp900-panda-flip-mug-upright`, `barx-sp900-panda-og-flip-mug-upright`, `barx-sp900-jaco-flip-mug-upright`

## Naming Convention

`vla-scripts/train.py` writes runs to:

```text
<run_root_dir>/<run_id>--<run_id_note>
```

Examples:

- `run_id="aux"` and `run_id_note="barx-xp900-pnp"` -> `aux--barx-xp900-pnp`
- `run_id="bbox"` and `run_id_note="barx-xp900-panda-flip-mug-upright-mix"` -> `bbox--barx-xp900-panda-flip-mug-upright-mix`

Important: `run_id_note` is only a name. The actual data comes from `--vla.data_mix`, which is a registered mixture id in
[`prismatic/vla/datasets/rlds/oxe/mixtures.py`](/iliad2/u/ajaysri/openvla-mini-aux/prismatic/vla/datasets/rlds/oxe/mixtures.py).
The BARX release aliases are supported directly there, so you can use paper-facing names in both `run_id_note` and
`data_mix`.

## Method Mapping

| Paper method | `run_id` | `--vla.transform_types` |
| --- | --- | --- |
| No Reps | `base` | `"action"` |
| Joint Reps | `aux` | `"bbox->,low_level_motion->,ee_pose_2D->,action"` |
| Bounding Box | `bbox` | `"bbox->,action"` |
| EE Trace | `ee` | `"ee_pose_2D->,action"` |
| Language Motions | `lm` | `"low_level_motion->,action"` |
| ECoT | `chain` | `"bbox->obj_pose->low_level_motion->ee_pose_2D,action"` |

Historical BARX launch scripts often used `chain` with `"bbox->ee_pose_2D->low_level_motion->,action"`. The data
pipeline also supports `obj_pose`, so the table above is the recommended release-time BARX ECoT definition.

## Task Mapping

### Cross-Embodiment Priors

| Task | Prior scale | Prior `data_mix` | Action tokenizer |
| --- | --- | --- | --- |
| `PnP` | `XP-3K` | `barx-xp3k-pnp` | `barx-xp3k-pnp-vq-extra-action-tokenizer` |
| `PnP` | `XP-900` | `barx-xp900-pnp` | `barx-xp900-pnp-vq-extra-action-tokenizer` |
| `Turn On Sink Faucet` | `XP-3K` | `barx-xp3k-turn-on-sink-faucet` | `barx-xp3k-turn-on-sink-faucet-vq-extra-action-tokenizer` |
| `Turn On Sink Faucet` | `XP-900` | `barx-xp900-turn-on-sink-faucet` | `barx-xp900-turn-on-sink-faucet-vq-extra-action-tokenizer` |
| `Flip Mug Upright` | `XP-3K` | `barx-xp3k-flip-mug-upright` | `barx-xp3k-flip-mug-upright-vq-extra-action-tokenizer` |
| `Flip Mug Upright` | `XP-900` | `barx-xp900-flip-mug-upright` | `barx-xp900-flip-mug-upright-vq-extra-action-tokenizer` |

### Cross-Embodiment Finetunes

| Task | Finetune scale | Finetune `data_mix` | `dataset_statistics_map` target |
| --- | --- | --- | --- |
| `PnP` | `XP-3K` | `barx-xp3k-<robot>-pnp-mix` | `{"barx-target-<robot>-pnp": "barx-xp3k-pnp"}` |
| `PnP` | `XP-900` | `barx-xp900-<robot>-pnp-mix` | `{"barx-target-<robot>-pnp": "barx-xp900-pnp"}` |
| `Turn On Sink Faucet` | `XP-3K` | `barx-xp3k-<robot>-turn-on-sink-faucet-mix` | `{"barx-target-<robot>-turn-on-sink-faucet": "barx-xp3k-turn-on-sink-faucet"}` |
| `Turn On Sink Faucet` | `XP-900` | `barx-xp900-<robot>-turn-on-sink-faucet-mix` | `{"barx-target-<robot>-turn-on-sink-faucet": "barx-xp900-turn-on-sink-faucet"}` |
| `Flip Mug Upright` | `XP-3K` | `barx-xp3k-<robot>-flip-mug-upright-mix` | `{"barx-target-<robot>-flip-mug-upright": "barx-xp3k-flip-mug-upright"}` |
| `Flip Mug Upright` | `XP-900` | `barx-xp900-<robot>-flip-mug-upright-mix` | `{"barx-target-<robot>-flip-mug-upright": "barx-xp900-flip-mug-upright"}` |

`<robot>` is one of `panda`, `panda-og`, or `jaco`.

### Same-Embodiment Priors and Finetunes

| Task | Same prior `data_mix` | Same finetune `data_mix` | `dataset_statistics_map` target |
| --- | --- | --- | --- |
| `PnP` | `barx-sp900-<robot>-pnp` | `barx-sp900-<robot>-pnp-mix` | `{"barx-target-<robot>-pnp": "barx-sp900-<robot>-pnp"}` |
| `Turn On Sink Faucet` | `barx-sp900-<robot>-turn-on-sink-faucet` | `barx-sp900-<robot>-turn-on-sink-faucet-mix` | `{"barx-target-<robot>-turn-on-sink-faucet": "barx-sp900-<robot>-turn-on-sink-faucet"}` |
| `Flip Mug Upright` | `barx-sp900-<robot>-flip-mug-upright` | `barx-sp900-<robot>-flip-mug-upright-mix` | `{"barx-target-<robot>-flip-mug-upright": "barx-sp900-<robot>-flip-mug-upright"}` |

The corresponding same-embodiment tokenizers follow the same pattern:

- `barx-sp900-<robot>-pnp-vq-extra-action-tokenizer`
- `barx-sp900-<robot>-turn-on-sink-faucet-vq-extra-action-tokenizer`
- `barx-sp900-<robot>-flip-mug-upright-vq-extra-action-tokenizer`

### Target-Only Baselines

For no-prior target-only training:

- use `data_mix="barx-target-<robot>-<task>"`
- use the matching target tokenizer `barx-target-<robot>-<task>-vq-extra-action-tokenizer`
- omit `--pretrained_checkpoint`
- omit `--dataset_statistics_map`

Default target schedules from the released BARX scripts:

- `PnP`: `max_steps=3000`, `save_interval=1000`
- `Turn On Sink Faucet`: `max_steps=2000`, `save_interval=500`
- `Flip Mug Upright`: `max_steps=2000`, `save_interval=500`

## Prior Training Template

Use this template for cross-embodiment or same-embodiment prior training.

```bash
export PRIOR_DATA_MIX=barx-xp900-pnp
export ACTION_TOKENIZER=barx-xp900-pnp-vq-extra-action-tokenizer
export RUN_ID=aux
export RUN_NOTE=barx-xp900-pnp
export TRANSFORM_TYPES='bbox->,low_level_motion->,ee_pose_2D->,action'
export MAX_STEPS=100000
export SAVE_INTERVAL=10000

torchrun --standalone --nnodes 1 --nproc-per-node "$NUM_GPUS" vla-scripts/train.py \
    --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-xembod-robocasa-full \
    --vla.base_vlm "$BARX_BASE_VLM" \
    --vla.data_mix "$PRIOR_DATA_MIX" \
    --data_root_dir "$BARX_DATA_ROOT" \
    --run_root_dir "$BARX_PRIOR_ROOT" \
    --vla.action_tokenizer "$ACTION_TOKENIZER" \
    --vla.expected_world_size "$NUM_GPUS" \
    --vla.global_batch_size "$GLOBAL_BATCH_SIZE" \
    --vla.per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --vla.lr_scheduler_type constant \
    --vla.max_steps "$MAX_STEPS" \
    --vla.use_wrist_image False \
    --vla.image_sequence_len 1 \
    --wandb_entity "$WANDB_ENTITY" \
    --run_id_note "$RUN_NOTE" \
    --run_id "$RUN_ID" \
    --vla.transform_types "$TRANSFORM_TYPES" \
    --is_resume False \
    --save_interval "$SAVE_INTERVAL"
```

For an exact paper rerun, copy the task-specific `max_steps`, `save_interval`, and any resume fields from the released
`config.json` for the prior checkpoint you are recreating.

## Finetuning Template

Use this template to finetune a target robot from a released prior checkpoint.

```bash
export TARGET_DATA_MIX=barx-xp3k-panda-pnp-mix
export TARGET_DATASET=barx-target-panda-pnp
export ACTION_TOKENIZER=barx-xp3k-pnp-vq-extra-action-tokenizer
export PRETRAIN_CKPT=$BARX_PRIOR_ROOT/aux--barx-xp3k-pnp/checkpoints/<step-file>.pt
export RUN_ID=aux
export RUN_NOTE=barx-xp3k-panda-pnp-mix
export TRANSFORM_TYPES='bbox->,low_level_motion->,ee_pose_2D->,action'
export MAX_STEPS=3000
export SAVE_INTERVAL=1000
export DATASET_STATS_JSON='{"barx-target-panda-pnp":"barx-xp3k-pnp"}'

torchrun --standalone --nnodes 1 --nproc-per-node "$NUM_GPUS" vla-scripts/train.py \
    --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-xembod-robocasa-full \
    --vla.base_vlm "$BARX_BASE_VLM" \
    --vla.data_mix "$TARGET_DATA_MIX" \
    --data_root_dir "$BARX_DATA_ROOT" \
    --run_root_dir "$BARX_FT_ROOT" \
    --vla.action_tokenizer "$ACTION_TOKENIZER" \
    --vla.expected_world_size "$NUM_GPUS" \
    --vla.global_batch_size "$GLOBAL_BATCH_SIZE" \
    --vla.per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
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
    --save_interval "$SAVE_INTERVAL"
```

If you are training a target-only baseline, remove:

- `--pretrained_checkpoint`
- `--dataset_statistics_map`

and use a target alias as `TARGET_DATA_MIX`, for example `barx-target-panda-pnp`.

## Concrete Example: Panda PnP Joint Reps from the XP-3K Prior

```bash
export TARGET_DATA_MIX=barx-xp3k-panda-pnp-mix
export ACTION_TOKENIZER=barx-xp3k-pnp-vq-extra-action-tokenizer
export PRETRAIN_CKPT=$BARX_PRIOR_ROOT/aux--barx-xp3k-pnp/checkpoints/<step-file>.pt
export DATASET_STATS_JSON='{"barx-target-panda-pnp":"barx-xp3k-pnp"}'

torchrun --standalone --nnodes 1 --nproc-per-node "$NUM_GPUS" vla-scripts/train.py \
    --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-xembod-robocasa-full \
    --vla.base_vlm "$BARX_BASE_VLM" \
    --vla.data_mix "$TARGET_DATA_MIX" \
    --data_root_dir "$BARX_DATA_ROOT" \
    --run_root_dir "$BARX_FT_ROOT" \
    --vla.action_tokenizer "$ACTION_TOKENIZER" \
    --vla.expected_world_size "$NUM_GPUS" \
    --vla.global_batch_size "$GLOBAL_BATCH_SIZE" \
    --vla.per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --vla.lr_scheduler_type constant \
    --vla.max_steps 3000 \
    --vla.use_wrist_image False \
    --vla.image_sequence_len 1 \
    --wandb_entity "$WANDB_ENTITY" \
    --run_id_note "barx-xp3k-panda-pnp-mix" \
    --run_id aux \
    --pretrained_checkpoint "$PRETRAIN_CKPT" \
    --is_resume False \
    --dataset_statistics_map "$DATASET_STATS_JSON" \
    --vla.transform_types "bbox->,low_level_motion->,ee_pose_2D->,action" \
    --vla.warmup_ratio 0.02 \
    --save_interval 1000
```

## Action-Free Variants

BARX also used action-free variants in some experiments.

Historical scripts used:

- a non-action prior stage such as `run_id="aux_no_act"` with `transform_types="bbox,ee_pose_2D"`
- a target finetune stage that still uses the standard joint-reps transform string but adds:

```bash
--non_action_datasets '["barx-xp900-pnp"]'
```

and points `--pretrained_checkpoint` at the released action-free prior.

The exact choice between `aux_no_act` and `aux_only` should come from the release manifest for the checkpoint set you ship.

<!-- ## Troubleshooting

If TFDS loading breaks:

```bash
pip install tensorflow-datasets==4.9.3
```

If `dlimp` incompatibilities show up:

```bash
pip install --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla
```

If a BARX VQ tokenizer fails to load, check:

- the `vq/` symlink or directory exists
- the expected `vq/<name>/config.json` exists
- the expected `vq/<name>/checkpoints/model.pt` exists -->

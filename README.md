# BARX: Cross-Embodiment Transfer via Behavior-Aligned Representations

Official simulation-only code release for **BARX** and the **RoboCasa-X**
benchmark, accompanying the ICRA 2026 paper
*Cross-Embodiment Transfer via Behavior-Aligned Representations*.

[Project website](https://ajaysridhar.com/barx/)

## Release contents

This repository contains:

- MiniVLA policy training and RoboCasa-X evaluation code;
- the modified RoboCasa-X benchmark and an exact robosuite 1.5.1 Git pin;
- one action layout shared by every simulator embodiment and the RLDS
  converter;
- a manifest describing the 283.13 GiB simulation-data collection; and
- paper-aligned experiment names, configuration, and protocols.

The simulation HDF5 files use a separate data archive; see
[`dataset/README.md`](dataset/README.md) for its composition and preparation.

Public model and dataset artifacts are hosted in the
[BARX Hugging Face collections](https://huggingface.co/collections/ajaysri/barx-pretraining-models-joint-reps-and-no-reps).
They are ungated and do not require a Hugging Face token.

## Repository layout

| Path | Purpose |
| --- | --- |
| `barx/` | lightweight naming and normalized action-space API |
| `policy/` | BARX policy training and RoboCasa-X evaluation |
| `robocasa_x/` | modified RoboCasa benchmark runtime |
| `dataset/rlds/` | unified HDF5-to-RLDS converter |
| `configs/experiments.toml` | paper protocol and compatibility identifiers |
| `scripts/` | training, evaluation, RLDS conversion, and result summaries |
| `tests/` | dependency-light compatibility tests |

MimicGen produced the synthetic demonstrations. The final rendered HDF5 files
form the training-data boundary, while evaluation uses the local RoboCasa
compatibility wrappers and the locked PyPI robomimic dependency.

## Quick start

The supported release platform is Linux x86-64 with Python 3.10 and CUDA 12.1.
Install [uv](https://docs.astral.sh/uv/) 0.11.11 or newer, then run:

```bash
uv sync --locked --no-dev
uv run --locked --no-dev python -m unittest discover -s tests -v
```

Before simulation or evaluation, download the upstream RoboCasa kitchen asset
bundle (about 3.5 GiB compressed and 8.8 GiB installed):

```bash
uv run --locked --no-dev python robocasa_x/robocasa/scripts/download_kitchen_assets.py --yes
uv run --locked --no-dev python -c "import robocasa, robosuite; print('RoboCasa-X import OK')"
```

Choose one external artifact directory, then download the base VLM, XP-900 PnP
RLDS data, its VQ tokenizer, and the pinned DINOv2, SigLIP, and Qwen runtime
dependencies. Every Hugging Face snapshot is locked to the revision recorded
in `configs/public_artifacts.json`.

```bash
export BARX_ARTIFACT_ROOT="$(pwd)/../barx-artifacts"
export HF_HOME="$BARX_ARTIFACT_ROOT/hf"
export BARX_VQ_ROOT="$BARX_ARTIFACT_ROOT/vq"
uv run --locked --no-dev python scripts/download_public_artifacts.py \
  --artifact-root "$BARX_ARTIFACT_ROOT"
```

This fetches only the runtime base checkpoint, not the three historical copies
or W&B files in its source repository. Plan for at least 60 GiB of free space
for the walkthrough and environment/cache. Add `--include-pretrain-checkpoint`
to also fetch the released Joint Reps source-prior run (about 5.2 GiB).

For training with FlashAttention and a CUDA development toolkit:

```bash
uv sync --locked --extra train --no-dev
```

The complete one-A40 initialization and optimizer smoke test is:

```bash
uv run --locked --extra train --no-dev python scripts/train.py prior \
  --prior xp_900 --task pnp --method joint_reps \
  --data-root "$BARX_ARTIFACT_ROOT/data" \
  --base-vlm "$BARX_ARTIFACT_ROOT/base-vlm" \
  --run-root "$BARX_ARTIFACT_ROOT/runs/smoke" \
  --gpus 1 --global-batch-size 1 --per-device-batch-size 1 \
  --max-steps 1 --save-interval 100 --skip-final-checkpoint
```

For a single simulator trial of the released checkpoint, rerun the artifact
download with `--include-pretrain-checkpoint`, then execute:

```bash
uv run --locked --extra train --no-dev python scripts/evaluate.py \
  --checkpoint "$BARX_ARTIFACT_ROOT/runs/xp900-pnp-joint-reps/checkpoints/step-050000-epoch-15-loss=0.2577.pt" \
  --embodiment panda --task pnp_counter_to_sink --unnorm-key mg_pnp_lite \
  --episodes 1 --rollout-dir "$BARX_ARTIFACT_ROOT/rollouts/one-trial"
```

This one trial is an execution check, not a success-rate estimate; the paper
protocol uses 100 held-out seeds per setting.

See [`docs/installation.md`](docs/installation.md) for system requirements,
[`docs/data.md`](docs/data.md) for dataset preparation, and
[`docs/experiments.md`](docs/experiments.md) for training and evaluation.

## Paper names and stored names

Public APIs use the names from the paper. Existing dataset fields and model
internals retain their original names so that conversion results and future
checkpoint releases remain compatible.

| Paper name | Stored/internal identifier |
| --- | --- |
| No Reps | `base` / `action` |
| Joint Reps | `aux` / `bbox->,low_level_motion->,ee_pose_2D->,action` |
| ECoT | `chain` / `bbox->ee_pose_2D->low_level_motion->,action` |
| bounding box | `bbox` |
| language motion | `low_level_motion` |
| end-effector trace | `ee_pose_2D` |
| XP-900 | `mg_*_lite` |
| XP-3K | `mg_*` |

Both forms are accepted at compatibility boundaries, but documentation and new
entry points use only paper names.

## Citation

```bibtex
@inproceedings{sridhar2026barx,
  title     = {Cross-Embodiment Transfer via Behavior-Aligned Representations},
  author    = {Sridhar, Ajay and Gao, Jensen and Yang, Jonathan and Mercat, Jean and Belkhale, Suneel and Sadigh, Dorsa},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026}
}
```

## Licenses

New BARX code is MIT licensed. Vendored upstream components retain their own
license files. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

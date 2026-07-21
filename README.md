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
uv run --locked python -m unittest discover -s tests -v
```

Before simulation or evaluation, download the upstream RoboCasa kitchen asset
bundle (about 5.8 GiB) through its standard distribution workflow:

```bash
uv run --locked python robocasa_x/robocasa/scripts/download_kitchen_assets.py
```

Download the base VLM, XP-900 PnP RLDS data, and its VQ tokenizer into the
runtime-compatible directory names:

```bash
uv run --locked python scripts/download_public_artifacts.py \
  --data-root /data/barx-rlds \
  --base-vlm-dir /models/barx-base
```

Plan for at least 60 GiB of free space for that training walkthrough and the
environment/cache. Add `--include-pretrain-checkpoint` to also download the
released Joint Reps source-prior run.

For training with FlashAttention and a CUDA development toolkit:

```bash
uv sync --locked --extra train --no-dev
```

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

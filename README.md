# BARX: Cross-Embodiment Transfer via Behavior-Aligned Representations

Official simulation-only code and data release for **BARX** and the
**RoboCasa-X** benchmark, accompanying the ICRA 2026 paper
*Cross-Embodiment Transfer via Behavior-Aligned Representations*.

[Project website](https://ajaysridhar.com/barx/)

## Release scope

This repository contains:

- MiniVLA policy training and RoboCasa-X evaluation code;
- the modified RoboCasa-X benchmark and an unmodified, pinned robosuite
  1.5.1 source snapshot;
- one action layout shared by every simulator embodiment and the RLDS
  converter;
- a manifest for the 283.13 GiB simulation-data release; and
- paper-aligned experiment names, configuration, and protocols.

It intentionally excludes model and VQ tokenizer checkpoints, real-robot code
and data, failed/raw intermediate demonstrations, and cluster-specific logs.
The simulation HDF5 files are distributed separately from Git; see
[`dataset/README.md`](dataset/README.md).

## Repository layout

| Path | Purpose |
| --- | --- |
| `barx/` | lightweight naming and normalized action-space API |
| `policy/` | BARX policy training and RoboCasa-X evaluation |
| `simulator/robocasa_x/` | modified RoboCasa benchmark |
| `simulator/robosuite/` | unmodified robosuite dependency at pinned upstream commit |
| `dataset/rlds/` | unified HDF5-to-RLDS converter |
| `configs/experiments.toml` | paper protocol and compatibility identifiers |
| `scripts/` | release-data and experiment entry points |
| `tests/` | dependency-light compatibility tests |

The `mg_robocasa_xembod` and custom `robomimic_xembod` repositories are not
included. MimicGen was used to create the released simulation data, but it is
not required to train or evaluate from the final files. Evaluation uses the
local RoboCasa compatibility wrappers; the locked PyPI robomimic dependency is
sufficient.

## Quick start

The supported release platform is Linux x86-64 with Python 3.10 and CUDA 12.1.
Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync --locked --no-dev
uv run --locked python -m unittest discover -s tests -v
```

Before simulation or evaluation, download the upstream RoboCasa kitchen asset
bundle (about 5.8 GiB, excluded from Git by its upstream license/distribution
workflow):

```bash
uv run --locked python simulator/robocasa_x/robocasa/scripts/download_kitchen_assets.py
```

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
license files. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The
simulation dataset license must be selected before the data host is published.

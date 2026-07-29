# BARX: Cross-Embodiment Transfer via Behavior-Aligned Representations

This is the simulation-only release for **BARX** and the **RoboCasa-X**
benchmark from the ICRA 2026 paper
*Cross-Embodiment Transfer via Behavior-Aligned Representations*.

[Project website](https://ajaysridhar.com/barx/)

## What's included

This repository includes:

- code for training MiniVLA policies and evaluating them in RoboCasa-X;
- the RoboCasa-X simulation benchmark;
- tools for converting the datasets;
- MimicGen support for generating new demonstrations from the human data; and
- the experiment configurations used in the paper.

For training and evaluation, start with the processed RLDS datasets used by the
quick start below. The raw HDF5 demonstrations are optional for users who want
to inspect simulator states, convert the data into another format, or generate
new demonstrations with MimicGen.

You can download the models and data from the
[BARX model collections](https://huggingface.co/collections/ajaysri/barx-pretraining-models-joint-reps-and-no-reps),
[processed RLDS collection](https://huggingface.co/collections/ajaysri/barx-rlds-datasets-69cad164926390ebbc395496),
and [raw HDF5 collection](https://huggingface.co/collections/ajaysri/barx-raw-hdf5-data-6a61b2d60e2a7ca90b75fb68).
No Hugging Face token is needed.

## Quick start

BARX is tested on Linux x86-64 with Python 3.10 and CUDA 12.1. The walkthrough
needs about 70 GiB of free space. Install
[uv](https://docs.astral.sh/uv/) 0.11.11 or newer and check that CMake is
available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
command -v cmake
cmake --version
```

If either CMake command fails, install CMake before running `uv sync`.

Then install BARX and run its tests:

```bash
uv sync --locked --no-dev
uv run --locked --no-dev python -m unittest discover -s tests -v
```

Download the RoboCasa kitchen assets:

```bash
uv run --locked --no-dev python robocasa_x/robocasa/scripts/download_kitchen_assets.py --yes
uv run --locked --no-dev python -c "import robocasa, robosuite; print('RoboCasa-X import OK')"
```

Choose a directory and download the processed XP-900 PnP RLDS data, base
model, and VQ tokenizer used by this walkthrough:

```bash
export BARX_ARTIFACT_ROOT="$(pwd)/../barx-artifacts"
export HF_HOME="$BARX_ARTIFACT_ROOT/hf"
export BARX_VQ_ROOT="$BARX_ARTIFACT_ROOT/vq"
uv run --locked --no-dev python scripts/download_public_artifacts.py \
  --artifact-root "$BARX_ARTIFACT_ROOT"
```

Install the training dependencies:

```bash
uv sync --locked --extra train --no-dev
```

Run a one-GPU, one-step training check:

```bash
uv run --locked --extra train --no-dev python scripts/train.py prior \
  --prior xp_900 --task pnp --method joint_reps \
  --data-root "$BARX_ARTIFACT_ROOT/data" \
  --base-vlm "$BARX_ARTIFACT_ROOT/base-vlm" \
  --run-root "$BARX_ARTIFACT_ROOT/runs/quick-check" \
  --gpus 1 --global-batch-size 1 --per-device-batch-size 1 \
  --max-steps 1 --save-interval 100 --skip-final-checkpoint
```

To try the released Joint Reps checkpoint, first add
`--include-pretrain-checkpoint` to the download command above. Check that both
files are present:

```bash
test -f "$BARX_ARTIFACT_ROOT/runs/xp900-pnp-joint-reps/checkpoints/step-050000-epoch-15-loss=0.2577.pt"
test -f "$BARX_VQ_ROOT/mg_pnp_lite/checkpoints/model.pt"
```

Run one simulator trial:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
uv run --locked --extra train --no-dev python scripts/evaluate.py \
  --checkpoint "$BARX_ARTIFACT_ROOT/runs/xp900-pnp-joint-reps/checkpoints/step-050000-epoch-15-loss=0.2577.pt" \
  --embodiment panda --task pnp_counter_to_sink --unnorm-key mg_pnp_lite \
  --episodes 1 --rollout-dir "$BARX_ARTIFACT_ROOT/rollouts/one-trial"
```

The benchmark uses 100 fixed trials per task and embodiment. Pick-and-place
targets come from RoboCasa `obj_set1`, instance split `A`. The included trials
use the task-specific scene filters and keep the target visible to the policy
camera. Omit `--episodes 1` to run the full set. See
[`evaluation/README.md`](evaluation/README.md) for the complete protocol.

### Optional: generate new data with MimicGen

The training and evaluation steps above do not require raw HDF5 data. To
generate new demonstrations from the human data, install the optional MimicGen
dependencies:

```bash
uv sync --locked --extra mg --no-dev
```

Follow the
[MimicGen guide](dataset/README.md#extending-the-demonstrations-with-mimicgen)
to download only the human HDF5 subset you need, prepare it, generate a small
dataset, and save a video for review.

## Guides

- [`docs/installation.md`](docs/installation.md): system requirements and
  installation
- [`dataset/README.md`](dataset/README.md): optional raw HDF5 downloads, custom
  RLDS conversion, and MimicGen generation
- [`docs/data.md`](docs/data.md): dataset aliases and artifact details
- [`docs/experiments.md`](docs/experiments.md): paper configurations, training,
  and evaluation
- [`evaluation/README.md`](evaluation/README.md): fixed evaluation conditions
  and result format

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

- New BARX code is MIT licensed.
- BARX raw HDF5 and processed RLDS datasets are released under CC BY 4.0.
- BARX base-model, VQ-tokenizer, and policy-checkpoint artifacts are released
  under Apache 2.0, subject to the separately identified upstream components
  and their terms.
- The vendored MimicGen source retains NVIDIA's non-commercial research and
  evaluation license; it is not covered by BARX's MIT license.

See [`docs/artifact_licenses.md`](docs/artifact_licenses.md) for artifact
scope, provenance, attribution, and reuse guidance, and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for bundled software.

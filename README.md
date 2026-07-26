# BARX: Cross-Embodiment Transfer via Behavior-Aligned Representations

Official simulation-only code release for **BARX** and the **RoboCasa-X**
benchmark, accompanying the ICRA 2026 paper
*Cross-Embodiment Transfer via Behavior-Aligned Representations*.

[Project website](https://ajaysridhar.com/barx/)

## What's included

This repository contains:

- MiniVLA policy training and RoboCasa-X evaluation code;
- the RoboCasa-X simulation benchmark;
- dataset conversion tools;
- a separately licensed MimicGen compatibility snapshot with BARX task
  preparation and bounded generation commands; and
- reproducible experiment configurations.

The raw HDF5 demonstrations can be read directly, converted to RLDS, or used
as the basis for new data-generation pipelines; see
[`dataset/README.md`](dataset/README.md).
Public artifacts are available from the
[BARX model collections](https://huggingface.co/collections/ajaysri/barx-pretraining-models-joint-reps-and-no-reps),
[processed RLDS collection](https://huggingface.co/collections/ajaysri/barx-rlds-datasets-69cad164926390ebbc395496),
and [raw HDF5 collection](https://huggingface.co/collections/ajaysri/barx-raw-hdf5-data-6a61b2d60e2a7ca90b75fb68).
They are ungated and do not require a Hugging Face token.

## Quick start

The supported release platform is Linux x86-64 with Python 3.10 and CUDA 12.1.
The walkthrough with the released checkpoint requires about 70 GiB of free
space. Install [uv](https://docs.astral.sh/uv/) 0.11.11 or newer and check that
CMake is available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
command -v cmake
cmake --version
```

If `command -v cmake` selects a broken user-local launcher, repair `PATH` or
select a working system CMake before running `uv sync`. Merely having a file
named `cmake` on `PATH` is not sufficient.

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

Choose a directory for the XP-900 PnP walkthrough artifacts and download them:

```bash
export BARX_ARTIFACT_ROOT="$(pwd)/../barx-artifacts"
export HF_HOME="$BARX_ARTIFACT_ROOT/hf"
export BARX_VQ_ROOT="$BARX_ARTIFACT_ROOT/vq"
uv run --locked --no-dev python scripts/download_public_artifacts.py \
  --artifact-root "$BARX_ARTIFACT_ROOT"
```

The downloader defaults to one worker for conservative public operation and
resumes partial Hugging Face downloads. On a reliable connection, add
`--max-workers 8` to download independent RLDS shards concurrently.

Install the training dependencies:

```bash
uv sync --locked --extra train --no-dev
```

Run a one-GPU training smoke test:

```bash
uv run --locked --extra train --no-dev python scripts/train.py prior \
  --prior xp_900 --task pnp --method joint_reps \
  --data-root "$BARX_ARTIFACT_ROOT/data" \
  --base-vlm "$BARX_ARTIFACT_ROOT/base-vlm" \
  --run-root "$BARX_ARTIFACT_ROOT/runs/smoke" \
  --gpus 1 --global-batch-size 1 --per-device-batch-size 1 \
  --max-steps 1 --save-interval 100 --skip-final-checkpoint
```

To evaluate the released Joint Reps checkpoint, first add
`--include-pretrain-checkpoint` to the artifact download command above. Then
verify the model and VQ paths before paying model-loading cost:

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

The paper protocol uses 100 trials; omit `--episodes 1` to run the full set.

To regenerate data from the released human HDF5 demonstrations, install the
separately licensed MimicGen extra:

```bash
uv sync --locked --extra mg --no-dev
```

The complete source-preparation, bounded one-success generation, HDF5
verification, and video-review commands are in
[Section 8 of the release walkthrough](RELEASE_TEST_README.md#8-mimicgen-regeneration-gate).

See [`docs/installation.md`](docs/installation.md) for system requirements,
[`docs/data.md`](docs/data.md) for dataset preparation, and
[`docs/experiments.md`](docs/experiments.md) for training and evaluation. The
maintainer launch gates are tracked in
[`docs/release_checklist.md`](docs/release_checklist.md). Independent release
testers should use
[`RELEASE_TEST_README.md`](RELEASE_TEST_README.md) to capture machine-checkable
and human-reviewable evidence for the complete paper-facing workflow. The
latest independent result is summarized in
[`RELEASE_ACCEPTANCE_REPORT.md`](RELEASE_ACCEPTANCE_REPORT.md), with every
sanitized command, attempt, exit code, duration, and output hash in
[`RELEASE_TEST_COMMAND_LOG.md`](RELEASE_TEST_COMMAND_LOG.md).

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

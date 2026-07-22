# BARX: Cross-Embodiment Transfer via Behavior-Aligned Representations

Official simulation-only code release for **BARX** and the **RoboCasa-X**
benchmark, accompanying the ICRA 2026 paper
*Cross-Embodiment Transfer via Behavior-Aligned Representations*.

[Project website](https://ajaysridhar.com/barx/)

## What's included

This repository contains:

- MiniVLA policy training and RoboCasa-X evaluation code;
- the RoboCasa-X simulation benchmark;
- dataset conversion tools; and
- reproducible experiment configurations.

Simulation data is documented in [`dataset/README.md`](dataset/README.md).
Public models and datasets are available from the
[BARX Hugging Face collections](https://huggingface.co/collections/ajaysri/barx-pretraining-models-joint-reps-and-no-reps).
They are ungated and do not require a Hugging Face token.

## Quick start

The supported release platform is Linux x86-64 with Python 3.10 and CUDA 12.1.
The walkthrough with the released checkpoint requires about 70 GiB of free
space. Install [uv](https://docs.astral.sh/uv/) 0.11.11 or newer and check that
CMake is available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
cmake --version
```

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
run one simulator trial:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
uv run --locked --extra train --no-dev python scripts/evaluate.py \
  --checkpoint "$BARX_ARTIFACT_ROOT/runs/xp900-pnp-joint-reps/checkpoints/step-050000-epoch-15-loss=0.2577.pt" \
  --embodiment panda --task pnp_counter_to_sink --unnorm-key mg_pnp_lite \
  --episodes 1 --rollout-dir "$BARX_ARTIFACT_ROOT/rollouts/one-trial"
```

The paper protocol uses 100 trials; omit `--episodes 1` to run the full set.

See [`docs/installation.md`](docs/installation.md) for system requirements,
[`docs/data.md`](docs/data.md) for dataset preparation, and
[`docs/experiments.md`](docs/experiments.md) for training and evaluation.

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

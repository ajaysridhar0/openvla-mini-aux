# Installation and environment reproducibility

BARX uses uv for the public release. The original experiments ran in a Conda
environment named `robocasa`; that environment remains the provenance
reference but is not required by users. Use uv 0.11.11 or newer; older uv
versions do not understand every lock and source field used by this project.

## Supported platform

- Linux x86-64
- Python 3.10 (uv selects 3.10 from `.python-version`)
- NVIDIA driver compatible with CUDA 12.1 for policy training/evaluation
- MuJoCo 3.1.1
- at least 60 GiB free for the XP-900 PnP walkthrough (70 GiB when retaining
  the released pretraining checkpoint)

From the repository root:

```bash
uv sync --locked --no-dev
uv run --locked --no-dev python -m unittest discover -s tests -v
```

uv installs `policy` and `robocasa_x` editably from this repository.
robosuite, dlimp, and VQ-BeT are pinned to immutable Git commits in `uv.lock`.

Download RoboCasa's Objaverse models and high-resolution kitchen textures
through the static upstream asset archives before constructing an environment:

```bash
uv run --locked --no-dev python robocasa_x/robocasa/scripts/download_kitchen_assets.py --yes
uv run --locked --no-dev python -c "import robocasa, robosuite; print('RoboCasa-X import OK')"
```

The download is about 3.5 GiB compressed and occupies about 8.8 GiB after
extraction. The script places assets under the editable
`robocasa_x/robocasa/models/assets/` tree, where the simulator
expects them. A missing bundle causes task construction to fail before an
episode begins; dependency-only tests do not require it.

The downloader verifies the published byte count and SHA-256 of each archive,
extracts into a temporary directory, and exits nonzero without replacing a
working asset tree if a download or verification fails. Omit `--yes` for an
interactive confirmation, or use repeated `--asset NAME` flags to install only
specific groups.

For training, install the optional FlashAttention build:

```bash
uv sync --locked --extra train --no-dev
```

FlashAttention 2.5.5 must compile against the locked Torch 2.2.0+cu121. A CUDA
development toolkit, Ninja, a compatible C/C++ compiler, and a compatible
NVIDIA driver are required. Policy evaluation loads the same attention
implementation, so install the `train` extra for both training and policy
evaluation. Simulation-only and compatibility tests do not require it.

## Historical environment differences

The working Conda environment used NumPy 1.23.3. The uv release uses NumPy
1.23.5 because TensorFlow 2.15 requires at least 1.23.5. Both are maintenance
releases in the same NumPy series. The intervening release notes contain no
change to random routines used by BARX; this is not a promise of bitwise
identity across every NumPy operation, so the historical 1.23.3 environment
remains recorded for archival reruns. The original environment used Protobuf 3.20.3
despite an outdated `<3.20` constraint in Tianshou 0.4.10, so the uv project
records an explicit Protobuf 3.20.3 override.

robosuite is pinned to public upstream commit
`2ebb2a0249f7a271a3ade56725d12e32d7da8898`. Upstream 1.5.1 package metadata
lists Mink and MuJoCo 3.2.3+, but BARX does not use the Mink example controller
and the experiments used MuJoCo 3.1.1. The root uv project records that
release-environment override without forking third-party source.

## Reproducible operation

Use `--locked` in published commands. `uv sync --locked` fails if
`pyproject.toml` and `uv.lock` disagree instead of silently updating packages.
Do not run `uv lock --upgrade` when reproducing paper results.

## Expected first-run warnings and pauses

RoboCasa/robosuite may warn that no private macro file exists, Mink whole-body
IK is unavailable, or MimicGen is not installed. MimicGen is required only for
the Section 8 regeneration workflow and is installed with the `mg` extra.
TensorFlow may report duplicate CUDA plugin
registration, missing TensorRT, and a Transformers cache migration; an absent
`OpenGL_accelerate` module is also optional. These messages are non-fatal if
the command continues and ultimately exits zero.

The artifact downloader prefetches the pinned DINOv2, SigLIP, and Qwen files
into `$HF_HOME`. Model initialization can still be quiet for several minutes
while multi-gigabyte checkpoints are read from a network filesystem. Treat an
exception, nonzero exit, checksum failure, CUDA out-of-memory error, or missing
asset/checkpoint message as actionable. `hf_xet` is optional and only speeds
Hugging Face transfers.

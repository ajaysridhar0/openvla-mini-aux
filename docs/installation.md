# Installation

BARX uses uv 0.11.11 or newer to install the environment from `uv.lock`.

## Supported platform

- Linux x86-64
- Python 3.10 (uv selects 3.10 from `.python-version`)
- NVIDIA driver compatible with CUDA 12.1 for policy training/evaluation
- MuJoCo 3.1.1
- about 70 GiB free for the XP-900 PnP walkthrough

From the repository root:

```bash
command -v cmake
cmake --version
uv sync --locked --no-dev
uv run --locked --no-dev python -m unittest discover -s tests -v
```

If either CMake command fails, install CMake before running `uv sync`.

Download RoboCasa's Objaverse models and high-resolution kitchen textures
before constructing an environment:

```bash
uv run --locked --no-dev python robocasa_x/robocasa/scripts/download_kitchen_assets.py --yes
uv run --locked --no-dev python -c "import robocasa, robosuite; print('RoboCasa-X import OK')"
```

The download is about 3.5 GiB and occupies about 8.8 GiB after extraction.

Omit `--yes` for an interactive confirmation, or use repeated `--asset NAME`
flags to install only specific groups.

For training, install the optional FlashAttention build:

```bash
uv sync --locked --extra train --no-dev
```

FlashAttention 2.5.5 must compile against the locked Torch 2.2.0+cu121. A CUDA
development toolkit, Ninja, a compatible C/C++ compiler, and a compatible
NVIDIA driver are required. Policy evaluation loads the same attention
implementation, so install the `train` extra for both training and policy
evaluation. Basic simulator tests do not require it.

## Expected first-run warnings and pauses

RoboCasa/robosuite may warn about missing private macros or Mink whole-body IK.
TensorFlow may also report duplicate CUDA plugins, missing TensorRT, or a cache
migration. These warnings are safe to ignore if the command keeps running.

Model initialization can be quiet for several minutes while checkpoints load.
A nonzero exit, CUDA out-of-memory error, or missing asset/checkpoint message
needs attention.

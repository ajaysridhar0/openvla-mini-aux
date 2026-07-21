# Installation and environment reproducibility

BARX uses uv for the public release. The original experiments ran in a Conda
environment named `robocasa`; that environment remains the provenance
reference but is not required by users.

## Supported platform

- Linux x86-64
- Python 3.10 (uv selects 3.10 from `.python-version`)
- NVIDIA driver compatible with CUDA 12.1 for policy training/evaluation
- MuJoCo 3.1.1

From the repository root:

```bash
uv sync --locked --no-dev
uv run --locked python -m unittest discover -s tests -v
```

uv installs `policy` and `robocasa_x` editably from this repository.
robosuite, dlimp, and VQ-BeT are pinned to immutable Git commits in `uv.lock`.

Download RoboCasa's Objaverse models and high-resolution kitchen textures
through the static upstream asset archives before constructing an environment:

```bash
uv run --locked python robocasa_x/robocasa/scripts/download_kitchen_assets.py
```

The download is about 5.8 GiB. The script places assets under the editable
`robocasa_x/robocasa/models/assets/` tree, where the simulator
expects them. A missing bundle causes task construction to fail before an
episode begins; dependency-only tests do not require it.

For training, install the optional FlashAttention build:

```bash
uv sync --locked --extra train --no-dev
```

FlashAttention 2.5.5 must compile against the locked Torch 2.2.0+cu121. A CUDA
development toolkit and compatible compiler are required. Simulation and the
dependency-light compatibility tests do not require this extra.

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

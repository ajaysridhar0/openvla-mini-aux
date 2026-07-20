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

uv installs `policy`, `simulator/robocasa_x`, and `simulator/robosuite`
editably from this repository. It also pins dlimp and VQ-BeT to Git commit
hashes in `uv.lock`.

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

robosuite 1.5.1 listed Mink as a mandatory dependency even though it is used
only by an example controller and requires a newer MuJoCo. BARX removes Mink
from mandatory dependencies and retains MuJoCo 3.1.1, which RoboCasa-X asserts
at import time and which was used in the experiments.

## Reproducible operation

Use `--locked` in published commands. `uv sync --locked` fails if
`pyproject.toml` and `uv.lock` disagree instead of silently updating packages.
Do not run `uv lock --upgrade` when reproducing paper results.

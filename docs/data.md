# Data format and conversion

Use the released processed RLDS repositories for standard BARX training and
evaluation. The raw HDF5 archive and conversion tools below are optional for
users who want source simulator states, another data format, a fresh RLDS
conversion, or new MimicGen generation.

The optional raw HDF5 release contains 23,400 demonstrations and is available
from
[`ajaysri/barx-raw-hdf5`](https://huggingface.co/datasets/ajaysri/barx-raw-hdf5).
See [`dataset/README.md`](../dataset/README.md) for full-archive and selective
download commands, its contents, and its schema.

The raw HDF5 and processed RLDS datasets are released under CC BY 4.0. See
[`artifact_licenses.md`](artifact_licenses.md) for reuse guidance.

## Canonical action boundary

Every released RoboCasa-X HDF5 action uses:

`[arm(6), gripper(1), base(3), torso(1), mode(1)]`

Policies and RLDS datasets expose the first seven values.

## Paper dataset aliases

The training registry accepts paper-facing aliases such as `xp_900_pnp`,
`xp_3k_pnp`, `panda_xp_900_pnp`, and `jaco_sp_900_pnp`. They resolve to the
original RLDS directory identifiers; users do not need to rename previously
converted datasets.

The HDF5 files can be consumed directly with `h5py` or another HDF5 reader.
To build BARX's training-compatible RLDS directories:

```bash
uv run --locked --no-dev python scripts/build_rlds.py \
  --dataset xp_3k --task flip_mug \
  --raw-root /data/barx --rlds-root /data/barx-rlds
```

Valid paper datasets are `xp_900`, `xp_3k`, `sp_900`, and `target_50`.
`sp_900` and `target_50` additionally require `--target panda`, `panda_og`, or
`jaco`.

The `human/` files include the simulator states and metadata needed to make new
MimicGen datasets. Install the optional MimicGen dependencies with
`uv sync --locked --extra mg --no-dev`; the preparation, generation, and video
commands are in
[`dataset/README.md`](../dataset/README.md#extending-the-demonstrations-with-mimicgen).
The source-preparation wrapper writes a new HDF5 rather than modifying the
downloaded human demonstrations.

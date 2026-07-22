# Data format and conversion

The raw HDF5 dataset is recorded and checksummed in `dataset/manifest.csv`; see
`dataset/README.md` for its composition, verification command, and schema. The
archive preserves the relative `mg/` and `human/` paths.

Retained episode metadata replaces installation prefixes with `<ROBOCASA>/`.
This token preserves asset-relative paths without exposing or depending on the
original collection machine. The frozen condition bundles store the exact
evaluation XML and settled state needed for deterministic replay.

## Canonical action boundary

Every released RoboCasa-X HDF5 action uses:

`[arm(6), gripper(1), base(3), torso(1), mode(1)]`

Policies and RLDS datasets expose the first seven values. Consequently,
`barx.action_space.canonicalize_action` and evaluation have no embodiment or
legacy-layout branches.

## Paper dataset aliases

The training registry accepts paper-facing aliases such as `xp_900_pnp`,
`xp_3k_pnp`, `panda_xp_900_pnp`, and `jaco_sp_900_pnp`. They resolve to the
original RLDS directory identifiers; users do not need to rename previously
converted datasets.

The HDF5 files can be consumed directly with `h5py` or another HDF5 reader.
To build BARX's training-compatible RLDS directories, use the manifest-driven
launcher rather than invoking the TFDS builder directly:

```bash
uv run --locked --no-dev python scripts/build_rlds.py \
  --dataset xp_3k --task flip_mug \
  --raw-root /data/barx --rlds-root /data/barx-rlds
```

Valid paper datasets are `xp_900`, `xp_3k`, `sp_900`, and `target_50`.
`sp_900` and `target_50` additionally require `--target panda`, `panda_og`, or
`jaco`. The launcher validates every selected file against the byte size in
`dataset/manifest.csv` before conversion.

The `human/` files retain the simulator states and metadata needed to prepare
new MimicGen source datasets. Reproducing BARX synthesis also requires the
custom MimicGen code and task configs, which are not yet part of this public
repository.

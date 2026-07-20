# Data preparation

The release selection is recorded in `dataset/manifest.csv`; see
`dataset/README.md` for its composition. The public archive is normalized
before upload and preserves the relative `mg/` and `human/` paths.

## Canonical action boundary

Every released RoboCasa-X HDF5 action uses:

`[arm(6), gripper(1), base(3), torso(1), mode(1)]`

Policies and RLDS datasets expose the first seven values. Consequently,
`barx.action_space.canonicalize_action` and evaluation have no embodiment or
legacy-layout branches. The one-time staging command losslessly permutes the
old non-Panda arrays before release; it does not rescale or recompute any
values, and source files are never modified.

Create normalized release copies with:

```bash
uv run --locked python scripts/stage_release_data.py \
  /path/to/original-data /path/to/barx-release-data
```

The command writes each file atomically, stamps its action-layout version,
makes Panda gripper selection explicit in playback metadata, and verifies the
demonstration count. Existing normalized destination files are verified and
skipped, so staging is resumable. It stages four independent files at a time
by default; use `--workers 1` on a bandwidth-constrained host. Use `--limit 1`
for an initial smoke test.
After staging the complete tree, regenerate the manifest with final sizes and
SHA-256 checksums as described in `dataset/README.md`.

## Paper dataset aliases

The training registry accepts paper-facing aliases such as `xp_900_pnp`,
`xp_3k_pnp`, `panda_xp_900_pnp`, and `jaco_sp_900_pnp`. They resolve to the
original RLDS directory identifiers; users do not need to rename previously
converted datasets.

Build those training-compatible directories through the manifest-driven
launcher rather than invoking the TFDS builder directly:

```bash
uv run --locked python scripts/build_rlds.py \
  --dataset xp_3k --task flip_mug \
  --raw-root /data/barx --rlds-root /data/barx-rlds
```

Valid paper datasets are `xp_900`, `xp_3k`, `sp_900`, and `target_50`.
`sp_900` and `target_50` additionally require `--target panda`, `panda_og`, or
`jaco`. The launcher validates every selected file against the byte size in
`dataset/manifest.csv` before conversion. Use the regenerated manifest that
accompanies the normalized public archive, not a pre-staging selection
manifest.

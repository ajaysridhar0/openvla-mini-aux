# Data preparation

The authoritative selection is `dataset/manifest.csv`; see
`dataset/README.md` for its composition. Preserve the relative `mg/` and
`human/` paths when extracting the separately hosted archive.

## Canonical action boundary

Policies and RLDS datasets always expose seven action values. Raw RoboCasa-X
mobile actions contain 12 values:

- Panda and Panda-OG: use raw values `0:7`.
- IIWA, Kinova 3, UR5e, and Jaco: use raw values `0:6` plus raw value `-2`.

`barx.action_space.canonicalize_action` implements these original converter
branches. `barx.action_space.robocasa_action` performs the inverse layout
adaptation used during evaluation. This unifies the public API without
changing simulator controller vectors or stored training values.

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
`dataset/manifest.csv` before conversion.

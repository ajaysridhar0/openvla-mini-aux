# BARX simulation dataset

The normalized HDF5 dataset is 283.13 GiB and uses a separate data archive.
`manifest.csv` lists every file, its size, SHA-256 checksum, embodiment, task,
demonstration count, seed, relative path, and paper dataset membership.

## Contents

- **XP-900**: 300 demonstrations per task for each of IIWA, Kinova3, and UR5e
  (900 demonstrations per task total).
- **XP-3K**: 1,000 demonstrations per task for each source embodiment (3,000
  demonstrations per task total). XP-900 is the first three 100-demo shards
  for each source embodiment and is a subset of XP-3K.
- **SP-900**: 900 synthesized demonstrations per target embodiment/task for
  Panda, Panda-OG, and Jaco.
- **target-50**: 50 human demonstrations per target embodiment/task.

All four RoboCasa-X tasks are present: pick-and-place counter-to-sink,
pick-and-place sink-to-counter, turn-on-sink-faucet, and flip-mug-upright.
The manifest selects the final rendered and annotated simulation files used by
the paper's XP-900, XP-3K, SP-900, and target-50 experiments.

## Data format

All actions use
`[arm(6), gripper(1), base(3), torso(1), mode(1)]`, independent of
embodiment. Stored episode metadata uses `<ROBOCASA>/` package-relative asset
paths instead of collection-machine paths. Exact evaluation replay uses the
XML and settled states in `evaluation/conditions/`.

## RLDS conversion

`rlds/robocasa_x_dataset_builder.py` reads the normalized layout directly for
every embodiment. The release launcher retains historical TFDS directory
names required by training statistics and future checkpoints:

```bash
uv run --locked python scripts/build_rlds.py \
  --dataset xp_900 --task pnp \
  --raw-root /data/barx --rlds-root /data/barx-rlds

uv run --locked python scripts/build_rlds.py \
  --dataset target_50 --target panda --task pnp \
  --raw-root /data/barx --rlds-root /data/barx-rlds
```

Run the launcher once for each dataset/task/target combination needed by an
experiment. `pnp` includes both pick-and-place tasks, matching the jointly
trained models in the paper. Use `--dry-run` to validate manifest selection
without importing TensorFlow or writing output.

The public action schema is
`[dx, dy, dz, droll, dpitch, dyaw, gripper]`. Legacy RLDS fields such as
`ee_pose_2D`, `obj_bboxes`, and `language_motions` remain unchanged to preserve
training compatibility.

# BARX simulation dataset

The HDF5 files are distributed separately because the selected release is
283.13 GiB. `manifest.csv` lists every selected file, its size, embodiment,
task, demonstration count, seed, relative path, and paper dataset membership.
The checksum column is intentionally empty until files are staged on the final
data host.

## Contents

- **XP-900**: 300 demonstrations per task for each of IIWA, Kinova 3, and UR5e
  (900 demonstrations per task total).
- **XP-3K**: 1,000 demonstrations per task for each source embodiment (3,000
  demonstrations per task total). XP-900 is the first three 100-demo shards
  for each source embodiment and is a subset of XP-3K.
- **SP-900**: 900 synthesized demonstrations per target embodiment/task for
  Panda, Panda-OG, and Jaco.
- **target-50**: 50 human demonstrations per target embodiment/task.

All four RoboCasa-X tasks are present: pick-and-place counter-to-sink,
pick-and-place sink-to-counter, turn-on-sink-faucet, and flip-mug-upright.
Only final annotated files are selected. Raw `demo.hdf5`, `demo_failed.hdf5`,
unannotated renders, held-out scratch files, checkpoints, and real-robot data
are excluded.

## Verify or regenerate the manifest

```bash
uv run --locked python scripts/build_data_manifest.py /path/to/robocasa_x/data \
  --output dataset/manifest.csv
```

Add `--sha256` when staging the public archive. Hashing the full release reads
all 283.13 GiB and is therefore not part of the fast test suite.

## RLDS conversion

The original experiments used separate `robocasa` and `robocasa_panda`
converters. `rlds/robocasa_x_dataset_builder.py` replaces both. The release
launcher reads the manifest, infers each embodiment, and produces the same
canonical action values while retaining historical TFDS directory names
required by training statistics and future checkpoints:

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

## Publication blockers

Before uploading the data, choose the data host and dataset license, run the
manifest command with `--sha256`, and replace this section with the download
URL and archive instructions. Do not put the HDF5 files in Git or Git LFS.

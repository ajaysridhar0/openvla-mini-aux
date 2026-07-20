# BARX simulation dataset

The source HDF5 selection is 283.13 GiB, so the normalized files are
distributed separately. `manifest.csv` lists every selected file, its size,
embodiment, task, demonstration count, seed, relative path, and paper dataset
membership. Until final staging is complete, its byte sizes describe the
archival inputs and its checksum column is intentionally empty.

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

## Stage and finalize the release

Do not edit the archival source tree in place. Create normalized copies:

```bash
uv run --locked python scripts/stage_release_data.py \
  /path/to/original-data /path/to/barx-release-data
```

All output actions then use
`[arm(6), gripper(1), base(3), torso(1), mode(1)]`, independent of
embodiment. Once all files have been staged, regenerate the checked release
manifest from the output tree:

```bash
uv run --locked python scripts/build_data_manifest.py /path/to/barx-release-data \
  --output dataset/manifest.csv --sha256
```

Hashing the full release reads roughly 283 GiB and is therefore not part of
the fast test suite.

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

## Publication blockers

Before uploading the data, choose the data host and dataset license, run the
manifest command with `--sha256`, and replace this section with the download
URL and archive instructions. Do not put the HDF5 files in Git or Git LFS.

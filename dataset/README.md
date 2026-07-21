# BARX simulation dataset

The normalized HDF5 release is 283.13 GiB, so its files are distributed
separately. `manifest.csv` lists every selected file, its final staged size,
embodiment, task, demonstration count, seed, relative path, and paper dataset
membership. Its checksum column remains empty until the final upload hashing
pass.

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

## Stage and finalize the release

Do not edit the archival source tree in place. Create normalized copies:

```bash
uv run --locked python scripts/stage_release_data.py \
  /path/to/original-data /path/to/barx-release-data
```

For a large copy, independent workers can process non-overlapping manifest
shards by passing the same `--shard-count` and a distinct `--shard-index` to
each invocation. A final unsharded invocation verifies the complete tree.

All output actions then use
`[arm(6), gripper(1), base(3), torso(1), mode(1)]`, independent of
embodiment. Stored episode metadata uses `<ROBOCASA>/` package-relative asset
paths instead of collection-machine paths. Exact evaluation replay uses the
XML and settled states in `evaluation/conditions/`. Once all files have been
staged, regenerate the checked release manifest from the output tree:

```bash
uv run --locked python scripts/build_data_manifest.py /path/to/barx-release-data \
  --output dataset/manifest.csv --sha256 --hash-workers 4
```

The final release-validation pass hashes all 283 GiB.

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

## Publishing the data

Choose the data host and dataset license, run the manifest command with
`--sha256`, and add the download URL and archive instructions here. A
dedicated dataset host is appropriate for the HDF5 archive.

# BARX raw HDF5 dataset

The normalized HDF5 dataset is 284.03 GiB and uses a separate data archive.
`manifest.csv` lists all 240 files with their size, SHA-256 checksum,
embodiment, task, demonstration count, seed, relative path, and paper dataset
membership. The archive contains 23,400 demonstrations in total.

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

## Download

The public archive is pinned in `configs/raw_dataset.json`. Download that exact
revision into a new directory:

```bash
hf download ajaysri/barx-raw-hdf5 --repo-type dataset \
  --revision c8b3aba2dfbbfbafa67fa82466320945d560208f \
  --local-dir /data/barx
```

The repository is public and the download requires no Hugging Face token.

### Download one paper subset

The 24 manifests under `dataset/subsets/` mirror the 24 repositories in the
public RLDS collection. Download and fully verify only the raw files underlying
one RLDS dataset with the paper-facing dataset, task, and target names:

```bash
uv run --locked --no-dev python scripts/download_raw_data.py \
  --dataset xp_900 --task pnp --output-dir /data/barx-xp900-pnp

uv run --locked --no-dev python scripts/download_raw_data.py \
  --dataset target_50 --target panda --task pnp \
  --output-dir /data/barx-target-panda-pnp
```

The downloader is anonymous, uses the pinned raw-data revision, and writes
`subset-manifest.csv` beside the selected `human/` or `mg/` tree. By default it
validates HDF5 structure, byte sizes, demonstration counts, and SHA-256 hashes.
Add `--dry-run` to print the exact paths and size before downloading, or
`--skip-checksums` to skip only the final full-byte hash pass.

Regenerate the checked-in views after changing the master manifest:

```bash
uv run --locked --no-dev python scripts/build_raw_subsets.py
uv run --locked --no-dev python scripts/build_raw_subsets.py --check
```

## Verify the archive

Run the full size, structure, metadata, and SHA-256 verification before using
or repackaging a downloaded archive:

```bash
uv run --locked --no-dev python scripts/verify_raw_data.py /data/barx
```

For a faster preflight that does not read every byte, add `--skip-checksums`.
This still validates every path, byte size, demonstration count, action shape,
environment, and portable metadata.

## Data format

Each file is a standard HDF5 container with demonstrations under
`data/demo_*`. The main fields are:

- `actions`: `[T, 12]` canonical simulator actions;
- `states`: simulator states for replay and data generation;
- `obs/agentview_rgb`: `[T, 180, 320, 3]` RGB observations;
- `obs/ee_states` and `obs/gripper_states`: policy state inputs;
- `aux_info/eef_normalized_image_pts` and `aux_info/bboxes_2d/*`: BARX
  representation annotations; and
- `ep_meta`: JSON episode metadata including the language instruction.

Actions use `[arm(6), gripper(1), base(3), torso(1), mode(1)]`, independent of
embodiment. Stored metadata uses `<ROBOCASA>/` package-relative asset paths.
Users can read these files directly with `h5py` to build another training
format; conversion to RLDS is optional.

## RLDS conversion

`rlds/robocasa_x_dataset_builder.py` reads the normalized layout directly for
every embodiment. The release launcher retains historical TFDS directory
names required by training statistics and future checkpoints:

```bash
uv run --locked --no-dev python scripts/build_rlds.py \
  --dataset xp_900 --task pnp \
  --raw-root /data/barx --rlds-root /data/barx-rlds

uv run --locked --no-dev python scripts/build_rlds.py \
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

## Extending the demonstrations with MimicGen

The `human/` files retain actions, simulator states, environment metadata, and
50 demonstrations per target/task, so they preserve the information needed to
prepare MimicGen source datasets. BARX ships the compatibility snapshot under
`third_party/mimicgen/`, pins its source revision in `configs/mimicgen.json`,
and exposes path-independent preparation and bounded generation wrappers.

Install the separately licensed generator and follow the complete regeneration
gate:

```bash
uv sync --locked --extra mg --no-dev
```

See [Section 8 of the release walkthrough](../RELEASE_TEST_README.md#8-mimicgen-regeneration-gate)
for source immutability checks, preparation, bounded generation, portable HDF5
validation, and video review. MimicGen source code retains NVIDIA's
non-commercial research/evaluation license. Released raw and processed BARX
demonstration data are CC BY 4.0; see
[`docs/artifact_licenses.md`](../docs/artifact_licenses.md).

## Rebuilding a portable archive

Maintainers starting from an older collection tree can scrub private paths and
normalize action metadata into a separate output tree, then rebuild the
manifest:

```bash
uv run --locked --no-dev python scripts/stage_release_data.py \
  /data/barx-archive /data/barx-public
uv run --locked --no-dev python scripts/build_data_manifest.py \
  /data/barx-public --output dataset/manifest.csv
```

Never stage in place: the source and output roots must differ.

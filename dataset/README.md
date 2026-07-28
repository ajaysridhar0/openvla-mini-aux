# BARX raw HDF5 dataset

Most users should use the released processed RLDS datasets as shown in the
root quick start. Download raw HDF5 data only if you want to inspect the source
demonstrations, build another data format, rerun RLDS conversion, or generate
new demonstrations with MimicGen.

The optional raw HDF5 release is 284 GiB and contains 23,400 demonstrations.

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
These are the rendered and annotated simulation demonstrations used by the
paper's XP-900, XP-3K, SP-900, and target-50 experiments.

## Download raw data

Download the public archive into a new directory:

```bash
hf download ajaysri/barx-raw-hdf5 --repo-type dataset \
  --local-dir /data/barx
```

The repository is public and the download requires no Hugging Face token.

### Download one paper subset

To avoid downloading the full archive, select one paper dataset using its
dataset, task, and target names:

```bash
uv run --locked --no-dev python scripts/download_raw_data.py \
  --dataset xp_900 --task pnp --output-dir /data/barx-xp900-pnp

uv run --locked --no-dev python scripts/download_raw_data.py \
  --dataset target_50 --target panda --task pnp \
  --output-dir /data/barx-target-panda-pnp
```

Add `--dry-run` to see the selected files and download size first.

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
embodiment. Users can read these files directly with `h5py` to build another
training format; conversion to RLDS is optional.

## RLDS conversion

`rlds/robocasa_x_dataset_builder.py` reads the normalized layout directly for
every embodiment. The launcher writes the TFDS directory names expected by the
released training configurations and checkpoints:

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
trained models in the paper. Use `--dry-run` to preview the conversion.

The public action schema is
`[dx, dy, dz, droll, dpitch, dyaw, gripper]`. Legacy RLDS fields such as
`ee_pose_2D`, `obj_bboxes`, and `language_motions` remain unchanged to preserve
training compatibility.

## Extending the demonstrations with MimicGen

The `human/` files retain actions, simulator states, environment metadata, and
50 demonstrations per target/task. BARX includes the MimicGen components used
by the paper under `third_party/mimicgen/`.

Install the optional dependency and download one human source subset:

```bash
uv sync --locked --extra mg --no-dev
uv run --locked --no-dev python scripts/download_raw_data.py \
  --dataset target_50 --target panda --task flip_mug \
  --output-dir /data/barx-target-panda-flip-mug
```

Prepare five demonstrations in a new HDF5 file:

```bash
mkdir -p /data/barx-mimicgen

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
uv run --locked --extra mg --no-dev python scripts/prepare_mimicgen_source.py \
  --source /data/barx-target-panda-flip-mug/human/PandaOmron/FlipMugUpright/demo_gentex_im320.hdf5 \
  --output /data/barx-mimicgen/panda-flip-mug-prepared.hdf5 \
  --task flip_mug_upright --demos 5 \
  --summary /data/barx-mimicgen/preparation-summary.json
```

Run a bounded one-success generation example:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
uv run --locked --extra mg --no-dev python scripts/generate_mimicgen.py \
  --source /data/barx-mimicgen/panda-flip-mug-prepared.hdf5 \
  --task flip_mug_upright --embodiment panda --seed 0 \
  --successes 1 --max-attempts 25 --source-demos 5 \
  --output-dir /data/barx-mimicgen/panda-flip-mug-seed0 \
  --video /data/barx-mimicgen/panda-flip-mug-seed0.mp4
```

The output directory contains the generated HDF5 data, resolved configuration,
and summary files. Review the MP4 to verify the generated behavior. Use
`--help` on either wrapper for the supported BARX tasks and embodiments.

MimicGen source code retains NVIDIA's non-commercial research/evaluation
license. Released raw and processed BARX demonstration data are CC BY 4.0; see
[`docs/artifact_licenses.md`](../docs/artifact_licenses.md).

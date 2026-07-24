# BARX independent release command log

This log records the clean-room acceptance run performed on July 23–24, 2026.
It complements [`RELEASE_TEST_README.md`](RELEASE_TEST_README.md): the
walkthrough defines the procedure, while this file records the commands that
were executed and the important output observed.

The tester started without BARX project context, used fresh Git checkouts,
empty BARX and Hugging Face caches, anonymous Hugging Face access, and an
NVIDIA A40. Machine-specific temporary paths have been replaced with the
documented `BARX_*` variables. Long progress bars, repeated simulator warnings,
and private host details are omitted. Numeric results and error messages are
preserved.

## Outcome

| Public commit | Sections attempted | Result |
| --- | --- | --- |
| `6bba3436488c57a83cb04954ff6116d51a37c2be` | 0–4, 10 | Sections 0–3 and 10 passed; the original Section 4 audit counted Hugging Face cache sidecars. |
| `e1593084ad8c02b964acf56acce042c5852aa465` | 4–5, 10 | Section 4 passed; the original dynamic TFDS builder failed from `__main__`. |
| `82f2c274fcb6f95082f4b0e128e2545a825bde5e` | 5–8, 10 | Sections 5–7 and 10 passed; Section 8 reached the documented missing MimicGen interface. |

The final reachable result was:

| Section | Result |
| --- | --- |
| 0. System and clone | PASS |
| 1. Install, assets, tests | PASS |
| 2. Public artifacts and checkpoints | PASS |
| 3. Raw HDF5 | PASS |
| 4. Published RLDS | PASS |
| 5. HDF5-to-RLDS conversion | PASS |
| 6. One-step training | PASS |
| 7. One-trial evaluation | PASS |
| 8. MimicGen generation | BLOCKED — documented public interface absent |
| 10. Git cleanliness | PASS |

## Test setup

The runner created the external evidence and artifact directories exactly as
documented:

```bash
set -euo pipefail

export BARX_ACCEPTANCE_ROOT="${BARX_ACCEPTANCE_ROOT:-$PWD/barx-acceptance}"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_ARTIFACT_ROOT="$BARX_ACCEPTANCE_ROOT/artifacts"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export BARX_RAW_XP900="$BARX_ACCEPTANCE_ROOT/raw-xp900-pnp"
export BARX_RAW_HUMAN="$BARX_ACCEPTANCE_ROOT/raw-target-panda-flip-mug"
export BARX_CONVERTED_RLDS="$BARX_ACCEPTANCE_ROOT/converted-rlds"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export BARX_VQ_ROOT="$BARX_ARTIFACT_ROOT/vq"

unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
mkdir -p "$BARX_EVIDENCE"/{00-system,01-install,02-artifacts,03-hdf5,04-rlds,05-conversion,06-training,07-eval,08-mimicgen,09-final}
```

The accepted preflight reported Git 2.43.0, CMake 3.28.3, an NVIDIA A40 with
46,068 MiB, and more than 1 TiB free. Two discarded attempts exposed host
contamination rather than repository defects:

- a stale user-local `cmake` wrapper was ahead of `/usr/bin/cmake`; and
- a preinstalled `uv 0.6.2` could not parse the project metadata.

The final clean run used `/usr/bin/cmake` and the documented `uv 0.11.11`.

## 0. System and clean clone

Commands:

```bash
(
  date --iso-8601=seconds
  uname -a
  command -v git
  command -v cmake
  command -v nvidia-smi
  git --version
  cmake --version
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  df -h "$BARX_ACCEPTANCE_ROOT"
) 2>&1 | tee "$BARX_EVIDENCE/00-system/preflight.txt"

git clone --branch barx-release-integration \
  https://github.com/ajaysridhar0/openvla-mini-aux.git "$BARX_REPO"
cd "$BARX_REPO"

{
  git rev-parse HEAD
  git remote -v
  git status --porcelain
} | tee "$BARX_EVIDENCE/00-system/git.txt"
```

Key output:

```text
git version 2.43.0
cmake version 3.28.3
NVIDIA A40, 610.43.02, 46068 MiB
6bba3436488c57a83cb04954ff6116d51a37c2be
origin  https://github.com/ajaysridhar0/openvla-mini-aux.git
```

`git status --porcelain` was empty.

## 1. Installation, tests, assets, and simulator import

Commands:

```bash
uv --version | tee "$BARX_EVIDENCE/01-install/uv-version.txt"
uv sync --locked --no-dev 2>&1 | tee "$BARX_EVIDENCE/01-install/uv-sync.txt"
uv run --locked --no-dev python -m unittest discover -s tests -v \
  2>&1 | tee "$BARX_EVIDENCE/01-install/unit-tests.txt"

uv run --locked --no-dev python \
  robocasa_x/robocasa/scripts/download_kitchen_assets.py --yes \
  2>&1 | tee "$BARX_EVIDENCE/01-install/assets.txt"

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
uv run --locked --no-dev python - 2>&1 <<'PY' \
  | tee "$BARX_EVIDENCE/01-install/simulator-import.txt"
import mujoco
import robocasa
import robosuite

print("RoboCasa-X import OK")
print("MuJoCo", mujoco.__version__)
print("RoboCasa package", robocasa.__file__)
print("robosuite", robosuite.__version__)
PY
```

Key output:

```text
uv 0.11.11 (x86_64-unknown-linux-gnu)
Ran 64 tests in 1.703s
OK
Checksum verified; extracting...  # repeated for all four asset archives
RoboCasa-X import OK
MuJoCo 3.1.1
robosuite 1.5.1
```

The locked sync did not change `uv.lock`.

## 2. Public artifacts and checkpoints

Command:

```bash
uv run --locked --no-dev python scripts/download_public_artifacts.py \
  --artifact-root "$BARX_ARTIFACT_ROOT" \
  --include-pretrain-checkpoint \
  2>&1 | tee "$BARX_EVIDENCE/02-artifacts/download.txt"
```

The tester then ran the full checkpoint-collection audit heredoc in Section 2
of the walkthrough and created the local inventory:

```bash
{
  find "$BARX_ARTIFACT_ROOT/base-vlm" -type f -printf '%P\t%s bytes\n' | sort
  find "$BARX_ARTIFACT_ROOT/vq" -type f -printf '%P\t%s bytes\n' | sort
  find "$BARX_ARTIFACT_ROOT/runs/xp900-pnp-joint-reps" \
    -type f -printf '%P\t%s bytes\n' | sort
} | tee "$BARX_EVIDENCE/02-artifacts/local-inventory.txt"
```

Key output:

```text
Base MiniVLA checkpoint:                         2,630,986,501 bytes
XP-900 PnP VQ tokenizer:                            9,519,250 bytes
Joint Reps step-050000-epoch-15-loss=0.2577.pt: 5,554,882,540 bytes
Public checkpoint collection models: 12
Private models: 0
Gated models: 0
```

The complete 260-file XP-900 PnP RLDS snapshot and the pinned DINOv2, SigLIP,
and Qwen runtime snapshots downloaded anonymously. Regular HTTP was used when
the optional Xet transport was unavailable.

## 3. Raw HDF5

Dry-run commands:

```bash
uv run --locked --no-dev python scripts/download_raw_data.py \
  --dataset xp_900 --task pnp \
  --output-dir "$BARX_RAW_XP900" --dry-run \
  > "$BARX_EVIDENCE/03-hdf5/xp900-dry-run.txt"

uv run --locked --no-dev python scripts/download_raw_data.py \
  --dataset target_50 --target panda --task flip_mug \
  --output-dir "$BARX_RAW_HUMAN" --dry-run \
  > "$BARX_EVIDENCE/03-hdf5/human-dry-run.txt"
```

Download and verification commands:

```bash
uv run --locked --no-dev python scripts/download_raw_data.py \
  --dataset xp_900 --task pnp \
  --output-dir "$BARX_RAW_XP900" \
  2>&1 | tee "$BARX_EVIDENCE/03-hdf5/xp900-download-and-verify.txt"

uv run --locked --no-dev python scripts/download_raw_data.py \
  --dataset target_50 --target panda --task flip_mug \
  --output-dir "$BARX_RAW_HUMAN" \
  2>&1 | tee "$BARX_EVIDENCE/03-hdf5/human-download-and-verify.txt"
```

The runner also executed the HDF5 inspection heredoc from Section 3.

Key output:

```text
robocasa-x-xp900-pnp: 18 files, 1800 demonstrations, 24.19 GiB
Verified 18 files, 1800 demonstrations, 24.19 GiB with SHA-256

robocasa-x-target-panda-flip-mug-upright:
  1 files, 50 demonstrations, 0.35 GiB
Verified 1 files, 50 demonstrations, 0.35 GiB with SHA-256
```

Two CDN read timeouts occurred during the XP-900 transfer. Hugging Face resumed
both downloads, and the final SHA-256 verification passed.

Representative decoded structure:

```json
{
  "xp900_pnp": {
    "files": 18,
    "demonstrations": 1800,
    "action_shape": [350, 12],
    "agentview_shape": [350, 180, 320, 3],
    "environment": "XPnPCounterToSink",
    "instruction": "pick the can from the counter and place it in the sink"
  },
  "target_panda_flip_mug": {
    "files": 1,
    "demonstrations": 50,
    "action_shape": [244, 12],
    "agentview_shape": [244, 180, 320, 3],
    "environment": "XFlipMugUpright",
    "instruction": "flip the mug on the counter upright"
  }
}
```

## 4. Published RLDS

The tester ran both Section 4 audit heredocs.

### Original failure at `6bba343`

Command:

```bash
uv run --locked --no-dev python - \
  > "$BARX_EVIDENCE/04-rlds/published-rlds.json" <<'PY'
# Original Section 4 audit
PY
```

Output:

```text
assert len(tfrecords) == 256, len(tfrecords)
AssertionError: 768
```

Read-only diagnosis:

```text
256 real TFRecord shards
256 Hugging Face .lock sidecars
256 Hugging Face .metadata sidecars
768 broad glob matches
```

The ordered run stopped, the audit was corrected to ignore `.cache`, and the
tester resumed from a clean checkout at `e159308`.

### Passing rerun at `e159308`

Key output:

```json
{
  "dataset_info": "1.0.0/dataset_info.json",
  "features": "1.0.0/features.json",
  "tfrecord_shards": 256,
  "total_bytes": 24899282513
}
```

The public collection audit reported exactly 24 public, ungated RLDS dataset
IDs, and those IDs matched the 24 checked-in raw subset manifests one-to-one.

## 5. HDF5-to-RLDS conversion

Command:

```bash
uv run --locked --no-dev python scripts/build_rlds.py \
  --dataset target_50 --target panda --task flip_mug \
  --raw-root "$BARX_RAW_HUMAN" \
  --rlds-root "$BARX_CONVERTED_RLDS" \
  2>&1 | tee "$BARX_EVIDENCE/05-conversion/conversion.txt"
```

### Original failure at `e159308`

```text
target_50/panda/flip_mug: 1 files -> .../panda_flip_mug
TypeError: Cannot access resources for '__main__' as it does not appear to
correspond to an importable module (__spec__ is None).
```

The ordered run stopped. The dynamic TFDS builder was changed to use its
importable base module, and the decoded-step audit was corrected for TFDS'
iterable episode representation.

### Passing rerun at `82f2c27`

```text
target_50/panda/flip_mug: 1 files -> .../panda_flip_mug
Generating with 32 workers!
Dataset panda_flip_mug downloaded and prepared
Prepared panda_flip_mug/1.0.0
```

The literal inspection heredoc returned:

```json
{
  "builder_name": "panda_flip_mug",
  "version": "1.0.0",
  "examples": 50,
  "observation_keys": ["image", "state"],
  "action_shape": [7],
  "instruction": "flip the mug on the counter upright"
}
```

The decoded step also contained `ee_pose_2D`, language-motion annotations, and
object bounding boxes.

## 6. Training

Paper-shaped dry-run commands:

```bash
uv run --locked --extra train --no-dev python scripts/train.py prior \
  --prior xp_900 --task pnp --method joint_reps \
  --data-root "$BARX_ARTIFACT_ROOT/data" \
  --base-vlm "$BARX_ARTIFACT_ROOT/base-vlm" \
  --run-root "$BARX_ARTIFACT_ROOT/runs/paper-dry-run" \
  --gpus 8 --global-batch-size 256 --per-device-batch-size 32 \
  --max-steps 50000 --dry-run

uv run --locked --extra train --no-dev python scripts/train.py adapt \
  --prior xp_900 --target panda --task pnp --method joint_reps \
  --data-root "$BARX_ARTIFACT_ROOT/data" \
  --base-vlm "$BARX_ARTIFACT_ROOT/base-vlm" \
  --checkpoint "$BARX_ARTIFACT_ROOT/runs/xp900-pnp-joint-reps/checkpoints/step-050000-epoch-15-loss=0.2577.pt" \
  --run-root "$BARX_ARTIFACT_ROOT/runs/paper-dry-run" \
  --gpus 8 --global-batch-size 256 --per-device-batch-size 32 \
  --max-steps 3000 --dry-run
```

Both printed eight-process `torchrun` commands with global batch 256, batch 32
per GPU, learning rate `2e-5`, the expected dataset/statistics mapping, and the
selected source-prior checkpoint.

Real one-step command:

```bash
uv sync --locked --extra train --no-dev \
  2>&1 | tee "$BARX_EVIDENCE/06-training/train-sync.txt"

uv run --locked --extra train --no-dev python scripts/train.py prior \
  --prior xp_900 --task pnp --method joint_reps \
  --data-root "$BARX_ARTIFACT_ROOT/data" \
  --base-vlm "$BARX_ARTIFACT_ROOT/base-vlm" \
  --run-root "$BARX_ARTIFACT_ROOT/runs/smoke" \
  --gpus 1 --global-batch-size 1 --per-device-batch-size 1 \
  --max-steps 1 --save-interval 100 --skip-final-checkpoint \
  2>&1 | tee "$BARX_EVIDENCE/06-training/one-step-training.txt"
```

Key output:

```text
FSDP Full-Shard Strategy
World size = 1
Max Steps = 1
Starting VLA Training Loop
Done with Training
```

Durable metric record:

```json
{
  "VLA Train/Step": 1,
  "VLA Train/Learning Rate": 0.00002,
  "VLA Train/Loss": 4.7211151123046875,
  "VLA Train/L1 Loss": 0.10572677105665207,
  "VLA Train/Step Time": 9.860999822616577,
  "checkpoints_written": []
}
```

`config.json`, `config.yaml`, `dataset_statistics.json`, and
`run-metrics.jsonl` were present. W&B was not contacted.

## 7. Frozen conditions and one-trial evaluation

Condition verification command:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
uv run --locked --no-dev python scripts/verify_eval_conditions.py \
  --task pnp_counter_to_sink --embodiment panda \
  2>&1 | tee "$BARX_EVIDENCE/07-eval/conditions.txt"
```

Key output:

```text
Verified pnp_counter_to_sink/panda 1/100:
  condition=12b53db1be6c88c7, seed=1000, layout=7, style=8
...
Verified pnp_counter_to_sink/panda 100/100:
  condition=2ed3d832e8d98030, seed=1099, layout=4, style=6
```

The first cold EGL/MuJoCo initialization took approximately 30 minutes. The
process remained compute-active and all 100 frozen conditions ultimately
passed.

One-trial command:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
uv run --locked --extra train --no-dev python scripts/evaluate.py \
  --checkpoint "$BARX_ARTIFACT_ROOT/runs/xp900-pnp-joint-reps/checkpoints/step-050000-epoch-15-loss=0.2577.pt" \
  --embodiment panda \
  --task pnp_counter_to_sink \
  --unnorm-key mg_pnp_lite \
  --episodes 1 \
  --rollout-dir "$BARX_ARTIFACT_ROOT/rollouts/one-trial" \
  2>&1 | tee "$BARX_EVIDENCE/07-eval/one-trial.txt"

uv run --locked --no-dev python scripts/summarize_evaluations.py \
  "$BARX_ARTIFACT_ROOT/rollouts" \
  --output "$BARX_EVIDENCE/07-eval/results.csv"
```

The exact wrapper flags are recorded in the walkthrough; the resolved evaluator
used PandaOmron, Robotiq85Gripper, `barx_panda_agentview`, action horizon 8,
maximum 600 environment steps, and frozen condition seed 1000.

Key output:

```text
Starting episode 1...
Environment steps: 610/610
Saved rollout MP4
status=complete
episodes=1
successes=0
success_rate=0.0
```

Structured result:

```json
{
  "status": "complete",
  "task": "pnp_counter_to_sink",
  "embodiment": "panda",
  "seed": 1000,
  "layout_id": 7,
  "style_id": 8,
  "steps": 600,
  "instruction": "pick the cucumber from the counter and place it in the sink",
  "success": false
}
```

The MP4 was H.264, 320×192, 600 frames, and 20 seconds. Human contact-sheet
review found a coherent Panda kitchen scene with the expected sink, robot, and
cucumber. A task failure was allowed by the smoke-test contract and was not
treated as a paper success-rate estimate.

## 8. MimicGen gate

Command:

```bash
(
  missing=0
  for path in \
    mimicgen \
    scripts/prepare_mimicgen_source.py \
    scripts/generate_mimicgen.py
  do
    if test -e "$path"; then
      echo "PRESENT $path"
    else
      echo "MISSING $path"
      missing=1
    fi
  done
  exit "$missing"
) 2>&1 | tee "$BARX_EVIDENCE/08-mimicgen/interface.txt"
```

Output and exit status:

```text
MISSING mimicgen
MISSING scripts/prepare_mimicgen_source.py
MISSING scripts/generate_mimicgen.py
exit 1
```

This was the expected documented public-release boundary. The tester did not
install an arbitrary upstream MimicGen version or use a private workaround.

## 10. Final cleanliness

Command:

```bash
{
  git status --porcelain
  git diff --exit-code
  git diff --cached --exit-code
  git rev-parse HEAD
} | tee "$BARX_EVIDENCE/09-final/git-clean.txt"
```

Output:

```text
82f2c274fcb6f95082f4b0e128e2545a825bde5e
```

Both status and diff outputs were empty. All downloads, converted data,
training outputs, logs, and videos remained outside the checkout.

## Interpretation

The command sequence demonstrates anonymous installation, artifact access, raw
and RLDS data access, real HDF5-to-RLDS conversion, one real optimizer step,
checkpoint loading, frozen-condition verification, and one complete simulator
rollout. It does not reproduce a paper success rate, and full end-to-end
acceptance remains blocked until the BARX-compatible MimicGen interface is
published and Section 8 passes.

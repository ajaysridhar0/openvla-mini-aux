# BARX paper-release acceptance walkthrough

This is the end-to-end acceptance procedure for a fresh user trying to use the
public BARX repository alongside the ICRA 2026 paper
[Cross-Embodiment Transfer via Behavior-Aligned Representations](https://ajaysridhar.com/barx/img/pdfs/icra_barx.pdf).
It is intentionally more demanding than the root README quick start.

The runner should begin with no private context, no pre-existing simulator
checkout, and empty Hugging Face and BARX artifact caches. Do not substitute
private files, unpublished commands, or cluster-only paths. A missing public
dependency or undocumented choice is a release failure, not permission to
guess.

## What this walkthrough proves

The acceptance run has two levels:

1. **Public release acceptance** executes one representative path through
   installation, public artifacts, raw HDF5, published RLDS, HDF5-to-RLDS
   conversion, one-step training, checkpoint loading, and one-trial headless
   evaluation. Every step must pass.
2. **Paper-scale reproduction** expands the same interfaces to the paper's full
   datasets, training schedules, three-checkpoint selection, and 100 frozen
   evaluation conditions. It is expensive and is not implied by a one-step or
   one-trial smoke test.

MimicGen regeneration is a hard public-release gate. At the time this document
was added, the repository preserved the human source demonstrations but did not
yet contain the customized BARX MimicGen fork, task configurations, or a public
generation launcher. Section 8 must therefore be reported as **BLOCKED** until
those files are integrated and its exact command is added here. Do not mark the
release end-to-end ready while that section is blocked.

## Rules for the independent runner

- Execute the sections in order from a fresh clone.
- Keep all downloads, generated data, logs, checkpoints, and videos outside the
  Git checkout.
- Run public Hugging Face access anonymously. The release paths explicitly pass
  `token=False`; do not log in to make a failed public download work.
- Save stdout and stderr for every long-running command.
- Do not edit the repository during the test.
- Stop at the first unexplained failure, but still write the report in
  Section 11.
- A command returning zero is not enough. Apply both the machine and human
  checks listed after it.
- Never compare a one-trial result to a paper success rate.

## Evidence directory

Run the following from a shell whose current directory is not inside another
BARX checkout:

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

The representative path needs approximately 120 GiB free. A full 284.03 GiB
raw-archive audit needs additional space for the Hugging Face cache and any
converted output; plan separately rather than placing it in the checkout.

## 0. System and clean-clone preflight

Record the machine before installing anything:

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
```

Clone the release branch and record its exact public revision:

```bash
git clone --branch barx-release-integration \
  https://github.com/ajaysridhar0/openvla-mini-aux.git "$BARX_REPO"
cd "$BARX_REPO"

{
  git rev-parse HEAD
  git remote -v
  git status --porcelain
} | tee "$BARX_EVIDENCE/00-system/git.txt"
```

Machine pass criteria:

- Linux x86-64, CMake, Git, and an NVIDIA GPU are visible.
- `git status --porcelain` prints nothing.
- At least 120 GiB is free for the representative run.
- The clone required no credentials.

Human evidence:

- `preflight.txt`
- `git.txt`

## 1. Install, assets, and unit tests

Install the documented `uv` release if it is not already available:

```bash
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version | tee "$BARX_EVIDENCE/01-install/uv-version.txt"
```

Resolve only the locked public environment and run the complete unit suite:

```bash
uv sync --locked --no-dev 2>&1 | tee "$BARX_EVIDENCE/01-install/uv-sync.txt"
uv run --locked --no-dev python -m unittest discover -s tests -v \
  2>&1 | tee "$BARX_EVIDENCE/01-install/unit-tests.txt"
```

Download the public RoboCasa assets and confirm the simulator imports:

```bash
uv run --locked --no-dev python \
  robocasa_x/robocasa/scripts/download_kitchen_assets.py --yes \
  2>&1 | tee "$BARX_EVIDENCE/01-install/assets.txt"

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
uv run --locked --no-dev python - 2>&1 <<'PY' | tee "$BARX_EVIDENCE/01-install/simulator-import.txt"
import mujoco
import robocasa
import robosuite

print("RoboCasa-X import OK")
print("MuJoCo", mujoco.__version__)
print("RoboCasa package", robocasa.__file__)
print("robosuite", robosuite.__version__)
PY
```

Machine pass criteria:

- `uv sync --locked --no-dev` does not modify `uv.lock`.
- The unit test summary is `OK`; an explicitly skipped full-simulator import
  unit test is acceptable because the literal import is exercised afterward.
- All four asset archives pass their size and SHA-256 checks.
- The final command prints `RoboCasa-X import OK`.

Human evidence:

- `uv-sync.txt`
- `unit-tests.txt`
- `assets.txt`
- `simulator-import.txt`

## 2. Public artifacts and pretrained checkpoints

Download the pinned XP-900 PnP walkthrough artifacts, including the released
Joint Reps source-prior checkpoint:

```bash
uv run --locked --no-dev python scripts/download_public_artifacts.py \
  --artifact-root "$BARX_ARTIFACT_ROOT" \
  --include-pretrain-checkpoint \
  2>&1 | tee "$BARX_EVIDENCE/02-artifacts/download.txt"
```

This command verifies the byte size and SHA-256 for the base MiniVLA
checkpoint, VQ tokenizer, and selected source-prior checkpoint. It also
downloads the pinned public XP-900 PnP RLDS repository and pinned DINOv2,
SigLIP, and Qwen runtime files.

Audit the complete public checkpoint collection without downloading all twelve
multi-gigabyte checkpoints:

```bash
uv run --locked --no-dev python - \
  > "$BARX_EVIDENCE/02-artifacts/checkpoint-collection.json" <<'PY'
import json
import urllib.request

collection_slug = "ajaysri/barx-pretraining-models-joint-reps-and-no-reps"

def read_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "barx-release-audit"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)

collection = read_json(f"https://huggingface.co/api/collections/{collection_slug}")
assert collection["private"] is False
assert len(collection["items"]) == 12, len(collection["items"])

models = []
for item in collection["items"]:
    assert item["type"] == "model"
    assert item["private"] is False
    assert item["gated"] is False
    model = read_json(f"https://huggingface.co/api/models/{item['id']}")
    checkpoints = [
        sibling["rfilename"]
        for sibling in model["siblings"]
        if sibling["rfilename"].endswith(".pt")
    ]
    assert len(model["sha"]) == 40
    assert len(checkpoints) == 1, (item["id"], checkpoints)
    models.append(
        {
            "repo_id": item["id"],
            "revision": model["sha"],
            "checkpoint": checkpoints[0],
            "private": model["private"],
            "gated": model["gated"],
        }
    )

print(json.dumps({"collection": collection["slug"], "models": models}, indent=2))
PY
```

Create a compact local inventory for human review:

```bash
{
  find "$BARX_ARTIFACT_ROOT/base-vlm" -type f -printf '%P\t%s bytes\n' | sort
  find "$BARX_ARTIFACT_ROOT/vq" -type f -printf '%P\t%s bytes\n' | sort
  find "$BARX_ARTIFACT_ROOT/runs/xp900-pnp-joint-reps" \
    -type f -printf '%P\t%s bytes\n' | sort
} | tee "$BARX_EVIDENCE/02-artifacts/local-inventory.txt"
```

Machine pass criteria:

- Every download uses the immutable revision printed in `download.txt`.
- The downloader finishes without a token.
- `checkpoint-collection.json` contains 12 public, ungated models, each with a
  40-character revision and exactly one `.pt` checkpoint.
- The downloaded XP-900 PnP Joint Reps checkpoint is
  `step-050000-epoch-15-loss=0.2577.pt` and passes the manifest SHA-256.

Human evidence:

- `download.txt`
- `checkpoint-collection.json`
- `local-inventory.txt`

## 3. Raw HDF5 data, including human demonstrations

First inspect the exact scope and size without downloading:

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

Expected summaries:

- XP-900 PnP: 18 HDF5 files, 1,800 demonstrations, 24.19 GiB.
- Panda target-50 Flip Mug Upright: 1 HDF5 file, 50 human demonstrations,
  approximately 0.35 GiB.

Download and fully verify both subsets:

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

Inspect representative structure and write JSON that a human can read:

```bash
uv run --locked --no-dev python - \
  > "$BARX_EVIDENCE/03-hdf5/inspection.json" <<'PY'
import csv
import json
import os
from pathlib import Path

import h5py

roots = {
    "xp900_pnp": Path(os.environ["BARX_RAW_XP900"]),
    "target_panda_flip_mug": Path(os.environ["BARX_RAW_HUMAN"]),
}
report = {}
for label, root in roots.items():
    with (root / "subset-manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    sample_path = root / rows[0]["relative_path"]
    with h5py.File(sample_path, "r") as dataset:
        demo_name = sorted(dataset["data"].keys())[0]
        demo = dataset["data"][demo_name]
        ep_meta = json.loads(demo.attrs["ep_meta"])
        env_args = json.loads(dataset["data"].attrs["env_args"])
        report[label] = {
            "files": len(rows),
            "demonstrations": sum(int(row["demonstrations"]) for row in rows),
            "sample_relative_path": rows[0]["relative_path"],
            "sample_demo": demo_name,
            "sample_keys": sorted(demo.keys()),
            "action_shape": list(demo["actions"].shape),
            "agentview_shape": list(demo["obs"]["agentview_rgb"].shape),
            "environment": env_args["env_name"],
            "instruction": ep_meta["lang"],
        }

print(json.dumps(report, indent=2))
PY
```

Machine pass criteria:

- Both downloaders use `ajaysri/barx-raw-hdf5` at the revision in
  `configs/raw_dataset.json`.
- Both finish with `Verified ... with SHA-256`.
- The XP-900 and human paths are under `mg/` and `human/`, respectively.
- Every sampled action has width 12 in raw simulator form.
- `obs/agentview_rgb`, an environment name, and a language instruction exist.

Human evidence:

- Both dry-run files
- Both download-and-verify logs
- Both `subset-manifest.csv` files
- `inspection.json`

## 4. Published RLDS data

The artifact download in Section 2 places the public XP-900 PnP RLDS dataset
under `$BARX_ARTIFACT_ROOT/data/mg_pnp_lite`. Audit its TFDS metadata and shard
inventory:

```bash
uv run --locked --no-dev python - \
  > "$BARX_EVIDENCE/04-rlds/published-rlds.json" <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["BARX_ARTIFACT_ROOT"]) / "data" / "mg_pnp_lite"
info_paths = list(root.rglob("dataset_info.json"))
feature_paths = list(root.rglob("features.json"))
tfrecords = [
    path for path in root.rglob("*.tfrecord-*") if ".cache" not in path.parts
]
assert len(info_paths) == 1, info_paths
assert len(feature_paths) == 1, feature_paths
assert len(tfrecords) == 256, len(tfrecords)

info = json.loads(info_paths[0].read_text())
report = {
    "root": str(root),
    "dataset_info": str(info_paths[0].relative_to(root)),
    "features": str(feature_paths[0].relative_to(root)),
    "tfrecord_shards": len(tfrecords),
    "total_bytes": sum(path.stat().st_size for path in tfrecords),
    "splits": info.get("splits"),
}
print(json.dumps(report, indent=2))
PY
```

Audit the public RLDS collection and its one-to-one relationship with the
checked-in raw subset manifests:

```bash
uv run --locked --no-dev python - \
  > "$BARX_EVIDENCE/04-rlds/collection-audit.json" <<'PY'
import json
import urllib.request
from pathlib import Path

from barx.raw_data import all_subsets

slug = "ajaysri/barx-rlds-datasets-69cad164926390ebbc395496"
request = urllib.request.Request(
    f"https://huggingface.co/api/collections/{slug}",
    headers={"User-Agent": "barx-release-audit"},
)
with urllib.request.urlopen(request, timeout=60) as response:
    collection = json.load(response)

actual = {item["id"] for item in collection["items"]}
expected = {subset.rlds_repo_id for subset in all_subsets()}
manifests = {path.stem for path in Path("dataset/subsets").glob("*.csv")}
assert collection["private"] is False
assert len(actual) == 24
assert actual == expected
assert manifests == {repo_id.split("/", 1)[1] for repo_id in expected}
assert all(not item["private"] and not item["gated"] for item in collection["items"])

print(
    json.dumps(
        {
            "collection": collection["slug"],
            "items": sorted(actual),
            "raw_subset_manifests": sorted(manifests),
        },
        indent=2,
    )
)
PY
```

Machine pass criteria:

- The pinned XP-900 PnP dataset contains one `dataset_info.json`, one
  `features.json`, and 256 TFRecord shards.
- The public collection contains exactly 24 public, ungated datasets.
- The 24 public RLDS IDs exactly match the 24 raw HDF5 subset manifest names.

Human evidence:

- `published-rlds.json`
- `collection-audit.json`

## 5. HDF5-to-RLDS conversion

Convert the downloaded 50-demo human subset. This is small enough to test the
real converter while still covering the target-embodiment path used for
adaptation:

```bash
uv run --locked --no-dev python scripts/build_rlds.py \
  --dataset target_50 --target panda --task flip_mug \
  --raw-root "$BARX_RAW_HUMAN" \
  --rlds-root "$BARX_CONVERTED_RLDS" \
  2>&1 | tee "$BARX_EVIDENCE/05-conversion/conversion.txt"
```

Inspect the resulting TFDS dataset and one decoded episode:

```bash
uv run --locked --no-dev python - \
  > "$BARX_EVIDENCE/05-conversion/converted-rlds.json" <<'PY'
import json
import os
from pathlib import Path

import tensorflow_datasets as tfds

root = Path(os.environ["BARX_CONVERTED_RLDS"])
info_paths = list(root.rglob("dataset_info.json"))
assert len(info_paths) == 1, info_paths
dataset_dir = info_paths[0].parent
builder = tfds.builder_from_directory(str(dataset_dir))
dataset = builder.as_dataset(split="train", shuffle_files=False)
episode = next(iter(tfds.as_numpy(dataset.take(1))))
steps = episode["steps"]
first = {key: value[0] for key, value in steps.items()}
report = {
    "builder_name": builder.info.name,
    "version": str(builder.info.version),
    "examples": builder.info.splits["train"].num_examples,
    "step_keys": sorted(steps),
    "observation_keys": sorted(first["observation"]),
    "action_shape": list(first["action"].shape),
    "instruction": first["language_instruction"].decode(),
}
assert report["examples"] == 50
assert report["action_shape"] == [7]
print(json.dumps(report, indent=2))
PY
```

Machine pass criteria:

- Conversion reports one selected HDF5 file and output `panda_flip_mug`.
- The TFDS train split contains 50 episodes.
- A decoded action has width 7, the public policy boundary.
- The decoded step includes images, representation annotations, and a language
  instruction.

Human evidence:

- `conversion.txt`
- `converted-rlds.json`
- The generated `dataset_info.json` and `features.json`

## 6. Training

First record the exact paper-shaped commands without allocating GPUs. These
must expose every important choice rather than relying on hidden defaults:

```bash
uv run --locked --extra train --no-dev python scripts/train.py prior \
  --prior xp_900 --task pnp --method joint_reps \
  --data-root "$BARX_ARTIFACT_ROOT/data" \
  --base-vlm "$BARX_ARTIFACT_ROOT/base-vlm" \
  --run-root "$BARX_ARTIFACT_ROOT/runs/paper-dry-run" \
  --gpus 8 --global-batch-size 256 --per-device-batch-size 32 \
  --max-steps 50000 --dry-run \
  | tee "$BARX_EVIDENCE/06-training/prior-paper-command.txt"

uv run --locked --extra train --no-dev python scripts/train.py adapt \
  --prior xp_900 --target panda --task pnp --method joint_reps \
  --data-root "$BARX_ARTIFACT_ROOT/data" \
  --base-vlm "$BARX_ARTIFACT_ROOT/base-vlm" \
  --checkpoint "$BARX_ARTIFACT_ROOT/runs/xp900-pnp-joint-reps/checkpoints/step-050000-epoch-15-loss=0.2577.pt" \
  --run-root "$BARX_ARTIFACT_ROOT/runs/paper-dry-run" \
  --gpus 8 --global-batch-size 256 --per-device-batch-size 32 \
  --max-steps 3000 --dry-run \
  | tee "$BARX_EVIDENCE/06-training/adaptation-paper-command.txt"
```

Install the GPU training extra and execute one real optimizer step:

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

Validate the durable training outputs:

```bash
uv run --locked --no-dev python - \
  > "$BARX_EVIDENCE/06-training/training-artifacts.json" <<'PY'
import json
import math
import os
from pathlib import Path

run = (
    Path(os.environ["BARX_ARTIFACT_ROOT"])
    / "runs"
    / "smoke"
    / "joint_reps--xp_900_pnp"
)
required = ["config.json", "config.yaml", "dataset_statistics.json", "run-metrics.jsonl"]
missing = [name for name in required if not (run / name).is_file()]
assert not missing, missing
metric_files = sorted(run.glob("*.jsonl"))
records = []
for path in metric_files:
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if any(
            "loss" in key.lower() or key.lower().endswith("/step")
            for key in record
        ):
            records.append({"file": path.name, "record": record})
assert records, metric_files
for item in records:
    for value in item["record"].values():
        if isinstance(value, float):
            assert math.isfinite(value)
report = {
    "run": str(run),
    "required_files": required,
    "metric_files": [path.name for path in metric_files],
    "metric_records": records,
    "checkpoints_written": [path.name for path in (run / "checkpoints").glob("*.pt")],
}
assert report["checkpoints_written"] == []
print(json.dumps(report, indent=2))
PY
```

Machine pass criteria:

- The prior dry run contains learning rate `2e-5`, global batch 256, 8
  processes, `joint_reps`, and 50,000 steps.
- The adaptation dry run contains the explicit selected prior checkpoint and
  target/prior statistics mapping.
- The one-step run exits zero and writes finite local metrics.
- It writes configuration and dataset statistics but no disposable 5.6 GiB
  final checkpoint.
- W&B is not contacted unless the runner explicitly adds `--use-wandb`.

Human evidence:

- Both paper command files
- `one-step-training.txt`
- `training-artifacts.json`
- The run's `config.json`, `dataset_statistics.json`, and JSONL metrics

## 7. Headless evaluation

Verify that the frozen Panda PnP Counter-to-Sink evaluation conditions restore
and hash-check:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
uv run --locked --no-dev python scripts/verify_eval_conditions.py \
  --task pnp_counter_to_sink --embodiment panda \
  2>&1 | tee "$BARX_EVIDENCE/07-eval/conditions.txt"
```

Run one literal trial using the downloaded public checkpoint:

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
```

Aggregate and validate the machine-readable result:

```bash
uv run --locked --no-dev python scripts/summarize_evaluations.py \
  "$BARX_ARTIFACT_ROOT/rollouts" \
  --output "$BARX_EVIDENCE/07-eval/results.csv"

uv run --locked --no-dev python - \
  > "$BARX_EVIDENCE/07-eval/evaluation-artifacts.json" <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["BARX_ARTIFACT_ROOT"]) / "rollouts"
summaries = list(root.rglob("summary.json"))
assert len(summaries) == 1, summaries
run = summaries[0].parent
summary = json.loads((run / "summary.json").read_text())
config = json.loads((run / "config.json").read_text())
episodes = [
    json.loads(line)
    for line in (run / "episodes.jsonl").read_text().splitlines()
]
videos = sorted(path.name for path in run.glob("*.mp4"))
assert summary["status"] == "complete", summary
assert summary["episodes"] == 1, summary
assert len(episodes) == 1
assert episodes[0]["seed"] == 1000
assert videos
print(
    json.dumps(
        {
            "run_directory": str(run),
            "summary": summary,
            "episode": episodes[0],
            "checkpoint": config["pretrained_checkpoint"],
            "videos": videos,
        },
        indent=2,
    )
)
PY
```

Machine pass criteria:

- Frozen conditions pass metadata, MuJoCo XML, and simulator-state hash checks.
- Evaluation uses seed 1000, action horizon 8, and the Panda camera/gripper
  pairing.
- `summary.json` has status `complete` and one episode.
- `episodes.jsonl`, `config.json`, `log.txt`, `summary.json`, and at least one
  MP4 exist.
- A task failure is acceptable for this one-trial execution check; a simulator
  or model-loading failure is not.

Human review:

- Open the MP4 and confirm that it shows a Panda PnP Counter-to-Sink rollout,
  the camera is usable, frames advance normally, and the terminal success flag
  agrees with the visible outcome.
- Compare the video, `episode` record, and instruction in
  `evaluation-artifacts.json`.
- Do not interpret the single outcome as a success-rate estimate.

Human evidence:

- `conditions.txt`
- `one-trial.txt`
- `results.csv`
- `evaluation-artifacts.json`
- The rollout MP4

## 8. MimicGen regeneration gate

The paper uses MimicGen to synthesize the XP and SP prior datasets from human
source demonstrations. A credible public release must provide all of the
following:

- a pinned BARX-compatible MimicGen fork or vendored package with its upstream
  license;
- task interfaces for PnP Counter to Sink, PnP Sink to Counter, Turn On Sink
  Faucet, and Flip Mug Upright;
- robot/task configurations with no private filesystem paths;
- a source-preparation command that never modifies the downloaded human HDF5
  in place;
- a generation launcher with explicit source, task, embodiment, seed, number
  of requested successes, output directory, and video path;
- a one-success smoke mode suitable for acceptance testing; and
- output that can be inspected with the same portable HDF5 structural checks
  used in Section 3.

Check whether the public interface is present:

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
  if test "$missing" -eq 0; then
    uv run --locked --no-dev python scripts/prepare_mimicgen_source.py --help
    uv run --locked --no-dev python scripts/generate_mimicgen.py --help
  fi
  exit "$missing"
) 2>&1 | tee "$BARX_EVIDENCE/08-mimicgen/interface.txt"
```

At the document's initial revision, this command is expected to fail because
the public generator is not yet integrated. Record **BLOCKED — public BARX
MimicGen interface absent** and stop. Do not install an arbitrary upstream
MimicGen version and claim equivalence: BARX needs its task-specific interfaces
and configurations.

Once the interface is published, this section must be revised to include one
literal command using the downloaded Panda human Flip Mug source. The required
human-visible evidence is:

- the resolved generation config and seed;
- source-preparation and generation logs;
- generated success and failed-attempt HDF5 files;
- an inventory containing demonstration counts, action shape, simulator states,
  environment name, and portable asset paths;
- success/attempt counts; and
- an MP4 of the generated trajectory.

The generated file must pass a one-row manifest built from the output and the
same HDF5 structural verifier. The source file's SHA-256 must be identical
before and after preparation, proving that preparation wrote a separate file.

## 9. Paper-scale reproduction extension

Run this only after Sections 0–8 pass.

Paper facts that the commands must preserve:

- Source embodiments: IIWA, Kinova3, and UR5e.
- Target embodiments: Panda, Panda-OG, and Jaco.
- Tasks: both PnP directions, Turn On Sink Faucet, and Flip Mug Upright.
- XP-900: 300 demonstrations per task/source embodiment.
- XP-3K: 1,000 demonstrations per task/source embodiment.
- Target data: 50 human demonstrations per task/target embodiment.
- MiniVLA initialized from the released base VLM without robot pretraining.
- Single third-person camera input.
- Seven policy action dimensions in chunks of eight.
- Learning rate `2e-5`, global batch 256, 8 GPUs × 32 examples/GPU.
- PnP directions trained jointly; faucet and flip-mug trained separately.
- Main evaluation predicts actions directly, even for representation-trained
  models.
- Each task/embodiment result uses 100 frozen conditions and the best of three
  evaluated checkpoints.

For each paper experiment, save:

- the exact resolved launcher command;
- Git commit, artifact revisions, dataset IDs, and checkpoint SHA-256;
- `config.json`, `dataset_statistics.json`, and all JSONL metrics;
- checkpoint filenames and selected-checkpoint rationale;
- one `summary.json` and `episodes.jsonl` per checkpoint/task/embodiment;
- aggregated CSVs; and
- representative rollout videos.

The paper reports comparisons among No Reps, single representations, ECoT, and
Joint Reps; comparisons across no prior, XP-900, XP-3K, and SP-900; a
representation-inference ablation; and action-free transfer. A release may
support only a declared subset, but it must not imply that a smoke test
reproduces every paper figure.

## 10. Repository cleanliness

Return to the checkout after every artifact has been written outside it:

```bash
cd "$BARX_REPO"
{
  git status --porcelain
  git diff --exit-code
  git diff --cached --exit-code
  git rev-parse HEAD
} 2>&1 | tee "$BARX_EVIDENCE/09-final/git-clean.txt"
```

Pass criteria:

- `git status --porcelain` is empty.
- Both diff commands return zero.
- The final commit equals the starting commit in `00-system/git.txt`.

## 11. Required independent report

Write `$BARX_EVIDENCE/BARX_RELEASE_ACCEPTANCE_REPORT.md` with this structure:

```markdown
# BARX public release acceptance report

- Date:
- Tester:
- Repository commit:
- Machine / GPU:
- Fresh clone: yes/no
- Empty BARX and Hugging Face caches: yes/no
- Anonymous Hugging Face access: yes/no

| Section | Status | Duration | Evidence | Notes |
| --- | --- | ---: | --- | --- |
| 0. System and clone | PASS/FAIL/BLOCKED | | | |
| 1. Install, assets, tests | PASS/FAIL/BLOCKED | | | |
| 2. Artifacts and checkpoints | PASS/FAIL/BLOCKED | | | |
| 3. Raw HDF5 | PASS/FAIL/BLOCKED | | | |
| 4. Published RLDS | PASS/FAIL/BLOCKED | | | |
| 5. HDF5-to-RLDS conversion | PASS/FAIL/BLOCKED | | | |
| 6. One-step training | PASS/FAIL/BLOCKED | | | |
| 7. One-trial evaluation | PASS/FAIL/BLOCKED | | | |
| 8. MimicGen generation | PASS/FAIL/BLOCKED | | | |
| 10. Git cleanliness | PASS/FAIL/BLOCKED | | | |

## First blocking issue

Exact command, exit code, concise error, and whether retrying from a clean state
changed the result.

## Human review

- HDF5 inspection:
- Training metrics:
- Evaluation video:
- MimicGen video:

## Paper-facing assessment

State separately whether the public release:

1. executes the representative end-to-end path;
2. exposes the inputs and outputs needed for paper-scale runs; and
3. reproduces any paper metric (only claim this after the full protocol).

## Suggested fixes

Prioritized, concrete, and limited to issues observed during this run.
```

The overall status is:

- **PASS** only if Sections 0–8 and 10 pass.
- **BLOCKED** if a required public component is absent.
- **FAIL** if a present/documented component does not work as written.

Do not average section statuses or soften a missing MimicGen workflow into a
partial pass.

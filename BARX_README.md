# BARX

This is the single entry point for BARX training, RoboCasa-X evaluation, and data generation. The release exposes a
small preset CLI; users choose a stage, data split, task, representation, and robot instead of reconstructing long
experiment commands.

Teleoperation and preparation of the first MimicGen source demonstration are intentionally deferred to the next
phase.

## 1. Install

BARX is tested with Python 3.10 and an NVIDIA GPU.

```bash
conda create -n barx python=3.10 -y
conda activate barx
pip install -e ".[sim,dev]"
barx setup-sim
```

`setup-sim` checks out the four simulator forks at the locked commits, obtains the RoboCasa kitchen assets, and
validates imports. The locked forks must be readable from the machine running setup; they need to be public before an
anonymous external beta.

Flash Attention is recommended for training and model evaluation:

```bash
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

List every supported choice:

```bash
barx presets
```

The preset source of truth is [`barx/configs/presets.yaml`](barx/configs/presets.yaml). It contains the paper training
schedules, representation transforms, task names, robot classes, grippers, cameras, action layouts, and evaluation
horizons.

## 2. Download one training setting

Set the two reusable paths:

```bash
export BARX_DATA_ROOT=data
export BARX_BASE_VLM=models/base-vlm
export BARX_VQ_ROOT=vq
```

Download the base VLM once:

```bash
hf download ajaysri/prism-qwen25-extra-dinosiglip-224px-0_5b-stage-finetune-x7 \
  --local-dir "$BARX_BASE_VLM"
```

Then ask BARX for exactly the physical datasets and VQ tokenizer needed by a training setting:

```bash
barx download \
  --stage pretrain \
  --split xp900 \
  --task pnp
```

For finetuning or target-only training, also select a target robot:

```bash
barx download \
  --stage finetune \
  --split xp3k \
  --task pnp \
  --robot jaco
```

The public artifacts are also browsable in the [RoboCasa-X datasets](https://huggingface.co/collections/ajaysri/robocasa-x),
[VQ tokenizers](https://huggingface.co/collections/ajaysri/robocasa-x-vq-action-tokenizers), and
[released pretraining models](https://huggingface.co/collections/ajaysri/barx-pretraining-models-joint-reps-and-no-reps)
collections.

## 3. Train

Pretrain Joint Reps on the cross-embodiment XP-900 PnP split:

```bash
barx train \
  --stage pretrain \
  --split xp900 \
  --task pnp \
  --representation joint-reps \
  --gpus 8
```

Finetune that representation on Jaco target demonstrations. Keep the downloaded run layout intact: the checkpoint
must remain under `<run>/checkpoints/`, beside the run's `config.json` and `dataset_statistics.json`.

```bash
barx train \
  --stage finetune \
  --split xp900 \
  --task pnp \
  --representation joint-reps \
  --robot jaco \
  --checkpoint runs/PRETRAIN_RUN/checkpoints/CHECKPOINT.pt \
  --gpus 8
```

Train a target-only baseline by changing `--stage` and omitting the checkpoint:

```bash
barx train \
  --stage target-only \
  --split xp900 \
  --task pnp \
  --representation no-reps \
  --robot jaco \
  --gpus 8
```

Supported representations are `no-reps`, `joint-reps`, `bbox`, `ee-trace`, `language-motion`, and `ecot`. Supported
splits are `xp900`, `xp3k`, and `sp900`. Add `--wandb` only when desired. Add `--dry-run` to print the resolved
training command without launching it.

BARX automatically selects the logical data mixture, physical tokenizer, action-statistics mapping, batch settings,
and paper schedule. Finetuning reuses the prior's tokenizer and normalization statistics.

## 4. Evaluate

Evaluate on the matching target setting:

```bash
barx eval \
  --checkpoint runs/FINETUNE_RUN/checkpoints/CHECKPOINT.pt \
  --environment pnp-counter-to-sink \
  --robot jaco \
  --trials 100
```

To test generalization, keep the checkpoint fixed and change only the environment or robot:

```bash
barx eval \
  --checkpoint runs/FINETUNE_RUN/checkpoints/CHECKPOINT.pt \
  --environment pnp-sink-to-counter \
  --robot panda \
  --trials 100
```

Evaluation infers the correct action normalization key from run metadata and writes videos, `trials.jsonl`, and
`result.json` below `experiments/rollouts/`. Start with `--trials 1`; use `--no-videos` for large sweeps.

## 5. Generate new RoboCasa-X data

MimicGen starts from one prepared source HDF5 containing `states`, `actions`, and `datagen_info`. Source collection
and preparation are the only inputs not covered in this phase.

Generate one fresh target-robot trajectory:

```bash
barx generate \
  --source artifacts/source/demo_seed0.hdf5 \
  --output artifacts/raw \
  --environment pnp-counter-to-sink \
  --robot jaco \
  --num-demos 1 \
  --seed 0
```

Replay states into the robot-relative third-person camera and wrist camera:

```bash
barx render \
  --input artifacts/raw \
  --robot jaco \
  --workers 1 \
  --max-demos 1
```

Convert the rendered HDF5 to the common RLDS schema:

```bash
barx build-rlds \
  --input artifacts/raw \
  --output artifacts/rlds \
  --robot jaco \
  --max-files 1
```

The robot preset chooses the camera and Panda/non-Panda action conversion. Remove the smoke limits only after the
one-demo pipeline succeeds.

## 6. Test before a beta

```bash
python -m unittest discover -s tests -p "test_barx_cli.py" -v
python -m compileall -q barx scripts/robocasa_x
bash -n experiments/robot/libero/extern/robocasa-x-eval/*.sh
barx setup-sim
```

For a new-user beta, the shortest useful path is: install, `barx presets`, one `barx download --dry-run`, one real
download, one `barx train --dry-run`, and one one-trial `barx eval` with a released run directory. Report any step
that requires an undocumented flag or source edit as a release bug.

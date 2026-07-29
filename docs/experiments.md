# Reproducing simulation experiments

The machine-readable constants are in `configs/experiments.toml`. Full
training starts by training the task-specific VQ tokenizer from the released
RLDS data.

For the public XP-900 PnP walkthrough, download the already prepared RLDS data,
VQ tokenizer, and base VLM with `scripts/download_public_artifacts.py` as shown
in the root README. No Hugging Face token is required. The manual VQ procedure
below is for rebuilding that tokenizer or preparing another dataset.

## Methods

- **No Reps** predicts actions only.
- **Single Rep** alternates actions with one of bounding box, language motion,
  or end-effector trace followed by actions.
- **Joint Reps** alternates actions with each representation independently.
- **ECoT** predicts bounding box → end-effector trace → language motion →
  actions as one chain, or predicts actions directly.

The release launcher accepts `no_reps`, `bounding_box`, `language_motion`,
`end_effector_trace`, `joint_reps`, or `ecot`. The compatibility layer resolves
these paper names to the exact original transform strings. During the main
paper evaluations, representations are used during training but actions are
predicted directly at inference unless evaluating the representation-inference
ablation.

## RoboCasa-X task variants

The paper tasks are registered as separate `X*` environments; they do not
replace the corresponding upstream RoboCasa tasks. This keeps unmodified
RoboCasa behavior available while making the benchmark changes explicit:

- all embodiments use the paper's robot base offsets, initial arm poses, and
  zero-height Omron torso pose;
- object names in instructions strip asset-number suffixes;
- **PnP Counter to Sink** samples the manipulated object in a `0.30 x 0.40`
  region and requires the released object to be inside the sink;
- **PnP Sink to Counter** samples the manipulated object in a `0.25 x 0.25`
  sink region and the goal receptacle in a `0.30 x 0.30` counter region;
- **Turn On Sink Faucet** fixes the upstream faucet task to the `turn_on`
  behavior; and
- **Flip Mug Upright** is the new BARX task, using the feasible mug-asset pool
  and an initial mug pose rotated onto its side.

The evaluation scene distribution uses layouts 4, 7, and 8 with styles 0--11.
Its layout-8 styles are 0, 1, 2, 4, 7, 8, 10, and 11, matching the paper
appendix. Panda-OG uses styles 0--3 and 5--11 for PnP Sink to Counter because
the Franka Hand cannot reach the style-4 sampled goal.
Turn On Sink Faucet uses styles 0, 1, 2, 3, 4, 7, 8, 10, and 11 over all
compatible layouts. These rules live in `barx/benchmark.py` and are serialized
into every frozen condition bundle.

Each embodiment selects its robot, gripper, and calibrated third-person camera
as one configuration. Camera pose randomization is used in the released
training data, but evaluation uses the fixed base camera pose.

## VQ tokenizer

Train one tokenizer per prior/task dataset before policy training. The paper
uses 7 action dimensions, an 8-action chunk (`future_action_horizon=7`), 256
codes, 7 residual groups, and 512 latent dimensions. For XP-900 PnP:

```bash
uv run --locked --no-dev python policy/vla-scripts/pretrain_vq.py \
  --data_dir /data/barx-rlds --data_mix xp_900_pnp \
  --save_folder /tmp/barx-vq --action_dim 7 \
  --future_action_horizon 7 --vqvae_n_embed 256 \
  --vqvae_groups 7 --n_latent_dims 512

mkdir -p policy/vq/mg_pnp_lite/checkpoints
cp /tmp/barx-vq/pretrain_vq+mx-xp_900_pnp+fach-7+ng-7+nemb-256+nlatent-512/checkpoints/model.pt \
  policy/vq/mg_pnp_lite/checkpoints/model.pt
```

The destination names are the stored identifiers in the root README table.
Checkpoint paths are ignored by Git.
VQ training writes `metrics.jsonl` locally. Add `--use_wandb` only when you
explicitly want to mirror those metrics to Weights & Biases.

## Training protocol

- MiniVLA architecture and base VLM initialization
- VQ action chunks of length 8
- learning rate `2e-5`
- global batch size `256` (8 GPUs × 32 per GPU)
- no image augmentation
- target data weighted 20× relative to XP-900 (legacy prior weight 0.05)
- target data weighted 100× relative to XP-3K (legacy prior weight 0.01)
- target actions normalized with prior statistics and the prior VQ tokenizer
  reused during co-finetuning
- the two pick-and-place tasks trained jointly; other tasks trained separately

For adaptation, use the 50k-step XP-900 checkpoint, the 150k-step XP-3K No Reps
checkpoint, or the 250k-step XP-3K checkpoint with BARX representations. Pass
the base VLM, data root, and selected prior checkpoint to the training command.

Paper-facing launchers print the complete underlying command before running it.
For example:

```bash
uv run --locked --extra train --no-dev python scripts/train.py prior \
  --prior xp_900 --task pnp --method joint_reps \
  --data-root /data/barx-rlds --base-vlm /models/minivla \
  --max-steps 50000

uv run --locked --extra train --no-dev python scripts/train.py adapt \
  --prior xp_900 --target panda --task pnp --method joint_reps \
  --data-root /data/barx-rlds --base-vlm /models/minivla \
  --checkpoint /models/xp_900_joint_reps_step_50000.pt \
  --max-steps 3000
```

Use `--prior none` for target-only training and `--prior sp_900` for the
same-embodiment prior. Those settings initialize from the base VLM and do not
take a source-prior checkpoint. Add `--dry-run` to inspect a command without
starting a distributed job.

Training writes JSONL metrics and checkpoints locally by default. To mirror a
run to Weights & Biases, add `--use-wandb` and optionally `--wandb-project` and
`--wandb-entity`; no account or project name is embedded in the default run.

For a quick one-GPU A40 initialization and optimizer check, use the command
below after setting `BARX_ARTIFACT_ROOT`, `HF_HOME`, and `BARX_VQ_ROOT` as
shown in the root README:

```bash
uv run --locked --extra train --no-dev python scripts/train.py prior \
  --prior xp_900 --task pnp --method joint_reps \
  --data-root "$BARX_ARTIFACT_ROOT/data" \
  --base-vlm "$BARX_ARTIFACT_ROOT/base-vlm" \
  --run-root "$BARX_ARTIFACT_ROOT/runs/quick-check" \
  --gpus 1 --global-batch-size 1 --per-device-batch-size 1 \
  --max-steps 1 --save-interval 100 --skip-final-checkpoint
```

This does not reproduce the paper batch, but it verifies the complete training
path without writing a roughly 5.6 GiB final checkpoint.

## Evaluation protocol

Use `scripts/evaluate.py` for paper-facing task and embodiment names. Each
task/embodiment setting uses 100 fixed held-out conditions selected from a
candidate stream starting at seed 1000, and executes all 8 predicted actions
before replanning. Pick-and-place targets must come from `obj_set1`, instance
split `A`, and expose at least 25 segmentation pixels in the policy camera.
The launcher loads a frozen condition bundle by default, including the exact
processed MuJoCo model and post-settling simulator state. Robot, gripper, and
calibrated agent camera are selected together from the embodiment; they are not
independent command-line choices.

| Task | Maximum steps |
| --- | ---: |
| PnP Counter to Sink | 600 |
| PnP Sink to Counter | 650 |
| Turn On Sink Faucet | 500 |
| Flip Mug Upright | 500 |

The paper evaluates three checkpoints per model and reports the best success
rate separately for each task/embodiment combination.

To try the optional source-prior checkpoint downloaded with
`--include-pretrain-checkpoint`:

```bash
uv run --locked --extra train --no-dev python scripts/evaluate.py \
  --checkpoint "$BARX_ARTIFACT_ROOT/runs/xp900-pnp-joint-reps/checkpoints/step-050000-epoch-15-loss=0.2577.pt" \
  --embodiment panda --task pnp_counter_to_sink --unnorm-key mg_pnp_lite \
  --episodes 1 --rollout-dir "$BARX_ARTIFACT_ROOT/rollouts/one-trial"
```

One trial checks execution but is not a success-rate estimate. Omit
`--episodes 1` to run all 100 conditions. Both commands default to
action-only inference, as used for the main results. For the inference
ablation, add `--inference-representation bounding_box`,
`language_motion`, or `end_effector_trace`. The launcher maps Panda-OG and Jaco
to their registered internal simulator classes.

Each run creates a new directory under `rollouts/<task>/<embodiment>/` with its
configuration, per-trial results, summary, log, and rollout videos. Add
`--use-wandb` to also send metrics and the first-trial videos to Weights &
Biases.

Aggregate any collection of complete and failed runs without parsing text logs:

```bash
uv run --locked --no-dev python scripts/summarize_evaluations.py rollouts \
  --output evaluation/results.csv
```

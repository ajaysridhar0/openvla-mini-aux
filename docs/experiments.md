# Reproducing simulation experiments

The machine-readable constants are in `configs/experiments.toml`. This release
does not include model or VQ tokenizer checkpoints, so full training starts by
training the task-specific VQ tokenizer from the released RLDS data.

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
- object names in instructions omit asset-number suffixes;
- **PnP Counter to Sink** samples the manipulated object in a `0.30 x 0.40`
  region and requires the released object to be inside the sink;
- **PnP Sink to Counter** samples the manipulated object in a `0.25 x 0.25`
  sink region and the goal receptacle in a `0.30 x 0.30` counter region;
- **Turn On Sink Faucet** fixes the upstream faucet task to the `turn_on`
  behavior; and
- **Flip Mug Upright** is the new BARX task, with four infeasible mug assets
  excluded and the initial mug rotated onto its side.

The evaluation scene distribution uses layouts 4, 7, and 8 with styles 0--11.
It excludes `(layout, style)` pairs `(8, 3)`, `(8, 5)`, `(8, 6)`, and `(8, 9)`
as described in the paper appendix. Panda-OG additionally excludes style 4 for
PnP Sink to Counter because the Franka Hand cannot reach that sampled goal.
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
cd policy
uv run --locked python vla-scripts/pretrain_vq.py \
  --data_dir /data/barx-rlds --data_mix xp_900_pnp \
  --save_folder /tmp/barx-vq --action_dim 7 \
  --future_action_horizon 7 --vqvae_n_embed 256 \
  --vqvae_groups 7 --n_latent_dims 512

mkdir -p vq/mg_pnp_lite/checkpoints
cp /tmp/barx-vq/pretrain_vq+mx-xp_900_pnp+fach-7+ng-7+nemb-256+nlatent-512/checkpoints/model.pt \
  vq/mg_pnp_lite/checkpoints/model.pt
```

The destination names are the stored identifiers in the root README table.
Checkpoint paths are ignored by Git.

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

The original source-prior checkpoint selected for adaptation was 50k steps for
XP-900, 150k for XP-3K No Reps, and 250k for XP-3K with BARX
representations. Each public training command must provide its base VLM,
data root, and (for adaptation) selected prior checkpoint explicitly; no
private filesystem paths are embedded in release entry points.

Paper-facing launchers print the complete underlying command before running it.
For example:

```bash
uv run --locked python scripts/train.py prior \
  --prior xp_900 --task pnp --method joint_reps \
  --data-root /data/barx-rlds --base-vlm /models/minivla \
  --max-steps 50000

uv run --locked python scripts/train.py adapt \
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

## Evaluation protocol

Use `scripts/evaluate.py` for paper-facing task and embodiment names. Each
task/embodiment setting uses 100 fixed held-out seeds, starting from seed 1000
in the release evaluator, and executes all 8 predicted actions before
replanning. The launcher loads a frozen condition bundle by default, including
the exact processed MuJoCo model and post-settling simulator state. Robot,
gripper, and calibrated agent camera are selected together from the embodiment;
they are not independent command-line choices.

| Task | Maximum steps |
| --- | ---: |
| PnP Counter to Sink | 600 |
| PnP Sink to Counter | 650 |
| Turn On Sink Faucet | 500 |
| Flip Mug Upright | 500 |

The paper evaluates three checkpoints per model and reports the best success
rate separately for each task/embodiment combination.

Example (checkpoint path and normalization key supplied by the user):

```bash
uv run --locked python scripts/evaluate.py \
  --checkpoint /path/to/checkpoint.pt \
  --embodiment panda \
  --task pnp_counter_to_sink \
  --unnorm-key mg_pnp_lite
```

This defaults to action-only inference, as used for the main results. For the
inference ablation, add `--inference-representation bounding_box`,
`language_motion`, or `end_effector_trace`. The launcher maps Panda-OG and Jaco
to their registered internal simulator classes.

Each invocation creates a non-overwriting directory under
`rollouts/<task>/<embodiment>/`. It contains the resolved `config.json`, one
record per trial in `episodes.jsonl`, an atomic `summary.json`, a readable
`log.txt`, and rollout videos. Failed runs retain their completed episode
records and write the exception to the summary before returning a nonzero exit
status. Add `--use-wandb` to mirror metrics and first-trial videos; the local
files remain the source of record.

Aggregate any collection of complete and failed runs without parsing text logs:

```bash
uv run --locked python scripts/summarize_evaluations.py rollouts \
  --output evaluation/results.csv
```

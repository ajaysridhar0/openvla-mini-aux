#!/usr/bin/env bash

# Compatibility wrapper. New users should call `barx eval` directly.
set -euo pipefail

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a BARX checkpoint.}"

case "${TASK:-PnPCounterToSink}" in
    PnPCounterToSink) ENVIRONMENT=pnp-counter-to-sink ;;
    PnPSinkToCounter) ENVIRONMENT=pnp-sink-to-counter ;;
    TurnOnSinkFaucet) ENVIRONMENT=turn-on-sink-faucet ;;
    FlipMugUpright) ENVIRONMENT=flip-mug-upright ;;
    *) echo "Unsupported TASK: ${TASK}" >&2; exit 2 ;;
esac

case "${ROBOT:-JacoOmron}:${GRIPPER:-Robotiq85Gripper}" in
    PandaOmron:PandaGripper) ROBOT_PRESET=panda-og ;;
    PandaOmron:*) ROBOT_PRESET=panda ;;
    JacoOmron:*) ROBOT_PRESET=jaco ;;
    IIWAOmron:*) ROBOT_PRESET=iiwa ;;
    UR5eOmron:*) ROBOT_PRESET=ur5e ;;
    Kinova3Omron:*) ROBOT_PRESET=kinova3 ;;
    *) echo "Unsupported ROBOT/GRIPPER: ${ROBOT:-}/${GRIPPER:-}" >&2; exit 2 ;;
esac

ARGS=(
    --checkpoint "${CHECKPOINT_PATH}"
    --environment "${ENVIRONMENT}"
    --robot "${ROBOT_PRESET}"
    --trials "${NUM_TRIALS:-10}"
    --start-seed "${START_SEED:-1000}"
    --max-videos "${MAX_VIDEOS:-10}"
)

if [[ -n "${UNNORM_KEY:-}" ]]; then ARGS+=(--unnorm-key "${UNNORM_KEY}"); fi
if [[ -n "${ROLLOUT_DIR:-}" ]]; then ARGS+=(--output "${ROLLOUT_DIR}"); fi
if [[ "${SAVE_VIDEOS:-True}" == "False" ]]; then ARGS+=(--no-videos); fi
if [[ "${USE_WANDB:-False}" == "True" ]]; then ARGS+=(--wandb); fi

exec python -m barx.cli eval "${ARGS[@]}" "$@"

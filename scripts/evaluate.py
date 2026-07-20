#!/usr/bin/env python3
"""Paper-facing launcher for BARX RoboCasa-X evaluation."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "policy" / "experiments" / "robot" / "robocasa_x" / "evaluate.py"
TARGETS = {
    "panda": ("PandaOmron", "Robotiq85Gripper"),
    "panda_og": ("PandaOmron", "PandaGripper"),
    "jaco": ("JacoOmron", "default"),
    "iiwa": ("IIWAOmron", "default"),
    "kinova3": ("Kinova3Omron", "default"),
    "ur5e": ("UR5eOmron", "default"),
}
TASKS = {
    "pnp_counter_to_sink": ("PnPCounterToSink", 600),
    "pnp_sink_to_counter": ("PnPSinkToCounter", 650),
    "turn_on_sink_faucet": ("TurnOnSinkFaucet", 500),
    "flip_mug_upright": ("FlipMugUpright", 500),
}


def command(args: argparse.Namespace) -> list[str]:
    robot, gripper = TARGETS[args.embodiment]
    task, default_steps = TASKS[args.task]
    rollout_dir = args.rollout_dir or ROOT / "rollouts" / task / args.embodiment / "STEP"
    result = [
        sys.executable,
        str(EVALUATOR),
        "--pretrained_checkpoint",
        str(args.checkpoint),
        "--robot",
        robot,
        "--gripper_types",
        gripper,
        "--task",
        task,
        "--unnorm_key",
        args.unnorm_key,
        "--num_trials_per_task",
        str(args.episodes),
        "--start_seed",
        str(args.start_seed),
        "--max_steps",
        str(args.max_steps or default_steps),
        "--act_horizon",
        "8",
        "--rollout_dir",
        str(rollout_dir),
    ]
    if args.inference_representation != "none":
        result.extend(["--inference_representation", args.inference_representation])
    if args.use_wandb:
        result.extend(["--use_wandb", "True"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embodiment", choices=TARGETS, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument(
        "--inference-representation",
        choices=("none", "bounding_box", "language_motion", "end_effector_trace"),
        default="none",
        help="Optional representation to predict before actions; paper main results use none",
    )
    parser.add_argument("--unnorm-key", required=True, help="Prior dataset normalization key stored in the model")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--rollout-dir", type=Path)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    evaluation_command = command(args)
    print(shlex.join(evaluation_command))
    if not args.dry_run:
        subprocess.run(evaluation_command, cwd=ROOT / "policy", check=True)


if __name__ == "__main__":
    main()

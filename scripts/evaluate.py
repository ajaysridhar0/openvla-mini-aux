#!/usr/bin/env python3
"""Paper-facing launcher for BARX RoboCasa-X evaluation."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from barx.benchmark import (
    ACTION_HORIZON,
    EMBODIMENTS,
    EVALUATION_EPISODES,
    EVALUATION_START_SEED,
    TASKS,
)

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "policy" / "experiments" / "robot" / "robocasa_x" / "evaluate.py"


def command(args: argparse.Namespace) -> list[str]:
    embodiment = EMBODIMENTS[args.embodiment]
    task = TASKS[args.task]
    conditions_dir = (
        getattr(args, "conditions_dir", None)
        or ROOT / "evaluation" / "conditions"
    )
    rollout_dir = (
        args.rollout_dir
        or ROOT / "rollouts" / args.task / args.embodiment / "STEP"
    )
    result = [
        sys.executable,
        str(EVALUATOR),
        "--pretrained_checkpoint",
        str(args.checkpoint),
        "--embodiment",
        args.embodiment,
        "--robot",
        embodiment.robot,
        "--gripper_types",
        embodiment.gripper,
        "--camera",
        embodiment.camera,
        "--task",
        task.environment,
        "--unnorm_key",
        args.unnorm_key,
        "--num_trials_per_task",
        str(args.episodes),
        "--start_seed",
        str(args.start_seed),
        "--conditions_dir",
        str(conditions_dir),
        "--max_steps",
        str(args.max_steps or task.max_steps),
        "--act_horizon",
        str(ACTION_HORIZON),
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
    parser.add_argument("--embodiment", choices=EMBODIMENTS, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument(
        "--inference-representation",
        choices=("none", "bounding_box", "language_motion", "end_effector_trace"),
        default="none",
        help="Optional representation to predict before actions; paper main results use none",
    )
    parser.add_argument("--unnorm-key", required=True, help="Prior dataset normalization key stored in the model")
    parser.add_argument("--episodes", type=int, default=EVALUATION_EPISODES)
    parser.add_argument("--start-seed", type=int, default=EVALUATION_START_SEED)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--rollout-dir", type=Path)
    parser.add_argument("--conditions-dir", type=Path)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    evaluation_command = command(args)
    print(shlex.join(evaluation_command))
    if not args.dry_run:
        subprocess.run(evaluation_command, cwd=ROOT / "policy", check=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paper-facing launcher for BARX source-prior training and adaptation."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY_ROOT = ROOT / "policy"
TRAINER = "vla-scripts/train.py"
VLA_TYPE = "prism-qwen25-dinosiglip-224px+0_5b+mx-xembod-robocasa-full"
TASKS = ("pnp", "turn_on_sink", "flip_mug")
TARGETS = ("panda", "panda_og", "jaco")
METHODS = (
    "no_reps",
    "bounding_box",
    "language_motion",
    "end_effector_trace",
    "joint_reps",
    "ecot",
)


def prior_dataset(prior: str, task: str) -> str:
    return f"{prior}_{task}"


def adaptation_dataset(prior: str, target: str, task: str) -> str:
    if prior == "none":
        return f"{target}_{task}"
    return f"{target}_{prior}_{task}"


def tokenizer_name(prior: str, target: str | None, task: str) -> str:
    if prior in {"xp_900", "xp_3k"}:
        return f"{prior}_{task}_action_tokenizer"
    if prior == "sp_900" and target:
        return f"{target}_sp_900_{task}_action_tokenizer"
    if target:
        return f"{target}_{task}_vq_extra_action_tokenizer"
    raise ValueError("A target embodiment is required for this tokenizer.")


def statistics_map(prior: str, target: str, task: str) -> dict[str, str] | None:
    if prior in {"xp_900", "xp_3k"}:
        legacy_prior = f"mg_{task}" + ("_lite" if prior == "xp_900" else "")
    elif prior == "sp_900":
        legacy_prior = f"mg_{target}_{task}"
    else:
        return None
    return {f"{target}_{task}": legacy_prior}


def build_command(args: argparse.Namespace) -> list[str]:
    target = args.target if args.stage == "adapt" else None
    if args.stage == "prior" and args.prior not in {"xp_900", "xp_3k"}:
        raise ValueError("Source-prior training supports xp_900 or xp_3k.")
    if args.stage == "adapt" and args.prior in {"xp_900", "xp_3k"} and args.checkpoint is None:
        raise ValueError("XP adaptation requires --checkpoint from the selected prior run.")
    if args.gpus < 1:
        raise ValueError("--gpus must be positive.")
    global_batch_size = getattr(args, "global_batch_size", None) or 256
    per_device_batch_size = getattr(args, "per_device_batch_size", None)
    if per_device_batch_size is None:
        if global_batch_size % args.gpus:
            raise ValueError("--global-batch-size must be divisible by --gpus.")
        per_device_batch_size = global_batch_size // args.gpus
    if global_batch_size != per_device_batch_size * args.gpus:
        raise ValueError(
            "VLA training has no gradient accumulation: global batch must equal "
            "per-device batch times GPU count."
        )
    if args.max_steps < 1 or (args.save_interval is not None and args.save_interval < 1):
        raise ValueError("--max-steps and --save-interval must be positive.")
    save_interval = args.save_interval or (10_000 if args.stage == "prior" else 1_000)

    data_mix = (
        prior_dataset(args.prior, args.task)
        if args.stage == "prior"
        else adaptation_dataset(args.prior, args.target, args.task)
    )
    command = [
        "torchrun",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc-per-node",
        str(args.gpus),
        TRAINER,
        "--vla.type",
        VLA_TYPE,
        "--vla.base_vlm",
        str(args.base_vlm),
        "--vla.data_mix",
        data_mix,
        "--data_root_dir",
        str(args.data_root),
        "--run_root_dir",
        str(args.run_root),
        "--vla.action_tokenizer",
        tokenizer_name(args.prior, target, args.task),
        "--vla.expected_world_size",
        str(args.gpus),
        "--vla.global_batch_size",
        str(global_batch_size),
        "--vla.per_device_batch_size",
        str(per_device_batch_size),
        "--vla.learning_rate",
        "2e-5",
        "--vla.lr_scheduler_type",
        "constant",
        "--vla.max_steps",
        str(args.max_steps),
        "--vla.use_wrist_image",
        "False",
        "--vla.image_sequence_len",
        "1",
        "--vla.transform_types",
        args.method,
        "--image_aug",
        "False",
        "--run_id",
        args.method,
        "--run_id_note",
        data_mix,
        "--save_interval",
        str(save_interval),
    ]
    if getattr(args, "skip_final_checkpoint", False):
        command.extend(["--save_final_checkpoint", "False"])
    if getattr(args, "hf_token_env", None):
        command.extend(["--hf_token", args.hf_token_env])
    if args.checkpoint is not None:
        command.extend(["--pretrained_checkpoint", str(args.checkpoint), "--is_resume", "False"])
    if args.stage == "adapt":
        mapping = statistics_map(args.prior, args.target, args.task)
        if mapping:
            command.extend(["--dataset_statistics_map", json.dumps(mapping, separators=(",", ":"))])
    use_wandb = getattr(args, "use_wandb", False)
    wandb_entity = getattr(args, "wandb_entity", None)
    if wandb_entity and not use_wandb:
        raise ValueError("--wandb-entity requires --use-wandb.")
    if use_wandb:
        command.extend(
            [
                "--trackers",
                '["jsonl","wandb"]',
                "--wandb_project",
                getattr(args, "wandb_project", "barx"),
            ]
        )
        if wandb_entity:
            command.extend(["--wandb_entity", wandb_entity])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prior", "adapt"))
    parser.add_argument("--prior", choices=("none", "xp_900", "xp_3k", "sp_900"), required=True)
    parser.add_argument("--target", choices=TARGETS)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-vlm", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument(
        "--save-interval",
        type=int,
        help="Defaults to 10000 for source priors and 1000 for target training",
    )
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--per-device-batch-size", type=int)
    parser.add_argument(
        "--skip-final-checkpoint",
        action="store_true",
        help="Skip a checkpoint written solely because max_steps was reached",
    )
    parser.add_argument(
        "--hf-token-env",
        help="Name of an environment variable containing a token for gated artifacts",
    )
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="barx")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.stage == "adapt" and args.target is None:
        parser.error("adapt requires --target")

    training_command = build_command(args)
    print(shlex.join(training_command))
    if not args.dry_run:
        subprocess.run(training_command, cwd=POLICY_ROOT, check=True)


if __name__ == "__main__":
    main()

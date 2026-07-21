"""Pure command and MimicGen-config builders used by the BARX CLI."""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from barx.config import ConfigError, require_target_robot, select

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_TYPE = "prism-qwen25-dinosiglip-224px+0_5b+mx-xembod-robocasa-full"


def dataset_name(stage: str, split: str, task: str, robot: Optional[str]) -> str:
    if stage == "target-only":
        if robot is None:
            raise ConfigError("--robot is required for target-only training")
        return f"robocasa-x-target-{robot}-{task}"
    if split in ("xp900", "xp3k"):
        suffix = f"-{robot}-{task}-mix" if stage == "finetune" else f"-{task}"
        return f"robocasa-x-{split}{suffix}"
    if split == "sp900":
        if robot is None:
            raise ConfigError("--robot is required for SP-900 training")
        suffix = "-mix" if stage == "finetune" else ""
        return f"robocasa-x-sp900-{robot}-{task}{suffix}"
    raise ConfigError("--split must be xp900, xp3k, or sp900")


def prior_dataset_name(split: str, task: str, robot: Optional[str]) -> str:
    return dataset_name("pretrain", split, task, robot)


def tokenizer_name(stage: str, split: str, task: str, robot: Optional[str]) -> str:
    source = (
        dataset_name("target-only", split, task, robot)
        if stage == "target-only"
        else prior_dataset_name(split, task, robot)
    )
    return f"{source}-vq-extra-action-tokenizer"


def downloadable_datasets(stage: str, split: str, task: str, robot: Optional[str]) -> List[str]:
    """Return the physical HF datasets needed by one logical training mixture."""
    if stage == "pretrain":
        return [prior_dataset_name(split, task, robot)]
    if robot is None:
        raise ConfigError(f"--robot is required for {stage} assets")
    target = dataset_name("target-only", split, task, robot)
    if stage == "target-only":
        return [target]
    if stage == "finetune":
        return [prior_dataset_name(split, task, robot), target]
    raise ConfigError("--stage must be pretrain, finetune, or target-only")


def training_schedule(
    presets: Dict[str, Any], stage: str, split: str, task: str, representation: str
) -> Tuple[int, int]:
    training = presets["training"]
    if stage in ("finetune", "target-only"):
        steps_key = "finetune_steps" if stage == "finetune" else "target_only_steps"
        return training[steps_key], training["finetune_save_interval"]
    schedule = select(select(training["pretrain"], split, "split"), task, "task")
    values = schedule.get(representation, schedule["default"])
    return values["max_steps"], values["save_interval"]


def build_train_command(args: Any, presets: Dict[str, Any]) -> Tuple[List[str], str]:
    task = select(presets["tasks"], args.task, "task")
    representation = select(presets["representations"], args.representation, "representation")
    robot = select(presets["robots"], args.robot, "robot") if args.robot else None
    if robot is not None and args.stage in ("finetune", "target-only"):
        require_target_robot(args.robot, robot)
    if args.stage == "finetune" and args.checkpoint is None:
        raise ConfigError("--checkpoint is required for finetuning")
    if args.stage == "finetune":
        checkpoint_run_dir(args.checkpoint)
    if args.stage == "pretrain" and args.split == "sp900" and robot is None:
        raise ConfigError("--robot is required for SP-900 pretraining")
    if args.gpus <= 0:
        raise ConfigError("--gpus must be positive")

    data_mix = dataset_name(args.stage, args.split, task["dataset"], args.robot)
    action_tokenizer = tokenizer_name(args.stage, args.split, task["dataset"], args.robot)
    max_steps, save_interval = training_schedule(presets, args.stage, args.split, task["dataset"], args.representation)
    max_steps = args.max_steps or max_steps
    save_interval = args.save_interval or save_interval
    run_name = args.run_name or "-".join(
        item for item in (args.stage, args.split, args.robot, task["dataset"], args.representation) if item
    )

    command = [
        "torchrun",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc-per-node",
        str(args.gpus),
        "vla-scripts/train.py",
        "--vla.type",
        presets["training"]["model_type"],
        "--vla.base_vlm",
        str(args.base_vlm),
        "--vla.data_mix",
        data_mix,
        "--data_root_dir",
        str(args.data_root),
        "--run_root_dir",
        str(args.run_root),
        "--vla.action_tokenizer",
        action_tokenizer,
        "--vla.expected_world_size",
        str(args.gpus),
        "--vla.global_batch_size",
        str(presets["training"]["global_batch_size"]),
        "--vla.per_device_batch_size",
        str(presets["training"]["per_device_batch_size"]),
        "--vla.learning_rate",
        str(presets["training"]["learning_rate"]),
        "--vla.lr_scheduler_type",
        "constant",
        "--vla.max_steps",
        str(max_steps),
        "--vla.use_wrist_image",
        "False",
        "--vla.image_sequence_len",
        "1",
        "--vla.transform_types",
        representation["transforms"],
        "--run_id",
        representation["run_id"],
        "--run_id_note",
        run_name,
        "--save_interval",
        str(save_interval),
        "--trackers",
        "[jsonl]" if not args.wandb else "[jsonl,wandb]",
    ]
    if args.wandb:
        if args.wandb_entity:
            command.extend(["--wandb_entity", args.wandb_entity])
        if args.wandb_project:
            command.extend(["--wandb_project", args.wandb_project])
    if args.checkpoint is not None:
        command.extend(["--pretrained_checkpoint", str(args.checkpoint), "--is_resume", "False"])
    if args.stage == "finetune":
        target = dataset_name("target-only", args.split, task["dataset"], args.robot)
        prior = prior_dataset_name(args.split, task["dataset"], args.robot)
        command.extend(["--dataset_statistics_map", json.dumps({target: prior})])
    if args.hf_token_env:
        command.extend(["--hf_token", args.hf_token_env])
    return command, run_name


def checkpoint_run_dir(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise ConfigError(f"Checkpoint does not exist: {checkpoint}")
    if checkpoint.parent.name != "checkpoints":
        raise ConfigError("Checkpoint must be inside <run>/checkpoints/")
    run_dir = checkpoint.parent.parent
    for filename in ("config.json", "dataset_statistics.json"):
        if not (run_dir / filename).is_file():
            raise ConfigError(f"Checkpoint run is missing {filename}: {run_dir}")
    return run_dir


def infer_unnorm_key(checkpoint: Path) -> str:
    run_dir = checkpoint_run_dir(checkpoint)
    with (run_dir / "config.json").open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    with (run_dir / "dataset_statistics.json").open("r", encoding="utf-8") as stats_file:
        keys = list(json.load(stats_file))

    stats_map = config.get("dataset_statistics_map") or {}
    mapped_values = [value for value in stats_map.values() if value in keys]
    if len(set(mapped_values)) == 1:
        return mapped_values[0]
    data_mix = config.get("vla", {}).get("data_mix")
    candidates = [data_mix, str(data_mix).replace("-", "_")]
    for candidate in candidates:
        if candidate in keys:
            return candidate
    if len(keys) == 1:
        return keys[0]
    raise ConfigError(f"Could not infer action statistics from {keys}; pass --unnorm-key explicitly")


def safe_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._") or "run"


def build_eval_command(args: Any, presets: Dict[str, Any]) -> Tuple[List[str], Path]:
    environment = select(presets["environments"], args.environment, "environment")
    robot = select(presets["robots"], args.robot, "robot")
    checkpoint = args.checkpoint.expanduser().resolve()
    run_dir = checkpoint_run_dir(checkpoint)
    unnorm_key = args.unnorm_key or infer_unnorm_key(checkpoint)
    model_tag = args.model_tag or run_dir.name
    rollout_dir = args.output or (
        args.run_root / environment["name"] / args.robot / safe_tag(model_tag) / safe_tag(checkpoint.stem)
    )
    eval_defaults = presets["evaluation"]
    command = [
        sys.executable,
        "experiments/robot/libero/run_robocasa_eval_new.py",
        "--model_family",
        "prismatic",
        "--pretrained_checkpoint",
        str(checkpoint),
        "--robot",
        robot["name"],
        "--gripper_types",
        robot["gripper"],
        "--task",
        environment["name"],
        "--unnorm_key",
        unnorm_key,
        "--num_trials_per_task",
        str(args.trials),
        "--start_seed",
        str(args.start_seed),
        "--max_steps",
        str(environment["max_steps"]),
        "--act_horizon",
        str(eval_defaults["act_horizon"]),
        "--num_steps_wait",
        str(eval_defaults["settling_steps"]),
        "--rollout_dir",
        str(rollout_dir),
        "--save_videos",
        str(not args.no_videos),
        "--max_videos",
        str(args.max_videos),
        "--use_wandb",
        str(args.wandb),
        "--aux_task_types",
        "null",
    ]
    if args.hf_token_env:
        command.extend(["--hf_token", args.hf_token_env])
    return command, rollout_dir


def build_generation_config(args: Any, presets: Dict[str, Any]) -> Dict[str, Any]:
    environment = select(presets["environments"], args.environment, "environment")
    robot = select(presets["robots"], args.robot, "robot")
    defaults = presets["generation"]
    noise = environment.get("action_noise", defaults["action_noise"])
    signal = environment.get("subtask_signal", "stage_contact_obj")
    stage_1 = {
        "object_ref": "obj" if environment["task"] != "turn-on-sink-faucet" else "handle",
        "subtask_term_signal": signal,
        "subtask_term_offset_range": environment["subtask_offset"],
        "action_noise": noise,
        "num_interpolation_steps": defaults["interpolation_steps"],
        "selection_strategy": "nearest_neighbor_interpolation",
        "selection_strategy_kwargs": {"nn_k": defaults["nearest_neighbors"]},
    }
    stage_2 = {
        "object_ref": environment["object_ref"],
        "subtask_term_signal": None,
        "subtask_term_offset_range": None,
        "action_noise": noise,
        "selection_strategy": "nearest_neighbor_interpolation",
        "selection_strategy_kwargs": {"nn_k": defaults["nearest_neighbors"]},
    }
    if environment.get("stage_2_interpolation_steps", True):
        stage_2["num_interpolation_steps"] = defaults["interpolation_steps"]
    generation_task = {"robot": robot["name"]}
    if robot.get("generation_gripper"):
        generation_task["gripper"] = robot["generation_gripper"]
    return {
        "name": environment["name"],
        "type": "robosuite",
        "experiment": {
            "source": {"dataset_path": str(args.source.expanduser().resolve())},
            "generation": {
                "path": str(args.output.expanduser().resolve()),
                "select_src_per_subtask": defaults["select_source_per_subtask"],
                "num_trials": args.num_demos,
                "guarantee": defaults["guarantee_success"],
                "keep_failed": defaults["keep_failed"],
                "transform_first_robot_pose": defaults["transform_first_robot_pose"],
            },
            "task": generation_task,
            "num_demo_to_render": defaults["render_successes"],
            "num_fail_demo_to_render": defaults["render_failures"],
            "seed": args.seed,
        },
        "task": {"task_spec": {"stage_1": stage_1, "stage_2": stage_2}},
    }

"""One small command-line interface for the BARX release."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from barx.commands import (
    REPO_ROOT,
    build_eval_command,
    build_generation_config,
    build_train_command,
    downloadable_datasets,
    tokenizer_name,
)
from barx.config import ConfigError, load_presets, require_target_robot, select

EXTERN_ROOT = REPO_ROOT / "experiments" / "robot" / "libero" / "extern" / "robocasa-x-eval"


def env_path(name: str, default: Optional[Path] = None) -> Optional[Path]:
    value = os.environ.get(name)
    return Path(value) if value else default


def print_command(command: List[str]) -> None:
    print(shlex.join(str(part) for part in command))


def run(command: List[str], *, dry_run: bool, env: Optional[Dict[str, str]] = None) -> None:
    print_command(command)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def runtime_env(extra_python_paths: Optional[List[Path]] = None) -> Dict[str, str]:
    env = os.environ.copy()
    python_paths = [
        EXTERN_ROOT / "src" / "robosuite_xembod",
        EXTERN_ROOT / "src" / "robocasa_xembod",
        EXTERN_ROOT / "src" / "robomimic_xembod",
        EXTERN_ROOT / "src" / "mimicgen_xembod",
        REPO_ROOT,
    ]
    if extra_python_paths:
        python_paths = extra_python_paths + python_paths
    existing = env.get("PYTHONPATH")
    if existing:
        python_paths.append(Path(existing))
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_paths)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("EGL_DEVICE_ID", "0")
    return env


def require_path(value: Optional[Path], flag: str, env_name: str) -> Path:
    if value is None:
        raise ConfigError(f"Set {env_name} or pass {flag}")
    path = value.expanduser().resolve()
    if not path.exists():
        raise ConfigError(f"Path does not exist: {path}")
    return path


def command_presets(args: argparse.Namespace, presets: Dict[str, Any]) -> None:
    del args
    print("Representations: " + ", ".join(presets["representations"]))
    print("Data splits: xp900, xp3k, sp900")
    print("Training tasks: " + ", ".join(presets["tasks"]))
    print("Evaluation environments: " + ", ".join(presets["environments"]))
    print("Robots: " + ", ".join(presets["robots"]))


def command_train(args: argparse.Namespace, presets: Dict[str, Any]) -> None:
    args.base_vlm = require_path(args.base_vlm, "--base-vlm", "BARX_BASE_VLM")
    args.data_root = require_path(args.data_root, "--data-root", "BARX_DATA_ROOT")
    args.run_root = args.run_root.expanduser().resolve()
    command, run_name = build_train_command(args, presets)
    print(f"Preset: {run_name}")
    run(command, dry_run=args.dry_run)


def command_download(args: argparse.Namespace, presets: Dict[str, Any]) -> None:
    task = select(presets["tasks"], args.task, "task")["dataset"]
    if args.robot:
        robot = select(presets["robots"], args.robot, "robot")
        if args.stage in ("finetune", "target-only"):
            require_target_robot(args.robot, robot)
    datasets = downloadable_datasets(args.stage, args.split, task, args.robot)
    tokenizer = tokenizer_name(args.stage, args.split, task, args.robot)
    data_root = args.data_root.expanduser().resolve()
    vq_root = args.vq_root.expanduser().resolve()

    downloads = [(f"ajaysri/{name}", "dataset", data_root / name.replace("-", "_")) for name in datasets]
    if not args.skip_tokenizer:
        suffix = "-vq-extra-action-tokenizer"
        tokenizer_dataset = tokenizer[: -len(suffix)]
        downloads.append((f"ajaysri/{tokenizer}", "model", vq_root / tokenizer_dataset.replace("-", "_")))

    for repo_id, repo_type, local_dir in downloads:
        command = ["hf", "download", repo_id, "--repo-type", repo_type, "--local-dir", str(local_dir)]
        run(command, dry_run=args.dry_run)


def command_eval(args: argparse.Namespace, presets: Dict[str, Any]) -> None:
    if args.trials <= 0:
        raise ConfigError("--trials must be positive")
    if args.max_videos < 0:
        raise ConfigError("--max-videos cannot be negative")
    command, rollout_dir = build_eval_command(args, presets)
    print(f"Rollouts: {rollout_dir}")
    if not args.dry_run:
        subprocess.run([str(EXTERN_ROOT / "validate.sh")], cwd=REPO_ROOT, check=True)
    run(command, dry_run=args.dry_run, env=runtime_env())


def mimicgen_root(args: argparse.Namespace) -> Path:
    root = args.mimicgen_root or env_path("BARX_MIMICGEN_ROOT")
    if root is None:
        root = EXTERN_ROOT / "src" / "mimicgen_xembod"
    root = root.expanduser().resolve()
    script = root / "mimicgen" / "scripts" / "generate_dataset.py"
    if not script.is_file():
        raise ConfigError(
            "MimicGen BARX fork not found. Set BARX_MIMICGEN_ROOT or pass --mimicgen-root. "
            "This dependency will move into the public bootstrap before beta testing."
        )
    return root


def command_generate(args: argparse.Namespace, presets: Dict[str, Any]) -> None:
    select(presets["environments"], args.environment, "environment")
    robot = select(presets["robots"], args.robot, "robot")
    if args.num_demos <= 0:
        raise ConfigError("--num-demos must be positive")
    args.source = args.source.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.source.is_file():
        raise ConfigError(f"Source HDF5 does not exist: {args.source}")
    validate_mimicgen_source(args.source)
    root = mimicgen_root(args)
    config = build_generation_config(args, presets)
    config_path = args.output / "configs" / f"{args.environment}_{args.robot}_seed{args.seed}.json"
    if args.dry_run:
        print(json.dumps(config, indent=2))
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    command = [sys.executable, str(root / "mimicgen" / "scripts" / "generate_dataset.py"), "--config", str(config_path)]
    print(f"Generation preset: {args.environment} -> {robot['output_name']}")
    run(command, dry_run=args.dry_run, env=runtime_env([root]))


def validate_mimicgen_source(path: Path) -> None:
    """Fail early when a raw state file has not been prepared for MimicGen."""
    try:
        import h5py
    except ImportError as exc:
        raise ConfigError("h5py is required for BARX data generation") from exc

    with h5py.File(path, "r") as dataset:
        if "data" not in dataset or not dataset["data"]:
            raise ConfigError(f"Source HDF5 has no demonstrations: {path}")
        demo_id = next(iter(dataset["data"]))
        demo = dataset["data"][demo_id]
        missing = [key for key in ("states", "actions", "datagen_info") if key not in demo]
        if missing:
            raise ConfigError(
                f"Source HDF5 is not MimicGen-ready; {demo_id} is missing {', '.join(missing)}. "
                "Teleoperation and source preparation are deferred to the next release phase."
            )


def command_render(args: argparse.Namespace, presets: Dict[str, Any]) -> None:
    robot = select(presets["robots"], args.robot, "robot")
    input_root = require_path(args.input, "--input", "BARX_RAW_ROOT")
    if args.workers <= 0:
        raise ConfigError("--workers must be positive")
    command = [
        sys.executable,
        "scripts/robocasa_x/batch_render.py",
        str(input_root),
        "--robot",
        args.robot,
        "--camera",
        robot["camera"],
        "--workers",
        str(args.workers),
    ]
    if args.max_demos is not None:
        command.extend(["--max-demos", str(args.max_demos)])
    if args.overwrite:
        command.append("--overwrite")
    run(command, dry_run=args.dry_run, env=runtime_env())


def command_build_rlds(args: argparse.Namespace, presets: Dict[str, Any]) -> None:
    robot = select(presets["robots"], args.robot, "robot")
    root = require_path(args.input, "--input", "BARX_RENDERED_ROOT")
    if args.max_files is not None and args.max_files <= 0:
        raise ConfigError("--max-files must be positive")
    pattern = str(root / "**" / "demo_gentex_im320.hdf5")
    command = [
        sys.executable,
        "scripts/robocasa_x/build_rlds.py",
        "--input-glob",
        pattern,
        "--action-space",
        robot["action_space"],
        "--source-root",
        str(root),
        "--output-dir",
        str(args.output.expanduser().resolve()),
    ]
    if args.max_files is not None:
        command.extend(["--max-files", str(args.max_files)])
    if args.no_wrist:
        command.append("--no-wrist")
    run(command, dry_run=args.dry_run)


def command_setup(args: argparse.Namespace, presets: Dict[str, Any]) -> None:
    del presets
    run([str(EXTERN_ROOT / "bootstrap.sh")], dry_run=args.dry_run)
    run([str(EXTERN_ROOT / "validate.sh")], dry_run=args.dry_run)


def add_common_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", choices=("pretrain", "finetune", "target-only"), required=True)
    parser.add_argument("--split", choices=("xp900", "xp3k", "sp900"), required=True)
    parser.add_argument("--task", choices=("pnp", "turn-on-sink-faucet", "flip-mug-upright"), required=True)
    parser.add_argument("--representation", default="joint-reps")
    parser.add_argument("--robot", choices=("panda", "panda-og", "jaco"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--base-vlm", type=Path, default=env_path("BARX_BASE_VLM"))
    parser.add_argument("--data-root", type=Path, default=env_path("BARX_DATA_ROOT"))
    parser.add_argument("--run-root", type=Path, default=env_path("BARX_RUN_ROOT", REPO_ROOT / "runs"))
    parser.add_argument("--gpus", type=int, default=int(os.environ.get("NUM_GPUS", "8")))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-project", default="barx")
    parser.add_argument("--hf-token-env", default="HF_TOKEN" if os.environ.get("HF_TOKEN") else None)
    parser.add_argument("--dry-run", action="store_true")


def build_parser(presets: Dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="barx", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preset_parser = subparsers.add_parser("presets", help="List supported choices.")
    preset_parser.set_defaults(handler=command_presets)

    download_parser = subparsers.add_parser("download", help="Download datasets and VQ assets for one preset.")
    download_parser.add_argument("--stage", choices=("pretrain", "finetune", "target-only"), required=True)
    download_parser.add_argument("--split", choices=("xp900", "xp3k", "sp900"), required=True)
    download_parser.add_argument("--task", choices=tuple(presets["tasks"]), required=True)
    download_parser.add_argument("--robot", choices=("panda", "panda-og", "jaco"))
    download_parser.add_argument("--data-root", type=Path, default=env_path("BARX_DATA_ROOT", REPO_ROOT / "data"))
    download_parser.add_argument("--vq-root", type=Path, default=env_path("BARX_VQ_ROOT", REPO_ROOT / "vq"))
    download_parser.add_argument("--skip-tokenizer", action="store_true")
    download_parser.add_argument("--dry-run", action="store_true")
    download_parser.set_defaults(handler=command_download)

    train_parser = subparsers.add_parser("train", help="Train from a validated paper preset.")
    add_common_train_arguments(train_parser)
    train_parser.set_defaults(handler=command_train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a checkpoint in one environment/robot setting.")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--environment", choices=tuple(presets["environments"]), required=True)
    eval_parser.add_argument("--robot", choices=tuple(presets["robots"]), required=True)
    eval_parser.add_argument("--trials", type=int, default=presets["evaluation"]["trials"])
    eval_parser.add_argument("--start-seed", type=int, default=presets["evaluation"]["start_seed"])
    eval_parser.add_argument("--unnorm-key")
    eval_parser.add_argument("--model-tag")
    eval_parser.add_argument("--output", type=Path)
    eval_parser.add_argument("--run-root", type=Path, default=REPO_ROOT / "experiments" / "rollouts")
    eval_parser.add_argument("--max-videos", type=int, default=presets["evaluation"]["max_videos"])
    eval_parser.add_argument("--no-videos", action="store_true")
    eval_parser.add_argument("--wandb", action="store_true")
    eval_parser.add_argument("--hf-token-env", default="HF_TOKEN" if os.environ.get("HF_TOKEN") else None)
    eval_parser.add_argument("--dry-run", action="store_true")
    eval_parser.set_defaults(handler=command_eval)

    generate_parser = subparsers.add_parser("generate", help="Generate raw state trajectories with MimicGen.")
    generate_parser.add_argument("--source", type=Path, required=True)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--environment", choices=tuple(presets["environments"]), required=True)
    generate_parser.add_argument("--robot", choices=tuple(presets["robots"]), required=True)
    generate_parser.add_argument("--num-demos", type=int, default=presets["generation"]["num_demos"])
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument("--mimicgen-root", type=Path)
    generate_parser.add_argument("--dry-run", action="store_true")
    generate_parser.set_defaults(handler=command_generate)

    render_parser = subparsers.add_parser("render", help="Replay raw states into BARX camera observations.")
    render_parser.add_argument("--input", type=Path, required=True)
    render_parser.add_argument("--robot", choices=tuple(presets["robots"]), required=True)
    render_parser.add_argument("--workers", type=int, default=10)
    render_parser.add_argument("--max-demos", type=int)
    render_parser.add_argument("--overwrite", action="store_true")
    render_parser.add_argument("--dry-run", action="store_true")
    render_parser.set_defaults(handler=command_render)

    rlds_parser = subparsers.add_parser("build-rlds", help="Convert rendered HDF5 into BARX RLDS.")
    rlds_parser.add_argument("--input", type=Path, required=True)
    rlds_parser.add_argument("--output", type=Path, required=True)
    rlds_parser.add_argument("--robot", choices=tuple(presets["robots"]), required=True)
    rlds_parser.add_argument("--max-files", type=int)
    rlds_parser.add_argument("--no-wrist", action="store_true")
    rlds_parser.add_argument("--dry-run", action="store_true")
    rlds_parser.set_defaults(handler=command_build_rlds)

    setup_parser = subparsers.add_parser("setup-sim", help="Install the pinned RoboCasa simulation stack.")
    setup_parser.add_argument("--dry-run", action="store_true")
    setup_parser.set_defaults(handler=command_setup)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    presets = load_presets()
    parser = build_parser(presets)
    args = parser.parse_args(argv)
    try:
        args.handler(args, presets)
    except (ConfigError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())

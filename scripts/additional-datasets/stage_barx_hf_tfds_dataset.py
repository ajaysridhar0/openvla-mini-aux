#!/usr/bin/env python3
"""
Stage a TFDS dataset under a new builder name and optionally upload it to Hugging Face.

This is intended for RoboCasa-X release packaging where the public HF repo name and the
underlying TFDS dataset builder name should be paper-facing, while preserving the
original dataset contents.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Path to the source TFDS dataset directory.")
    parser.add_argument(
        "--target-name",
        type=str,
        required=True,
        help="New TFDS builder name to stage, e.g. `robocasa_x_xp3k_pnp`.",
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        required=True,
        help="Root directory to create the staged dataset under.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Optional HF dataset repo id to create/upload, e.g. `ajaysri/robocasa-x-xp3k-pnp`.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="If set, create the HF dataset repo if needed and upload the staged folder.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the HF repo as private when used with --upload.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete any existing staged dataset directory before rebuilding it.",
    )
    parser.add_argument(
        "--copy-small-files",
        action="store_true",
        help="Copy metadata files instead of hardlinking them. Shards are still hardlinked when possible.",
    )
    parser.add_argument(
        "--upload-workers",
        type=int,
        default=8,
        help="Number of upload workers for HF upload_large_folder.",
    )
    return parser.parse_args()


def ensure_valid_target_name(target_name: str) -> None:
    if "-" in target_name:
        raise ValueError(
            f"Target TFDS name `{target_name}` contains '-'. TFDS builder names must use underscores instead."
        )
    if not target_name.replace("_", "").isalnum():
        raise ValueError(f"Target TFDS name `{target_name}` must be alphanumeric plus underscores.")


def list_versions(source_dir: Path) -> Iterable[Path]:
    versions = sorted(path for path in source_dir.iterdir() if path.is_dir())
    if not versions:
        raise FileNotFoundError(f"No version directories found under {source_dir}")
    return versions


def rewrite_dataset_info(source_path: Path, destination_path: Path, target_name: str) -> None:
    with source_path.open("r") as handle:
        dataset_info = json.load(handle)
    dataset_info["name"] = target_name
    with destination_path.open("w") as handle:
        json.dump(dataset_info, handle, indent=2)
        handle.write("\n")


def shard_destination_name(source_name: str, source_dataset_name: str, target_name: str) -> str:
    prefix = f"{source_dataset_name}-"
    if source_name.startswith(prefix):
        return f"{target_name}-{source_name[len(prefix):]}"
    return source_name


def link_or_copy(source_path: Path, destination_path: Path, *, copy_file: bool = False) -> Tuple[str, Path]:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if copy_file:
        shutil.copy2(source_path, destination_path)
        return "copied", destination_path

    try:
        os.link(source_path, destination_path)
        return "hardlinked", destination_path
    except OSError:
        shutil.copy2(source_path, destination_path)
        return "copied", destination_path


ROBOCASA_X_TASK_METADATA: Dict[str, Dict[str, str]] = {
    "pnp": {
        "title": "PnP Counter to Sink / PnP Sink to Counter",
        "robots": "Panda, Panda-OG, Jaco",
        "xp3k_demos": "6000",
        "xp900_demos": "1800",
        "sp900_demos": "1800",
        "target_demos": "100",
    },
    "turn_on_sink_faucet": {
        "title": "Turn On Sink Faucet",
        "robots": "Panda, Panda-OG, Jaco",
        "xp3k_demos": "3000",
        "xp900_demos": "900",
        "sp900_demos": "900",
        "target_demos": "50",
    },
    "flip_mug_upright": {
        "title": "Flip Mug Upright",
        "robots": "Panda, Panda-OG, Jaco",
        "xp3k_demos": "3000",
        "xp900_demos": "900",
        "sp900_demos": "900",
        "target_demos": "50",
    },
}

ROBOCASA_X_ROBOT_LABELS: Dict[str, str] = {
    "panda": "Panda",
    "panda_og": "Panda-OG",
    "jaco": "Jaco",
}


def infer_robocasa_x_release_metadata(target_name: str) -> Dict[str, str]:
    if target_name.startswith("robocasa_x_xp3k_"):
        task_key = target_name[len("robocasa_x_xp3k_") :]
        return {
            "title_prefix": "RoboCasa-X XP-3K",
            "split_label": "XP-3K",
            "task_key": task_key,
            "robots": ROBOCASA_X_TASK_METADATA[task_key]["robots"],
            "num_demos": ROBOCASA_X_TASK_METADATA[task_key]["xp3k_demos"],
        }

    if target_name.startswith("robocasa_x_xp900_"):
        task_key = target_name[len("robocasa_x_xp900_") :]
        return {
            "title_prefix": "RoboCasa-X XP-900",
            "split_label": "XP-900",
            "task_key": task_key,
            "robots": ROBOCASA_X_TASK_METADATA[task_key]["robots"],
            "num_demos": ROBOCASA_X_TASK_METADATA[task_key]["xp900_demos"],
        }

    if target_name.startswith("robocasa_x_sp900_"):
        remainder = target_name[len("robocasa_x_sp900_") :]
        for robot_key in ("panda_og", "panda", "jaco"):
            robot_prefix = f"{robot_key}_"
            if remainder.startswith(robot_prefix):
                task_key = remainder[len(robot_prefix) :]
                return {
                    "title_prefix": f"RoboCasa-X SP-900 {ROBOCASA_X_ROBOT_LABELS[robot_key]}",
                    "split_label": "SP-900",
                    "task_key": task_key,
                    "robots": ROBOCASA_X_ROBOT_LABELS[robot_key],
                    "num_demos": ROBOCASA_X_TASK_METADATA[task_key]["sp900_demos"],
                }

    if target_name.startswith("robocasa_x_target_"):
        remainder = target_name[len("robocasa_x_target_") :]
        for robot_key in ("panda_og", "panda", "jaco"):
            robot_prefix = f"{robot_key}_"
            if remainder.startswith(robot_prefix):
                task_key = remainder[len(robot_prefix) :]
                return {
                    "title_prefix": f"RoboCasa-X Target {ROBOCASA_X_ROBOT_LABELS[robot_key]}",
                    "split_label": "Target",
                    "task_key": task_key,
                    "robots": ROBOCASA_X_ROBOT_LABELS[robot_key],
                    "num_demos": ROBOCASA_X_TASK_METADATA[task_key]["target_demos"],
                }

    return {
        "title_prefix": "RoboCasa-X Dataset",
        "split_label": "RoboCasa-X",
        "task_key": "",
        "robots": "",
        "num_demos": "",
    }


def build_dataset_card(target_name: str) -> str:
    metadata = infer_robocasa_x_release_metadata(target_name)
    task_metadata = ROBOCASA_X_TASK_METADATA.get(metadata["task_key"], {})
    task_title = task_metadata.get("title", target_name.replace("_", " "))
    title = f"{metadata['title_prefix']} {task_title}".strip()

    return "\n".join(
        [
            "---",
            f"pretty_name: {title}",
            "tags:",
            "  - robotics",
            "  - robocasa-x",
            "  - tensorflow-datasets",
            "---",
            "",
            f"# {title}",
            "",
            f"- Task: `{task_title}`",
            f"- TFDS builder id after download: `{target_name}`",
            f"- Robots in dataset: `{metadata['robots']}`",
            f"- Number of demos: `{metadata['num_demos']}`",
            "",
        ]
    )


def stage_dataset(source_dir: Path, target_name: str, stage_root: Path, overwrite: bool, copy_small_files: bool) -> Path:
    source_dir = source_dir.resolve()
    source_dataset_name = source_dir.name
    staged_dataset_dir = stage_root.resolve() / target_name

    if staged_dataset_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Staged dataset dir already exists: {staged_dataset_dir}. Pass --overwrite to rebuild it."
            )
        shutil.rmtree(staged_dataset_dir)

    staged_dataset_dir.mkdir(parents=True, exist_ok=True)

    for version_dir in list_versions(source_dir):
        staged_version_dir = staged_dataset_dir / version_dir.name
        staged_version_dir.mkdir(parents=True, exist_ok=True)

        for file_path in sorted(version_dir.iterdir()):
            if file_path.name == "dataset_info.json":
                rewrite_dataset_info(file_path, staged_version_dir / "dataset_info.json", target_name)
            elif file_path.name == "features.json":
                link_or_copy(file_path, staged_version_dir / "features.json", copy_file=copy_small_files)
            else:
                destination_name = shard_destination_name(file_path.name, source_dataset_name, target_name)
                link_or_copy(file_path, staged_version_dir / destination_name)

    readme_path = staged_dataset_dir / "README.md"
    readme_path.write_text(build_dataset_card(target_name))
    gitattributes_path = staged_dataset_dir / ".gitattributes"
    gitattributes_path.write_text("*.tfrecord-* filter=lfs diff=lfs merge=lfs -text\n")

    return staged_dataset_dir


def upload_to_hf(staged_dataset_dir: Path, repo_id: str, private: bool, num_workers: int) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=staged_dataset_dir,
        num_workers=num_workers,
    )


def main() -> None:
    args = parse_args()
    ensure_valid_target_name(args.target_name)

    staged_dataset_dir = stage_dataset(
        source_dir=args.source_dir,
        target_name=args.target_name,
        stage_root=args.stage_root,
        overwrite=args.overwrite,
        copy_small_files=args.copy_small_files,
    )
    print(f"Staged dataset at {staged_dataset_dir}")

    if args.upload:
        if args.repo_id is None:
            raise ValueError("--repo-id is required when using --upload")
        upload_to_hf(
            staged_dataset_dir=staged_dataset_dir,
            repo_id=args.repo_id,
            private=args.private,
            num_workers=args.upload_workers,
        )
        print(f"Uploaded staged dataset to {args.repo_id}")


if __name__ == "__main__":
    main()

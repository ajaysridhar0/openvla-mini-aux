"""Portable BARX MimicGen configuration and evidence helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "mimicgen.json"


def registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text())


def task_names() -> tuple[str, ...]:
    return tuple(registry()["tasks"])


def embodiment_names() -> tuple[str, ...]:
    return tuple(registry()["embodiments"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_inventory(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as dataset:
        data = dataset.get("data")
        if not isinstance(data, h5py.Group):
            raise ValueError(f"{path} does not contain an HDF5 data group")
        demos = [
            (name, demo)
            for name, demo in sorted(data.items())
            if isinstance(demo, h5py.Group) and "actions" in demo
        ]
        prepared = [
            name for name, demo in demos if isinstance(demo.get("datagen_info"), h5py.Group)
        ]
        action_widths = sorted({int(demo["actions"].shape[-1]) for _, demo in demos})
        return {
            "demonstrations": len(demos),
            "prepared_demonstrations": len(prepared),
            "action_widths": action_widths,
            "environment": json.loads(data.attrs["env_args"])["env_name"],
        }


def resolved_config(
    *,
    source: Path,
    task: str,
    embodiment: str,
    output_dir: Path,
    seed: int,
    successes: int,
    max_attempts: int,
    source_demos: int,
) -> dict[str, Any]:
    release = registry()
    try:
        task_config = release["tasks"][task]
    except KeyError as exc:
        raise ValueError(f"Unknown MimicGen task: {task}") from exc
    try:
        embodiment_config = release["embodiments"][embodiment]
    except KeyError as exc:
        raise ValueError(f"Unknown MimicGen embodiment: {embodiment}") from exc
    if successes < 1 or max_attempts < successes or source_demos < 1:
        raise ValueError(
            "successes and source_demos must be positive, and max_attempts "
            "must be at least successes"
        )
    return {
        "name": task_config["config_name"],
        "type": "robosuite",
        "experiment": {
            "source": {
                "dataset_path": str(source.resolve()),
                "n": source_demos,
            },
            "generation": {
                "path": str(output_dir.resolve()),
                "select_src_per_subtask": True,
                "num_trials": successes,
                "guarantee": True,
                "keep_failed": True,
                "max_attempts": max_attempts,
                "transform_first_robot_pose": True,
            },
            "task": {
                "robot": embodiment_config["robot"],
                "gripper": embodiment_config["gripper"],
                "interface": task_config["interface"],
                "interface_type": "robosuite",
            },
            "render_video": False,
            "num_demo_to_render": 1,
            "num_fail_demo_to_render": 1,
            "seed": seed,
        },
        "task": {
            "task_spec": {
                "stage_1": {},
                "stage_2": {},
            }
        },
    }


def generated_inventory(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as dataset:
        data = dataset.get("data")
        if not isinstance(data, h5py.Group):
            raise ValueError(f"{path} does not contain an HDF5 data group")
        demos = [
            (name, demo)
            for name, demo in sorted(data.items())
            if isinstance(demo, h5py.Group) and "actions" in demo
        ]
        return {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "demonstrations": len(demos),
            "action_widths": sorted(
                {int(demo["actions"].shape[-1]) for _, demo in demos}
            ),
            "state_shapes": sorted(
                {tuple(int(size) for size in demo["states"].shape) for _, demo in demos}
            ),
            "environment": json.loads(data.attrs["env_args"])["env_name"],
            "action_layout": data.attrs.get("barx_action_layout"),
        }

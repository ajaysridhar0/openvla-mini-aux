#!/usr/bin/env python3
"""Copy BARX HDF5s into a verified, portable public-release tree."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from barx.evaluation_conditions import portable_ep_meta

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "dataset" / "manifest.csv"
LAYOUT_ATTRIBUTE = "barx_action_layout"
NORMALIZED_LAYOUT = "arm_gripper_base_torso_mode_v1"
BODY_PART_ORDER = ["right", "right_gripper", "base", "torso"]
LEGACY_NON_PANDA_EMBODIMENTS = {"IIWAOmron", "JacoOmron", "Kinova3Omron", "UR5eOmron"}
PANDA_GRIPPERS = {
    "PandaOmron": "Robotiq85Gripper",
    "PandaOGGripperOmron": "PandaGripper",
    "PandaOGOmron": "PandaGripper",
}
ENVIRONMENT_ALIASES = {
    "PnPCounterToSink": "XPnPCounterToSink",
    "PnPSinkToCounter": "XPnPSinkToCounter",
    "TurnOnSinkFaucet": "XTurnOnSinkFaucet",
    "FlipMugUpright": "XFlipMugUpright",
}
LEGACY_TO_NORMALIZED = [0, 1, 2, 3, 4, 5, 10, 6, 7, 8, 9, 11]
PRIVATE_METADATA_MARKERS = ("/iliad", "/sailhome", "/home/", "/users/")


def internal_embodiment(relative_path: Path) -> str:
    if len(relative_path.parts) < 2:
        raise ValueError(f"Unexpected release path: {relative_path}")
    return relative_path.parts[1]


def normalize_env_args(raw: str | bytes, embodiment: str) -> str:
    env_args = json.loads(raw)
    env_kwargs = env_args.get("env_kwargs")
    if not isinstance(env_kwargs, dict):
        raise ValueError("env_args must contain an env_kwargs object")
    controller = env_kwargs.get("controller_configs")
    if not isinstance(controller, dict):
        raise ValueError("env_kwargs must contain controller_configs")
    specific = controller.setdefault("composite_controller_specific_configs", {})
    specific["body_part_ordering"] = BODY_PART_ORDER
    if embodiment in PANDA_GRIPPERS:
        env_kwargs["gripper_types"] = PANDA_GRIPPERS[embodiment]
    if env_args.get("env_name") in ENVIRONMENT_ALIASES:
        env_args["env_name"] = ENVIRONMENT_ALIASES[env_args["env_name"]]
    return json.dumps(env_args, separators=(",", ":"), sort_keys=True)


def normalize_demo_metadata(data: h5py.Group) -> None:
    for demo in data.values():
        if not isinstance(demo, h5py.Group):
            continue
        if "ep_meta" in demo.attrs:
            value = portable_ep_meta(json.loads(demo.attrs["ep_meta"]))
            demo.attrs["ep_meta"] = json.dumps(value, separators=(",", ":"), sort_keys=True)
        if "model_file" in demo.attrs:
            del demo.attrs["model_file"]


def attribute_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return " ".join(attribute_text(item) for item in value.reshape(-1))
    return str(value)


def private_attribute_hits(file: h5py.File) -> list[str]:
    hits: list[str] = []

    def inspect(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        for key, value in obj.attrs.items():
            if any(marker in attribute_text(value).lower() for marker in PRIVATE_METADATA_MARKERS):
                hits.append(f"{name or '/'}:{key}")

    inspect("", file)
    data = file.get("data")
    if isinstance(data, h5py.Group):
        inspect("data", data)
        for name, demo in data.items():
            if isinstance(demo, h5py.Group):
                inspect(f"data/{name}", demo)
    return hits


def normalize_hdf5(path: Path, embodiment: str) -> int:
    """Normalize an unmarked copied file; never call this on a marked source."""
    with h5py.File(path, "r+") as output:
        data = output.get("data")
        if not isinstance(data, h5py.Group) or "env_args" not in data.attrs:
            raise ValueError(f"{path} lacks data/env_args")
        if LAYOUT_ATTRIBUTE in data.attrs:
            raise ValueError(f"{path} already has an action-layout marker")
        demos = 0
        for name, demo in data.items():
            if not isinstance(demo, h5py.Group) or "actions" not in demo:
                continue
            actions = demo["actions"]
            if actions.ndim < 1 or actions.shape[-1] != 12:
                raise ValueError(f"{path}:{name}/actions has shape {actions.shape}, expected (..., 12)")
            if embodiment in LEGACY_NON_PANDA_EMBODIMENTS:
                actions[...] = actions[...][..., LEGACY_TO_NORMALIZED]
            demos += 1
        if not demos:
            raise ValueError(f"{path} contains no demonstrations")
        normalize_demo_metadata(data)
        data.attrs["env_args"] = normalize_env_args(data.attrs["env_args"], embodiment)
        data.attrs[LAYOUT_ATTRIBUTE] = NORMALIZED_LAYOUT
        output.flush()
    return demos


def normalize_existing_hdf5(path: Path, embodiment: str) -> None:
    """Normalize metadata idempotently without permuting already-normalized actions."""
    with h5py.File(path, "r+") as output:
        data = output.get("data")
        if not isinstance(data, h5py.Group) or data.attrs.get(LAYOUT_ATTRIBUTE) != NORMALIZED_LAYOUT:
            raise ValueError(f"{path} is not marked with the normalized BARX layout")
        normalize_demo_metadata(data)
        data.attrs["env_args"] = normalize_env_args(data.attrs["env_args"], embodiment)
        output.flush()


def verify_hdf5(path: Path, expected_demos: int) -> None:
    with h5py.File(path, "r") as staged:
        data = staged.get("data")
        if not isinstance(data, h5py.Group) or data.attrs.get(LAYOUT_ATTRIBUTE) != NORMALIZED_LAYOUT:
            raise ValueError(f"{path} is not marked with the normalized BARX layout")
        demos = [demo for demo in data.values() if isinstance(demo, h5py.Group) and "actions" in demo]
        if len(demos) != expected_demos:
            raise ValueError(f"{path} contains {len(demos)} demos; expected {expected_demos}")
        if any(demo["actions"].shape[-1] != 12 for demo in demos):
            raise ValueError(f"{path} contains a non-12-D action array")
        if json.loads(data.attrs["env_args"]).get("env_name") not in ENVIRONMENT_ALIASES.values():
            raise ValueError(f"{path} has an unsupported environment")
        if hits := private_attribute_hits(staged):
            raise ValueError(f"{path} exposes private paths in attributes: {', '.join(hits[:5])}")


def stage_file(source: Path, destination: Path, embodiment: str, expected_demos: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        normalize_existing_hdf5(destination, embodiment)
        verify_hdf5(destination, expected_demos)
        return
    temporary = destination.with_name(f".{destination.name}.staging")
    try:
        shutil.copy2(source, temporary)
        with h5py.File(temporary, "r") as copied:
            data = copied.get("data")
            marker = data.attrs.get(LAYOUT_ATTRIBUTE) if isinstance(data, h5py.Group) else None
        if marker is None:
            demos = normalize_hdf5(temporary, embodiment)
            if demos != expected_demos:
                raise ValueError(f"{source} contains {demos} demos; manifest says {expected_demos}")
        elif marker == NORMALIZED_LAYOUT:
            # The archival tree may already have normalized actions. Reordering
            # again would silently corrupt it; only scrub metadata in this case.
            normalize_existing_hdf5(temporary, embodiment)
        else:
            raise ValueError(f"{source} has unknown action-layout marker {marker!r}")
        verify_hdf5(temporary, expected_demos)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def select_manifest_shard(rows, *, shard_count: int, shard_index: int, limit: int | None):
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid manifest shard")
    selected = rows[shard_index::shard_count]
    return selected if limit is None else selected[:limit]


def stage_manifest_row(row: dict[str, str], source_root: Path, output_root: Path) -> None:
    relative = Path(row["relative_path"])
    source = source_root / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    stage_file(source, output_root / relative, internal_embodiment(relative), int(row["demonstrations"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    source_root, output_root = args.source_root.resolve(), args.output_root.resolve()
    if source_root == output_root:
        parser.error("source_root and output_root must differ; source data is never modified")
    if args.workers < 1:
        parser.error("--workers must be positive")
    rows = select_manifest_shard(
        manifest_rows(args.manifest),
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        limit=args.limit,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(lambda row: stage_manifest_row(row, source_root, output_root), rows))
    print(f"Staged and verified {len(rows)} files")


if __name__ == "__main__":
    main()

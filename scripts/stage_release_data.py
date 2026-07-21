#!/usr/bin/env python3
"""Copy the selected BARX data and normalize it for public release.

Source files are never modified. Each destination is written through a
temporary file and atomically renamed only after its action arrays and
environment metadata have been validated.
"""

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
PRIVATE_METADATA_MARKERS = ("/iliad", "/sailhome", "/home/")


def internal_embodiment(relative_path: Path) -> str:
    if len(relative_path.parts) < 2:
        raise ValueError(f"Unexpected release path: {relative_path}")
    return relative_path.parts[1]


def normalize_env_args(raw_env_args: str | bytes, embodiment: str) -> str:
    """Return playback metadata for the normalized controller layout."""
    env_args = json.loads(raw_env_args)
    env_kwargs = env_args.get("env_kwargs")
    if not isinstance(env_kwargs, dict):
        raise ValueError("env_args must contain an env_kwargs object")

    controller = env_kwargs.get("controller_configs")
    if not isinstance(controller, dict):
        raise ValueError("env_kwargs must contain a controller_configs object")
    specific = controller.setdefault("composite_controller_specific_configs", {})
    specific["body_part_ordering"] = BODY_PART_ORDER

    if embodiment in PANDA_GRIPPERS:
        env_kwargs["gripper_types"] = PANDA_GRIPPERS[embodiment]
    stored_environment = env_args.get("env_name")
    if stored_environment in ENVIRONMENT_ALIASES:
        env_args["env_name"] = ENVIRONMENT_ALIASES[stored_environment]
    return json.dumps(env_args, separators=(",", ":"), sort_keys=True)


def normalize_demo_metadata(data: h5py.Group) -> None:
    """Keep converter metadata portable and discard unused replay-only XML."""

    for demo_name, demo in data.items():
        if not isinstance(demo, h5py.Group):
            continue
        if "ep_meta" in demo.attrs:
            original_ep_meta = demo.attrs["ep_meta"]
            ep_meta = json.loads(original_ep_meta)
            normalized_ep_meta = json.dumps(
                portable_ep_meta(ep_meta), separators=(",", ":"), sort_keys=True
            )
            if normalized_ep_meta != original_ep_meta:
                demo.attrs["ep_meta"] = normalized_ep_meta
        if "model_file" in demo.attrs:
            del demo.attrs["model_file"]


def attribute_text(value: object) -> str:
    """Return a searchable text representation of an HDF5 attribute."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return " ".join(attribute_text(item) for item in value.reshape(-1))
    return str(value)


def private_attribute_hits(file: h5py.File) -> list[str]:
    """Return file, data, or demo attributes with collection-machine paths."""

    hits: list[str] = []

    def inspect(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        for key, value in obj.attrs.items():
            text = attribute_text(value).lower()
            if any(marker in text for marker in PRIVATE_METADATA_MARKERS):
                hits.append(f"{name or '/'}:{key}")

    inspect("", file)
    data = file.get("data")
    if isinstance(data, h5py.Group):
        inspect("data", data)
        for demo_name, demo in data.items():
            if isinstance(demo, h5py.Group):
                inspect(f"data/{demo_name}", demo)
    return hits


def normalize_hdf5(path: Path, embodiment: str) -> int:
    """Normalize one copied HDF5 file and return its demonstration count."""
    with h5py.File(path, "r+") as output:
        data = output.get("data")
        if not isinstance(data, h5py.Group):
            raise ValueError(f"{path} does not contain a data group")
        if LAYOUT_ATTRIBUTE in data.attrs:
            raise ValueError(
                f"{path} is already marked as {data.attrs[LAYOUT_ATTRIBUTE]!r}"
            )
        if "env_args" not in data.attrs:
            raise ValueError(f"{path} does not contain data.attrs['env_args']")

        demo_count = 0
        for demo_name, demo in data.items():
            if not isinstance(demo, h5py.Group) or "actions" not in demo:
                continue
            actions = demo["actions"]
            if actions.ndim < 1 or actions.shape[-1] != 12:
                raise ValueError(
                    f"{path}:{demo_name}/actions has shape {actions.shape}, expected (..., 12)"
                )
            if embodiment in LEGACY_NON_PANDA_EMBODIMENTS:
                original = actions[...]
                normalized = original[..., LEGACY_TO_NORMALIZED]
                actions[...] = normalized
                if not np.array_equal(actions[...], normalized, equal_nan=True):
                    raise ValueError(
                        f"Failed to verify normalized actions in {path}:{demo_name}"
                    )
            demo_count += 1

        if demo_count == 0:
            raise ValueError(f"{path} contains no demonstrations")
        normalize_demo_metadata(data)
        data.attrs["env_args"] = normalize_env_args(data.attrs["env_args"], embodiment)
        data.attrs[LAYOUT_ATTRIBUTE] = NORMALIZED_LAYOUT
        output.flush()
    return demo_count


def verify_hdf5(path: Path, expected_demos: int) -> None:
    with h5py.File(path, "r") as staged:
        data = staged.get("data")
        if (
            not isinstance(data, h5py.Group)
            or data.attrs.get(LAYOUT_ATTRIBUTE) != NORMALIZED_LAYOUT
        ):
            raise ValueError(
                f"{path} is not marked with the normalized BARX action layout"
            )
        demos = [
            demo
            for demo in data.values()
            if isinstance(demo, h5py.Group) and "actions" in demo
        ]
        if len(demos) != expected_demos:
            raise ValueError(
                f"{path} contains {len(demos)} demos; expected {expected_demos}"
            )
        if any(demo["actions"].shape[-1] != 12 for demo in demos):
            raise ValueError(f"{path} contains a non-12-D action array")
        env_args = json.loads(data.attrs["env_args"])
        environment = env_args.get("env_name")
        if environment not in ENVIRONMENT_ALIASES.values():
            raise ValueError(f"{path} has unsupported environment name {environment!r}")
        private_hits = private_attribute_hits(staged)
        if private_hits:
            preview = ", ".join(private_hits[:5])
            raise ValueError(f"{path} exposes private paths in attributes: {preview}")


def normalize_existing_hdf5(path: Path, embodiment: str) -> None:
    """Apply idempotent metadata normalization to an existing staged file."""

    with h5py.File(path, "r+") as output:
        data = output.get("data")
        if (
            not isinstance(data, h5py.Group)
            or data.attrs.get(LAYOUT_ATTRIBUTE) != NORMALIZED_LAYOUT
        ):
            raise ValueError(f"{path} is not an existing normalized BARX file")
        normalize_demo_metadata(data)
        data.attrs["env_args"] = normalize_env_args(data.attrs["env_args"], embodiment)
        output.flush()


def stage_file(
    source: Path, destination: Path, embodiment: str, expected_demos: int
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        normalize_existing_hdf5(destination, embodiment)
        verify_hdf5(destination, expected_demos)
        print(f"verified {destination}")
        return

    temporary = destination.with_name(f".{destination.name}.staging")
    try:
        shutil.copy2(source, temporary)
        demo_count = normalize_hdf5(temporary, embodiment)
        if demo_count != expected_demos:
            raise ValueError(
                f"{source} contains {demo_count} demos; manifest says {expected_demos}"
            )
        verify_hdf5(temporary, expected_demos)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"staged {destination}")


def manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def select_manifest_shard(
    rows: list[dict[str, str]],
    *,
    shard_count: int,
    shard_index: int,
    limit: int | None,
) -> list[dict[str, str]]:
    """Select one deterministic, non-overlapping slice of the manifest."""

    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    selected = rows[shard_index::shard_count]
    return selected if limit is None else selected[:limit]


def stage_manifest_row(
    row: dict[str, str], source_root: Path, output_root: Path
) -> None:
    relative_path = Path(row["relative_path"])
    source = source_root / relative_path
    if not source.is_file():
        raise FileNotFoundError(source)
    stage_file(
        source,
        output_root / relative_path,
        internal_embodiment(relative_path),
        int(row["demonstrations"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_root",
        type=Path,
        help="Original directory containing the mg/ and human/ trees",
    )
    parser.add_argument(
        "output_root", type=Path, help="New directory for normalized release copies"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--limit", type=int, help="Stage only the first N files for a dry run"
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Split the manifest into this many non-overlapping shards",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard to process (default: 0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Files to stage concurrently (default: 4)",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if source_root == output_root:
        parser.error(
            "source_root and output_root must be different; source data is never modified"
        )
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.shard_count < 1:
        parser.error("--shard-count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")

    rows = select_manifest_shard(
        manifest_rows(args.manifest),
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        limit=args.limit,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(
            executor.map(
                lambda row: stage_manifest_row(row, source_root, output_root), rows
            )
        )

    print(f"Staged and verified {len(rows)} files in {output_root}")


if __name__ == "__main__":
    main()

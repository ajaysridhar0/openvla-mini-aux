#!/usr/bin/env python3
"""Build the BARX simulation-data release manifest from a RoboCasa-X data tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


STORED_TASK_NAMES = {
    "PnPCounterToSink": "PnP Counter to Sink",
    "PnPSinkToCounter": "PnP Sink to Counter",
    "TurnOnSinkFaucet": "Turn On Sink Faucet",
    "FlipMugUpright": "Flip Mug Upright",
}
EMBODIMENT_NAMES = {
    "IIWAOmron": "IIWA",
    "Kinova3Omron": "Kinova 3",
    "UR5eOmron": "UR5e",
    "PandaOmron": "Panda",
    "PandaOGGripperOmron": "Panda-OG",
    "PandaOGOmron": "Panda-OG",
    "JacoOmron": "Jaco",
}
SOURCE_EMBODIMENTS = {"IIWAOmron", "Kinova3Omron", "UR5eOmron"}
TARGET_EMBODIMENTS = {"PandaOmron", "PandaOGGripperOmron", "PandaOGOmron", "JacoOmron"}
SEED_PATTERN = re.compile(r"100demos_seed(\d+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_for_file(path: Path, data_root: Path) -> dict[str, object] | None:
    relative = path.relative_to(data_root)
    parts = relative.parts
    if (
        len(parts) < 4
        or parts[1] not in EMBODIMENT_NAMES
        or parts[2] not in STORED_TASK_NAMES
    ):
        return None

    partition, internal_embodiment, internal_task = parts[:3]
    seed = ""
    if partition == "mg":
        if path.name != "demo_gentex_im320_randcams.hdf5":
            return None
        match = SEED_PATTERN.match(path.parent.name)
        if match is None:
            return None
        seed = int(match.group(1))
        if internal_embodiment in SOURCE_EMBODIMENTS:
            release_group = "cross-embodiment prior"
            paper_sets = "XP-3K"
            if seed % 10 <= 2:
                paper_sets += ";XP-900"
        elif internal_embodiment in TARGET_EMBODIMENTS:
            release_group = "same-embodiment prior"
            paper_sets = "SP-900"
        else:
            return None
    elif partition == "human":
        if (
            internal_embodiment not in TARGET_EMBODIMENTS
            or path.name != "demo_gentex_im320.hdf5"
        ):
            return None
        release_group = "target demonstrations"
        paper_sets = "target-50"
    else:
        return None

    return {
        "release_group": release_group,
        "paper_sets": paper_sets,
        "embodiment": EMBODIMENT_NAMES[internal_embodiment],
        "task": STORED_TASK_NAMES[internal_task],
        "demonstrations": 100 if partition == "mg" else 50,
        "seed": seed,
        "bytes": path.stat().st_size,
        "sha256": "",
        "relative_path": relative.as_posix(),
    }


def build_manifest(data_root: Path) -> list[dict[str, object]]:
    candidates = list((data_root / "mg").rglob("*.hdf5")) + list(
        (data_root / "human").rglob("*.hdf5")
    )
    rows = [
        row for path in candidates if (row := row_for_file(path, data_root)) is not None
    ]
    rows.sort(
        key=lambda row: (
            str(row["release_group"]),
            str(row["embodiment"]),
            str(row["task"]),
            str(row["seed"]),
        )
    )
    return rows


def add_hashes(rows: list[dict[str, object]], data_root: Path, workers: int) -> None:
    """Populate hashes in manifest order using bounded parallel reads."""

    paths = [data_root / str(row["relative_path"]) for row in rows]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, digest in enumerate(executor.map(sha256, paths), start=1):
            rows[index - 1]["sha256"] = digest
            print(f"Hashed {index}/{len(rows)}", flush=True)


def validate(rows: list[dict[str, object]]) -> None:
    counts = {}
    for row in rows:
        key = (row["release_group"], row["embodiment"], row["task"])
        counts[key] = counts.get(key, 0) + 1

    expected_files = {
        "cross-embodiment prior": 10,
        "same-embodiment prior": 9,
        "target demonstrations": 1,
    }
    errors = []
    for group, expected in expected_files.items():
        group_rows = [key for key in counts if key[0] == group]
        if len(group_rows) != 12:
            errors.append(
                f"{group}: expected 12 embodiment/task groups, found {len(group_rows)}"
            )
        for key in group_rows:
            if counts[key] != expected:
                errors.append(f"{key}: expected {expected} files, found {counts[key]}")
    if errors:
        raise ValueError("Manifest validation failed:\n" + "\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_root", type=Path, help="Directory containing the mg/ and human/ trees"
    )
    parser.add_argument("--output", type=Path, default=Path("dataset/manifest.csv"))
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="Hash every selected file (slow for the full release)",
    )
    parser.add_argument(
        "--hash-workers",
        type=int,
        default=4,
        help="Parallel file readers used with --sha256 (default: 4)",
    )
    args = parser.parse_args()
    if args.hash_workers <= 0:
        parser.error("--hash-workers must be positive")

    data_root = args.data_root.resolve()
    rows = build_manifest(data_root)
    validate(rows)
    if args.sha256:
        add_hashes(rows, data_root, args.hash_workers)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_suffix(args.output.suffix + ".staging")
    with staging.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(staging, args.output)

    total_bytes = sum(int(row["bytes"]) for row in rows)
    print(f"Wrote {len(rows)} files ({total_bytes / 2**30:.2f} GiB) to {args.output}")


if __name__ == "__main__":
    main()

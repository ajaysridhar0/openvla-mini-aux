#!/usr/bin/env python3
"""Prepare a separate, immutable-copy BARX HDF5 source for MimicGen."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from barx.mimicgen_release import (  # noqa: E402
    sha256,
    source_inventory,
    task_names,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=task_names(), required=True)
    parser.add_argument(
        "--demos",
        type=int,
        default=1,
        help="number of source demonstrations to annotate (default: 1)",
    )
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        parser.error(f"source HDF5 does not exist: {source}")
    if source == output:
        parser.error("--output must differ from --source")
    if output.exists():
        parser.error(f"output already exists: {output}")
    if args.demos < 1:
        parser.error("--demos must be positive")

    from barx.mimicgen_release import registry

    task = registry()["tasks"][args.task]
    source_before = sha256(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Importing RoboCasa registers the BARX environments with RoboSuite.
    import robocasa  # noqa: F401
    from mimicgen.scripts.prepare_src_dataset import prepare_src_dataset

    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".staging", dir=output.parent
    )
    os.close(descriptor)
    staging = Path(staging_name)
    staging.unlink()
    try:
        prepare_src_dataset(
            dataset_path=str(source),
            env_interface_name=task["interface"],
            env_interface_type="robosuite",
            n=args.demos,
            output_path=str(staging),
        )
        source_after = sha256(source)
        if source_after != source_before:
            raise RuntimeError("source HDF5 changed during preparation")
        inventory = source_inventory(staging)
        if inventory["prepared_demonstrations"] != args.demos:
            raise RuntimeError(
                "prepared demo count mismatch: "
                f"{inventory['prepared_demonstrations']} != {args.demos}"
            )
        output_sha256 = sha256(staging)
        staging.replace(output)
    finally:
        if staging.exists():
            staging.unlink()
    report = {
        "task": args.task,
        "source_sha256_before": source_before,
        "source_sha256_after": source_after,
        "source_unchanged": True,
        "output_sha256": output_sha256,
        "output": inventory,
    }
    summary = args.summary or output.with_suffix(".prepare.json")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

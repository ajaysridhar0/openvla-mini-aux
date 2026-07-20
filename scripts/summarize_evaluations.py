#!/usr/bin/env python3
"""Collect BARX evaluation summaries into one CSV file."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


FIELDS = (
    "task",
    "embodiment",
    "checkpoint",
    "status",
    "episodes",
    "successes",
    "success_rate",
    "duration_seconds",
    "run_directory",
    "error",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect(rollout_root: Path) -> list[dict[str, Any]]:
    """Return completed and failed run summaries below ``rollout_root``."""

    rows = []
    for summary_path in sorted(rollout_root.rglob("summary.json")):
        run_directory = summary_path.parent
        config_path = run_directory / "config.json"
        if not config_path.is_file():
            raise ValueError(f"Missing config.json beside {summary_path}")
        summary = read_json(summary_path)
        config = read_json(config_path)
        rows.append(
            {
                "task": summary.get("task", config.get("task")),
                "embodiment": summary.get("embodiment", config.get("embodiment")),
                "checkpoint": config.get("pretrained_checkpoint"),
                "status": summary["status"],
                "episodes": summary.get("episodes", 0),
                "successes": summary.get("successes", 0),
                "success_rate": summary.get("success_rate"),
                "duration_seconds": summary.get("duration_seconds"),
                "run_directory": run_directory.relative_to(rollout_root).as_posix(),
                "error": summary.get("error", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".staging")
    with staging.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(staging, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_root", type=Path, nargs="?", default=Path("rollouts"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results.csv"))
    args = parser.parse_args()

    rows = collect(args.rollout_root.resolve())
    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} run summaries to {args.output}")


if __name__ == "__main__":
    main()

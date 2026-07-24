#!/usr/bin/env python3
"""Generate a bounded, inspectable BARX MimicGen dataset."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from barx.mimicgen_release import (  # noqa: E402
    embodiment_names,
    generated_inventory,
    registry,
    resolved_config,
    source_inventory,
    task_names,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--task", choices=task_names(), required=True)
    parser.add_argument("--embodiment", choices=embodiment_names(), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--successes", type=int, required=True)
    parser.add_argument("--max-attempts", type=int, required=True)
    parser.add_argument("--source-demos", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    video = args.video.resolve()
    if not source.is_file():
        parser.error(f"prepared source HDF5 does not exist: {source}")
    if output_dir.exists():
        parser.error(f"output directory already exists: {output_dir}")
    inventory = source_inventory(source)
    if inventory["prepared_demonstrations"] < args.source_demos:
        parser.error(
            "prepared source contains fewer annotated demonstrations than "
            f"--source-demos ({inventory['prepared_demonstrations']} < "
            f"{args.source_demos})"
        )

    config = resolved_config(
        source=source,
        task=args.task,
        embodiment=args.embodiment,
        output_dir=output_dir,
        seed=args.seed,
        successes=args.successes,
        max_attempts=args.max_attempts,
        source_demos=args.source_demos,
    )
    output_dir.mkdir(parents=True)
    video.parent.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "resolved-config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    command = [
        sys.executable,
        "-m",
        "mimicgen.scripts.generate_dataset",
        "--config",
        str(config_path),
        "--source",
        str(source),
        "--folder",
        str(output_dir),
        "--num_demos",
        str(args.successes),
        "--seed",
        str(args.seed),
        "--max_attempts",
        str(args.max_attempts),
        "--video_path",
        str(video),
    ]
    print("Resolved command:", shlex.join(command), flush=True)
    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYOPENGL_PLATFORM", "egl")
    subprocess.run(command, check=True, env=environment)

    run_dir = output_dir / f"{args.successes}demos_seed{args.seed}"
    stats_path = run_dir / "important_stats.json"
    if not stats_path.is_file():
        raise RuntimeError(f"generation stats are missing: {stats_path}")
    stats = json.loads(stats_path.read_text())
    if stats["num_success"] < args.successes:
        raise RuntimeError(
            f"collected {stats['num_success']} successes; expected {args.successes}"
        )

    from scripts.stage_release_data import normalize_hdf5, verify_hdf5

    normalization_embodiment = registry()["embodiments"][args.embodiment][
        "normalization_embodiment"
    ]
    generated = []
    for name in ("demo.hdf5", "demo_failed.hdf5"):
        path = run_dir / name
        if not path.is_file():
            continue
        demos = normalize_hdf5(
            path,
            normalization_embodiment,
            actions_are_canonical=True,
        )
        verify_hdf5(path, demos)
        generated.append(generated_inventory(path))
    if not generated:
        raise RuntimeError("generation produced no HDF5 output")
    if not video.is_file() or video.stat().st_size == 0:
        raise RuntimeError(f"generation video is missing or empty: {video}")

    report = {
        "task": args.task,
        "embodiment": args.embodiment,
        "seed": args.seed,
        "requested_successes": args.successes,
        "max_attempts": args.max_attempts,
        "stats": stats,
        "generated_hdf5": generated,
        "video": {
            "file": video.name,
            "bytes": video.stat().st_size,
        },
    }
    summary_path = output_dir / "generation-summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit frozen BARX conditions for target-set and policy-camera validity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from barx.benchmark import EMBODIMENTS, IMAGE_HEIGHT, IMAGE_WIDTH, TASKS
from barx.evaluation_conditions import (
    load_bundle,
    project_target_center,
    validate_condition_semantics,
)

ROOT = Path(__file__).resolve().parents[1]


def audit(args: argparse.Namespace) -> dict[str, object]:
    tasks = args.task or list(TASKS)
    embodiments = args.embodiment or list(EMBODIMENTS)
    results: list[dict[str, object]] = []
    invalid: list[dict[str, object]] = []

    for task in tasks:
        for embodiment in embodiments:
            payload, states, model_xmls = load_bundle(
                args.conditions_dir,
                task=task,
                embodiment=embodiment,
            )
            bundle_invalid = 0
            for entry, state, model_xml in zip(payload["episodes"], states, model_xmls):
                projection = None
                try:
                    validate_condition_semantics(entry, task=task)
                    projection = None
                    if task != "turn_on_sink_faucet":
                        projection = project_target_center(
                            entry,
                            state,
                            model_xml,
                            camera_name=EMBODIMENTS[embodiment].camera,
                            image_width=IMAGE_WIDTH,
                            image_height=IMAGE_HEIGHT,
                        )
                        if not projection["center_in_frame"]:
                            raise ValueError(
                                "target center is outside the policy camera"
                            )
                except ValueError as error:
                    bundle_invalid += 1
                    invalid.append(
                        {
                            "task": task,
                            "embodiment": embodiment,
                            "episode": entry["episode"],
                            "seed": entry["seed"],
                            "condition_id": entry["condition_id"],
                            "instruction": entry["instruction"],
                            "reason": str(error),
                            "projection": projection,
                        }
                    )
            results.append(
                {
                    "task": task,
                    "embodiment": embodiment,
                    "conditions": len(payload["episodes"]),
                    "valid": len(payload["episodes"]) - bundle_invalid,
                    "invalid": bundle_invalid,
                }
            )
            print(
                f"{task}/{embodiment}: "
                f"{len(payload['episodes']) - bundle_invalid}/"
                f"{len(payload['episodes'])} valid"
            )

    report = {
        "status": "PASS" if not invalid else "FAIL",
        "conditions_dir": str(args.conditions_dir),
        "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "bundles": results,
        "invalid_conditions": invalid,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions-dir",
        type=Path,
        default=ROOT / "evaluation" / "conditions",
    )
    parser.add_argument("--task", action="append", choices=TASKS)
    parser.add_argument("--embodiment", action="append", choices=EMBODIMENTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write the report but do not exit nonzero for invalid conditions",
    )
    args = parser.parse_args()
    report = audit(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "bundles": len(report["bundles"]),
                "invalid": len(report["invalid_conditions"]),
            }
        )
    )
    if report["status"] != "PASS" and not args.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

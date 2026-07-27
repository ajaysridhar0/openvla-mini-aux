#!/usr/bin/env python3
"""Load, restore, and hash-verify a frozen RoboCasa-X condition bundle."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from barx.benchmark import (
    EMBODIMENTS,
    EVALUATION_GLOBAL_SEED,
    EVALUATION_START_SEED,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MIN_TARGET_VISIBLE_PIXELS,
    TASKS,
)
from barx.evaluation_conditions import (
    load_bundle,
    restore_frozen_condition,
    target_visible_pixel_count,
    validate_condition_bundle_for_evaluation,
)
from barx.simulation import build_environment_config, initialize_observation_utils

ROOT = Path(__file__).resolve().parents[1]


def verify(args: argparse.Namespace) -> None:
    from robocasa.utils.robomimic.robomimic_env_utils import create_env

    payload, states, model_xmls = load_bundle(
        args.conditions_dir,
        task=args.task,
        embodiment=args.embodiment,
    )
    entries = payload["episodes"]
    count = len(entries) if args.episodes is None else args.episodes
    if count < 1 or count > len(entries):
        raise ValueError(
            f"--episodes must be between 1 and {len(entries)}, received {count}"
        )
    validate_condition_bundle_for_evaluation(
        payload,
        states,
        model_xmls,
        task=args.task,
        camera_name=EMBODIMENTS[args.embodiment].camera,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        count=count,
    )

    random.seed(EVALUATION_GLOBAL_SEED)
    np.random.seed(EVALUATION_GLOBAL_SEED)
    config = build_environment_config(args.task, args.embodiment)
    initialize_observation_utils(config)
    env = create_env(
        **config,
        env_type=1,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
        use_camera_obs=False,
        rng=np.random.default_rng(EVALUATION_START_SEED),
    )
    camera_key = f"{EMBODIMENTS[args.embodiment].camera}_image"
    try:
        for index, (entry, state, model_xml) in enumerate(
            zip(entries[:count], states[:count], model_xmls[:count]),
            start=1,
        ):
            observation, metadata = restore_frozen_condition(
                env, entry, state, model_xml
            )
            if camera_key not in observation:
                raise RuntimeError(
                    f"Condition {entry['condition_id']} has no {camera_key} observation"
                )
            visible_pixels = None
            if args.task != "turn_on_sink_faucet":
                visible_pixels = target_visible_pixel_count(
                    env,
                    camera_name=EMBODIMENTS[args.embodiment].camera,
                    image_width=IMAGE_WIDTH,
                    image_height=IMAGE_HEIGHT,
                )
                if visible_pixels < MIN_TARGET_VISIBLE_PIXELS:
                    raise RuntimeError(
                        f"Condition {entry['condition_id']} exposes only "
                        f"{visible_pixels} target pixels; expected at least "
                        f"{MIN_TARGET_VISIBLE_PIXELS}"
                    )
            visibility = (
                "" if visible_pixels is None else f", target_pixels={visible_pixels}"
            )
            print(
                f"Verified {args.task}/{args.embodiment} {index}/{count}: "
                f"condition={entry['condition_id']}, seed={entry['seed']}, "
                f"layout={metadata['layout_id']}, style={metadata['style_id']}"
                f"{visibility}"
            )
    finally:
        env.env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--embodiment", choices=EMBODIMENTS, required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument(
        "--conditions-dir",
        type=Path,
        default=ROOT / "evaluation" / "conditions",
    )
    args = parser.parse_args()
    verify(args)


if __name__ == "__main__":
    main()

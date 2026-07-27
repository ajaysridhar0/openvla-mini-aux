#!/usr/bin/env python3
"""Generate frozen BARX policy-start states from the paper seed protocol."""

from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path

import numpy as np

from barx.action_space import robocasa_noop_action
from barx.benchmark import (
    EMBODIMENTS,
    EVALUATION_EPISODES,
    EVALUATION_GLOBAL_SEED,
    EVALUATION_START_SEED,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MIN_TARGET_VISIBLE_PIXELS,
    SETTLE_STEPS,
    TASKS,
)
from barx.evaluation_conditions import (
    make_entry,
    project_target_center,
    target_visible_pixel_count,
    validate_condition_semantics,
    write_bundle,
)
from barx.simulation import build_environment_config, initialize_observation_utils

ROOT = Path(__file__).resolve().parents[1]


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def generate(args: argparse.Namespace) -> tuple[Path, Path]:
    from robocasa.utils.robomimic.robomimic_env_utils import create_env

    # Robosuite uses the legacy global NumPy RNG for arm initialization
    # noise. This is separate from RoboCasa's per-episode Generator and was
    # seeded to 7 by the paper evaluator before environment construction.
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
        rng=np.random.default_rng(args.start_seed),
    )

    entries = []
    states = []
    model_xmls = []
    rejected_candidates = []
    candidates_examined = 0
    max_candidates = args.max_candidates or args.episodes * 4
    try:
        while len(entries) < args.episodes:
            if candidates_examined >= max_candidates:
                raise RuntimeError(
                    f"Found only {len(entries)} valid conditions after "
                    f"{candidates_examined} candidates"
                )
            seed = args.start_seed + candidates_examined
            candidates_examined += 1
            env.env.rng = np.random.default_rng(seed)
            env.reset()
            for _ in range(SETTLE_STEPS):
                env.step(robocasa_noop_action())

            state = np.asarray(env.env.sim.get_state().flatten(), dtype=np.float64)
            ep_meta = env.env.get_ep_meta()
            model_xml = env.env._last_model_xml
            entry = make_entry(
                episode=len(entries),
                seed=seed,
                ep_meta=ep_meta,
                model_xml=model_xml,
                policy_start_state=state,
            )
            validate_condition_semantics(entry, task=args.task)

            projection = None
            visible_pixels = None
            rejection_reason = None
            if args.task != "turn_on_sink_faucet":
                projection = project_target_center(
                    entry,
                    state,
                    model_xml,
                    camera_name=EMBODIMENTS[args.embodiment].camera,
                    image_width=IMAGE_WIDTH,
                    image_height=IMAGE_HEIGHT,
                )
                if not projection["center_in_frame"]:
                    rejection_reason = "target center outside policy camera"
                    visible_pixels = 0
                else:
                    visible_pixels = target_visible_pixel_count(
                        env,
                        camera_name=EMBODIMENTS[args.embodiment].camera,
                        image_width=IMAGE_WIDTH,
                        image_height=IMAGE_HEIGHT,
                    )
                    if visible_pixels < MIN_TARGET_VISIBLE_PIXELS:
                        rejection_reason = (
                            f"target has only {visible_pixels} visible pixels"
                        )
                entry["target_visibility"] = {
                    **projection,
                    "visible_pixels": visible_pixels,
                    "minimum_visible_pixels": MIN_TARGET_VISIBLE_PIXELS,
                    "camera": EMBODIMENTS[args.embodiment].camera,
                    "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
                }

            if (
                args.condition_protocol == "visible-target-v1"
                and rejection_reason is not None
            ):
                rejected_candidates.append(
                    {
                        "seed": seed,
                        "layout_id": ep_meta["layout_id"],
                        "style_id": ep_meta["style_id"],
                        "reason": rejection_reason,
                        "projection": projection,
                        "visible_pixels": visible_pixels,
                    }
                )
                print(
                    f"Rejected {args.task}/{args.embodiment} seed={seed}: "
                    f"{rejection_reason}"
                )
                continue

            entries.append(entry)
            states.append(state)
            model_xmls.append(model_xml)
            print(
                f"{args.task}/{args.embodiment} episode {len(entries)}/{args.episodes}: "
                f"seed={seed}, layout={ep_meta['layout_id']}, style={ep_meta['style_id']}"
            )
    finally:
        env.env.close()

    return write_bundle(
        args.output,
        task=args.task,
        embodiment=args.embodiment,
        entries=entries,
        states=states,
        model_xmls=model_xmls,
        provenance={
            "description": (
                "Generated from the BARX scene protocol with target visibility "
                "validation"
                if args.condition_protocol == "visible-target-v1"
                else "Regenerated from the frozen BARX paper code and seed protocol"
            ),
            "repository_revision": git_revision(),
            "condition_protocol": args.condition_protocol,
            "start_seed": args.start_seed,
            "global_seed": EVALUATION_GLOBAL_SEED,
            "settle_steps": SETTLE_STEPS,
            "candidates_examined": candidates_examined,
            "rejected_candidates": rejected_candidates,
            "minimum_target_visible_pixels": MIN_TARGET_VISIBLE_PIXELS,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--embodiment", choices=EMBODIMENTS, required=True)
    parser.add_argument("--episodes", type=int, default=EVALUATION_EPISODES)
    parser.add_argument("--start-seed", type=int, default=EVALUATION_START_SEED)
    parser.add_argument(
        "--condition-protocol",
        choices=("visible-target-v1", "historical-consecutive-seeds"),
        default="visible-target-v1",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        help="Maximum candidate seeds to scan; defaults to four times --episodes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation" / "conditions",
    )
    args = parser.parse_args()
    json_path, state_path = generate(args)
    print(f"Wrote {json_path} and {state_path}")


if __name__ == "__main__":
    main()

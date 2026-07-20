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
    SETTLE_STEPS,
    TASKS,
)
from barx.evaluation_conditions import make_entry, write_bundle
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
    try:
        for episode in range(args.episodes):
            seed = args.start_seed + episode
            env.env.rng = np.random.default_rng(seed)
            env.reset()
            for _ in range(SETTLE_STEPS):
                env.step(robocasa_noop_action())

            state = np.asarray(env.env.sim.get_state().flatten(), dtype=np.float64)
            ep_meta = env.env.get_ep_meta()
            model_xml = env.env._last_model_xml
            entries.append(
                make_entry(
                    episode=episode,
                    seed=seed,
                    ep_meta=ep_meta,
                    model_xml=model_xml,
                    policy_start_state=state,
                )
            )
            states.append(state)
            model_xmls.append(model_xml)
            print(
                f"{args.task}/{args.embodiment} episode {episode + 1}/{args.episodes}: "
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
            "description": "Regenerated from the frozen BARX paper code and seed protocol",
            "repository_revision": git_revision(),
            "start_seed": args.start_seed,
            "global_seed": EVALUATION_GLOBAL_SEED,
            "settle_steps": SETTLE_STEPS,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--embodiment", choices=EMBODIMENTS, required=True)
    parser.add_argument("--episodes", type=int, default=EVALUATION_EPISODES)
    parser.add_argument("--start-seed", type=int, default=EVALUATION_START_SEED)
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

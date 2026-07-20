#!/usr/bin/env python3
"""Compare reconstructed BARX conditions with historical rollout videos."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

import cv2
import imageio
import numpy as np

from barx.action_space import robocasa_noop_action
from barx.benchmark import (
    EMBODIMENTS,
    EVALUATION_GLOBAL_SEED,
    EVALUATION_START_SEED,
    SETTLE_STEPS,
    TASKS,
)
from barx.simulation import build_environment_config, initialize_observation_utils


def first_video_frame(path: Path) -> np.ndarray:
    """Decode the first RGB frame of an MP4."""

    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not decode {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def historical_video(rollout_dir: Path, episode: int) -> Path:
    matches = list(rollout_dir.glob(f"*--episode={episode}--*.mp4"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one episode-{episode} MP4 in {rollout_dir}, found {len(matches)}"
        )
    return matches[0]


def encode_like_historical(image: np.ndarray, path: Path) -> np.ndarray:
    """Apply the imageio writer used by the paper evaluator, then decode it."""

    writer = imageio.get_writer(path, fps=30)
    try:
        # Two frames avoid codec edge cases for a one-frame video.
        writer.append_data(image)
        writer.append_data(image)
    finally:
        writer.close()
    return first_video_frame(path)


def structural_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Compute the standard Gaussian-window SSIM over RGB channels."""

    left = left.astype(np.float64)
    right = right.astype(np.float64)
    c1, c2 = 6.5025, 58.5225
    left_mean = cv2.GaussianBlur(left, (11, 11), 1.5)
    right_mean = cv2.GaussianBlur(right, (11, 11), 1.5)
    left_variance = cv2.GaussianBlur(left * left, (11, 11), 1.5) - left_mean**2
    right_variance = (
        cv2.GaussianBlur(right * right, (11, 11), 1.5) - right_mean**2
    )
    covariance = (
        cv2.GaussianBlur(left * right, (11, 11), 1.5)
        - left_mean * right_mean
    )
    numerator = (2 * left_mean * right_mean + c1) * (2 * covariance + c2)
    denominator = (
        (left_mean**2 + right_mean**2 + c1)
        * (left_variance + right_variance + c2)
    )
    return float(np.mean(numerator / denominator))


def image_metrics(current: np.ndarray, historical: np.ndarray) -> dict[str, float]:
    if current.shape != historical.shape:
        raise ValueError(
            f"Frame shapes differ: reconstructed {current.shape}, historical {historical.shape}"
        )
    difference = current.astype(np.float64) - historical
    mse = float(np.mean(difference**2))
    return {
        "mae": float(np.mean(np.abs(difference))),
        "psnr_db": float("inf") if mse == 0 else float(10 * np.log10(255**2 / mse)),
        "ssim": structural_similarity(current, historical),
    }


def audit(args: argparse.Namespace) -> list[dict[str, object]]:
    from robocasa.utils.robomimic.robomimic_env_utils import create_env

    episodes = sorted(set(args.episodes))
    if not episodes or episodes[0] < 1:
        raise ValueError("Episodes use historical 1-based numbering and must be positive")

    random.seed(args.global_seed)
    np.random.seed(args.global_seed)
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
    camera = EMBODIMENTS[args.embodiment].camera
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="barx-rollout-audit-") as directory:
            temporary_dir = Path(directory)
            selected_episodes = set(episodes)
            for episode in range(1, episodes[-1] + 1):
                # Replay every preceding reset. RoboCasa keeps some sampled
                # Python-side scene state across hard resets, so merely
                # advancing the random generators is not equivalent.
                seed = args.start_seed + episode - 1
                env.env.rng = np.random.default_rng(seed)
                observation = env.reset()
                for _ in range(SETTLE_STEPS):
                    observation, *_ = env.step(robocasa_noop_action())

                if episode not in selected_episodes:
                    print(f"Replayed prerequisite episode {episode} (seed={seed})")
                    continue

                image = (
                    255
                    * np.transpose(
                        observation[f"{camera}_image"],
                        (1, 2, 0),
                    )
                ).astype(np.uint8)
                reconstructed = encode_like_historical(
                    image,
                    temporary_dir / f"episode-{episode}.mp4",
                )
                video = historical_video(args.rollout_dir, episode)
                historical = first_video_frame(video)
                metadata = env.env.get_ep_meta()
                if args.artifacts_dir:
                    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
                    prefix = (
                        f"{args.task}-{args.embodiment}-episode-{episode}"
                    )
                    for suffix, frame in (
                        ("reconstructed-raw", image),
                        ("reconstructed-mp4", reconstructed),
                        ("historical-mp4", historical),
                    ):
                        cv2.imwrite(
                            str(args.artifacts_dir / f"{prefix}-{suffix}.png"),
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        )
                results.append(
                    {
                        "episode": episode,
                        "seed": seed,
                        "layout_id": metadata["layout_id"],
                        "style_id": metadata["style_id"],
                        "instruction": metadata.get("lang", ""),
                        "historical_video": str(video),
                        **image_metrics(reconstructed, historical),
                    }
                )
    finally:
        env.env.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--embodiment", choices=EMBODIMENTS, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--start-seed", type=int, default=EVALUATION_START_SEED)
    parser.add_argument("--global-seed", type=int, default=EVALUATION_GLOBAL_SEED)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifacts-dir", type=Path)
    args = parser.parse_args()
    payload = {
        "task": args.task,
        "embodiment": args.embodiment,
        "global_seed": args.global_seed,
        "results": audit(args),
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

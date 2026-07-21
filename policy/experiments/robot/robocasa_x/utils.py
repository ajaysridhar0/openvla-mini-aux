"""Utilities for evaluating BARX policies in RoboCasa-X."""

import math
import os

import imageio
import numpy as np
from barx.action_space import robocasa_action, robocasa_noop_action

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


def get_robocasa_dummy_action():
    """Return the unified RoboCasa-X no-op action."""
    return robocasa_noop_action()


def pad_action_robocasa(action: list):
    """Expand a policy action into the unified RoboCasa-X action layout."""
    return robocasa_action(action)


def patch_model_for_generation(model):
    """Add required attributes for newer transformers compatibility."""
    if not hasattr(model, '_supports_cache_class'):
        model._supports_cache_class = False
    return model


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, rollout_dir=None):
    """Saves an MP4 replay of an episode."""
    if rollout_dir is None:
        rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

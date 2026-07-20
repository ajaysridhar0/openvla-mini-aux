"""Utilities for evaluating BARX policies in RoboCasa-X."""

import math
import os

import imageio
import numpy as np
import tensorflow as tf
from barx.action_space import robocasa_action, robocasa_noop_action
from PIL import Image

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)

# def get_libero_env(task, model_family, resolution=256):
#     """Initializes and returns the LIBERO environment, along with the task description."""
#     task_description = task.language
#     task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
#     env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
#     env = OffScreenRenderEnv(**env_args)
#     env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
#     return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_robocasa_dummy_action():
    """Return the unified RoboCasa-X no-op action."""
    return robocasa_noop_action()


def pad_action_robocasa(action: list):
    """Expand a policy action into the unified RoboCasa-X action layout."""
    return robocasa_action(action)


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size, key="agentview_image", flip_image=True, is_robocasa=False):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs[key]
    if flip_image:
        img = np.flipud(img)
    # img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    if is_robocasa:
        img = np.transpose(img, (1, 2, 0))
        img = (255*img).astype(np.uint8)
    img = Image.fromarray(img)
    img = img.resize(resize_size, Image.Resampling.LANCZOS)  # resize to size seen at train time
    img = img.convert("RGB")
    return np.array(img)


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

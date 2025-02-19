"""
traj_transforms.py

Contains trajectory transforms used in the orca data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory length).
"""

import logging
from typing import Dict

import tensorflow as tf

def chunk_act_obs(
    traj: Dict, 
    window_size: int, 
    past_obj_pose_window_size: int = 0,
    past_2D_trace_window_size: int = 0,
    future_action_window_size: int = 0, 
    future_obj_pose_window_size: int = 0,
    future_2D_trace_window_size: int = 0,
    obj_pose_stride: int = 1,
    ee_pose_2D_stride: int = 1,
) -> Dict:
    """
    Chunks actions and observations into windows, with separate past/current/future for aux task data.
    """
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]
    
    # Create indices for the main observation window
    chunk_indices = tf.broadcast_to(tf.range(-window_size + 1, 1), [traj_len, window_size]) + tf.broadcast_to(
        tf.range(traj_len)[:, None], [traj_len, window_size]
    )

    # Create indices for action chunks (current + future only)
    action_chunk_indices = tf.broadcast_to(
        tf.range(0, 1 + future_action_window_size),
        [traj_len, 1 + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(traj_len)[:, None],
        [traj_len, 1 + future_action_window_size],
    )

    # Handle past indices for aux tasks
    past_obj_pose_indices = None
    if past_obj_pose_window_size > 0:
        past_obj_pose_indices = tf.broadcast_to(
            tf.range(-past_obj_pose_window_size, 0, obj_pose_stride),
            [traj_len, (past_obj_pose_window_size + obj_pose_stride - 1) // obj_pose_stride],
        ) + tf.broadcast_to(
            tf.range(traj_len)[:, None],
            [traj_len, (past_obj_pose_window_size + obj_pose_stride - 1) // obj_pose_stride],
        )

    past_ee_pose_indices = None
    if past_2D_trace_window_size > 0:
        past_ee_pose_indices = tf.broadcast_to(
            tf.range(-past_2D_trace_window_size, 0, ee_pose_2D_stride),
            [traj_len, (past_2D_trace_window_size + ee_pose_2D_stride - 1) // ee_pose_2D_stride],
        ) + tf.broadcast_to(
            tf.range(traj_len)[:, None],
            [traj_len, (past_2D_trace_window_size + ee_pose_2D_stride - 1) // ee_pose_2D_stride],
        )

    # Handle future indices for aux tasks
    future_obj_pose_indices = None
    if future_obj_pose_window_size > 0:
        future_obj_pose_indices = tf.broadcast_to(
            tf.range(1, future_obj_pose_window_size + 1, obj_pose_stride),
            [traj_len, (future_obj_pose_window_size + obj_pose_stride - 1) // obj_pose_stride],
        ) + tf.broadcast_to(
            tf.range(traj_len)[:, None],
            [traj_len, (future_obj_pose_window_size + obj_pose_stride - 1) // obj_pose_stride],
        )

    future_ee_pose_indices = None
    if future_2D_trace_window_size > 0:
        future_ee_pose_indices = tf.broadcast_to(
            tf.range(1, future_2D_trace_window_size + 1, ee_pose_2D_stride),
            [traj_len, (future_2D_trace_window_size + ee_pose_2D_stride - 1) // ee_pose_2D_stride],
        ) + tf.broadcast_to(
            tf.range(traj_len)[:, None],
            [traj_len, (future_2D_trace_window_size + ee_pose_2D_stride - 1) // ee_pose_2D_stride],
        )

    # Floor indices at 0 and ceiling at goal_timestep
    if "timestep" in traj["task"]:
        goal_timestep = traj["task"]["timestep"]
    else:
        goal_timestep = tf.fill([traj_len], traj_len - 1)

    floored_chunk_indices = tf.maximum(chunk_indices, 0)
    
    # Gather data using computed indices
    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), traj["observation"])
    
    # Handle actions (current + future only)
    floored_action_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])
    traj["action"] = tf.gather(traj["action"], floored_action_indices)
    
    # Handle absolute vs relative actions
    if "absolute_action_mask" not in traj and future_action_window_size > 0:
        logging.warning(
            "future_action_window_size > 0 but no absolute_action_mask was provided. "
            "Assuming all actions are relative for the purpose of making neutral actions."
        )
    absolute_action_mask = traj.get("absolute_action_mask", tf.zeros([traj_len, action_dim], dtype=tf.bool))
    neutral_actions = tf.where(
        absolute_action_mask[:, None, :],
        traj["action"],  # absolute actions are repeated (already done during chunking)
        tf.zeros_like(traj["action"]),  # relative actions are zeroed
    )

    # Make actions neutral past the goal timestep
    action_past_goal = action_chunk_indices > goal_timestep[:, None]
    traj["action"] = tf.where(action_past_goal[:, :, None], neutral_actions, traj["action"])
    
    # Convert bboxes to center poses
    bboxes = tf.cast(traj["obj_bboxes"], tf.float32)
    center_x = (bboxes[..., 0] + bboxes[..., 2]) / 2.0
    center_y = (bboxes[..., 1] + bboxes[..., 3]) / 2.0
    center_poses = tf.stack([center_x, center_y], axis=-1)
    
    # Handle object poses
    traj["obj_poses_current"] = tf.gather(center_poses, tf.maximum(chunk_indices, 0))
    if past_obj_pose_indices is not None:
        traj["obj_poses_past"] = tf.gather(center_poses, tf.maximum(past_obj_pose_indices, 0))
    if future_obj_pose_indices is not None:
        traj["obj_poses_future"] = tf.gather(center_poses, tf.minimum(future_obj_pose_indices, goal_timestep[:, None]))
    
    # Handle ee poses
    traj["ee_pose_2D_current"] = tf.gather(traj["ee_pose_2D"], tf.maximum(chunk_indices, 0))
    if past_ee_pose_indices is not None:
        traj["ee_pose_2D_past"] = tf.gather(traj["ee_pose_2D"], tf.maximum(past_ee_pose_indices, 0))
    if future_ee_pose_indices is not None:
        traj["ee_pose_2D_future"] = tf.gather(traj["ee_pose_2D"], tf.minimum(future_ee_pose_indices, goal_timestep[:, None]))

    # Add padding mask
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    return traj


def subsample(traj: Dict, subsample_length: int) -> Dict:
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)

    return traj


def add_pad_mask_dict(traj: Dict) -> Dict:
    """
    Adds a dictionary indicating which elements of the observation/task should be treated as padding.
        =>> traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]

    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            # Handles "language_instruction", "image_*", and "depth_*"
            if traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0

            # All other keys should not be treated as padding
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)

        traj[key]["pad_mask_dict"] = pad_mask_dict

    return traj

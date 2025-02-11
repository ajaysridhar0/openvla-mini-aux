"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Tuple

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb
import cv2
from matplotlib import colors
import random

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_robocasa_dummy_action, 
    pad_action_robocasa,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
    patch_model_for_generation,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

# Core imports needed
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import h5py
import json

def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic"):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    if ds_format == "robomimic":
        env_meta = json.loads(f["data"].attrs["env_args"])
    elif ds_format == "r2d2":
        env_meta = dict(f.attrs)
    else:
        raise ValueError
    f.close()
    return env_meta


# Core function to get env_meta from a dataset
def get_env_meta(dataset_path, data_format="robomimic"):
    """
    Load basic metadata from training file, including:
    - env_name
    - env_args
    - env_kwargs
    """
    env_meta = get_env_metadata_from_dataset(
        dataset_path=dataset_path,
        ds_format=data_format
    )
    return env_meta


# Key functions/components for env creation:
def create_env(env_meta, env_name=None, render=False, render_offscreen=False, use_image_obs=True, seed=None):
    env_kwargs = dict(
        env_meta=env_meta,
        env_name=env_name,
        render=render,
        render_offscreen=render_offscreen,
        use_image_obs=use_image_obs,
        seed=seed,
    )
    env = EnvUtils.create_env_from_metadata(**env_kwargs)
    return env


# Map task names to their corresponding dataset paths
ROBOT_TASK_DATA_PATHS = {
    "panda__pn_p_counter_to_cab_aux": "/work/hdd/bcwv/ajaysri/datasets/final_human_collected_demos/panda/PnPCounterToCab/demo_im224_libero_right.hdf5",
    "sawyer__pn_p_counter_to_sink_aux": "/work/hdd/bcwv/ajaysri/datasets/final_human_collected_demos/sawyer/PnPCounterToSink/demo_im224_libero_right.hdf5",
    "ur5e__pn_p_sink_to_counter_aux": "/work/hdd/bcwv/ajaysri/datasets/final_human_collected_demos/ur5e/PnPSinkToCounter/demo_im224_libero_right.hdf5",
    "kinova3__pn_p_counter_to_cab_aux": "/work/hdd/bcwv/ajaysri/datasets/final_human_collected_demos/kinova3/PnPCounterToCab/demo_im224_libero_right.hdf5",
    "panda__pn_p_counter_to_sink_aux": "/work/hdd/bcwv/ajaysri/datasets/final_human_collected_demos/panda/PnPCounterToSink/demo_im128_im224_libero_right.hdf5",
    "sawyer__pn_p_sink_to_counter_aux": "/work/hdd/bcwv/ajaysri/datasets/final_human_collected_demos/sawyer/PnPSinkToCounter/demo_im128_im224_libero_right.hdf5",
    "kinova3__pn_p_counter_to_sink_aux": "/work/hdd/bcwv/ajaysri/datasets/final_human_collected_demos/kinova3/PnPCounterToSink/demo_im224_libero_right.hdf5",
    "panda__pn_p_sink_to_counter_aux": "/work/hdd/bcwv/ajaysri/datasets/final_human_collected_demos/panda/PnPSinkToCounter/demo_im224_libero_right.hdf5",
    "ur5e__pn_p_counter_to_cab_aux": "/work/hdd/bcwv/ajaysri/datasets/final_human_collected_demos/ur5e/PnPCounterToCab/demo_im224_libero_right.hdf5",
    "kinova3__pn_p_sink_to_counter_aux": "/work/hdd/bcwv/ajaysri/datasets/final_human_collected_demos/kinova3/PnPSinkToCounter/demo_im224_libero_right.hdf5",
    "sawyer__pn_p_counter_to_cab_aux": "/work/hdd/bcwv/ajaysri/datasets/final_human_collected_demos/sawyer/PnPCounterToCab/demo_im128_im224_libero_right.hdf5",
    "ur5e__pn_p_counter_to_sink_aux": "/work/hdd/bcwv/ajaysri/datasets/final_human_collected_demos/ur5e/PnPCounterToSink/demo_im224_libero_right.hdf5"
}


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    hf_token: str = Path(".hf_token")                       # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    obs_history: int = 1                             # Number of images to pass in from history
    use_wrist_image: bool = False                    # Use wrist images (doubles the number of input images)

    #################################################################################################################
    # ROBOCASA environment-specific parameters
    #################################################################################################################
    robot_task_names: List[str] = field(default_factory=lambda: [
        "panda__pn_p_counter_to_cab_aux", 
        "kinova3__pn_p_counter_to_sink_aux",
        "sawyer__pn_p_sink_to_counter_aux"
    ])
    #                                       Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id: str = "robocasa_xembod_eval"
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    prefix: str = ''

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "prismatic"        # Name of W&B project to log to (use default!)
    wandb_entity: Optional[str] = None          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    aux_task_types: Optional[str] = None     # Auxiliary task types to query before action prediction
    aux_context_freq: int = 1
    # fmt: on


END_TEXT = "<|im_end|>"


def draw_bbox_on_image(img, bbox_dict, color_map):
    """Draw bounding boxes on image with consistent colors per object.
    
    Args:
        img: numpy array of shape (H,W,3) with values in [0,255]
        bbox_dict: dict of object_name: [x1,y1,x2,y2] in normalized coordinates
        color_map: dict mapping object names to RGB colors
    """
    img_with_bbox = img.copy()
    h, w = img.shape[:2]
    
    for obj_name, bbox in bbox_dict.items():
        if obj_name not in color_map:
            # Generate random RGB color if not already assigned
            color_map[obj_name] = tuple(random.random() for _ in range(3))
        
        color = color_map[obj_name]
        x1, y1, x2, y2 = bbox
        
        # Convert normalized coords to pixel coords
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)
        
        # Draw rectangle
        cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), 
                     tuple(int(c * 255) for c in color), 2)
        
        # Add label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_with_bbox, obj_name, (x1, y1-5), font, 0.5, 
                    tuple(int(c * 255) for c in color), 1)
    
    return img_with_bbox


def draw_trajectory_on_image(img, trajectory_points):
    """Draw end-effector trajectory on image.
    
    Args:
        img: numpy array of shape (H,W,3) with values in [0,255]
        trajectory_points: list of (x,y) tuples in normalized coordinates
    """
    img_with_traj = img.copy()
    h, w = img.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    pixel_points = [(int(x * w), int(y * h)) for x, y in trajectory_points]
    
    # Draw lines connecting consecutive points
    for i in range(len(pixel_points)-1):
        cv2.line(img_with_traj, pixel_points[i], pixel_points[i+1], 
                (0, 255, 0), 2)  # Green color for trajectory
        
    # Draw points
    for point in pixel_points:
        cv2.circle(img_with_traj, point, 3, (255, 0, 0), -1)  # Red dots for waypoints
        
    return img_with_traj


def draw_motion_text_on_image(img, motion_text):
    """Draw motion text as subtitles on the image with black background and white text.
    
    Args:
        img: numpy array of shape (H,W,3) with values in [0,255]
        motion_text: string describing the motion
    """
    img_with_text = img.copy()
    h, w = img.shape[:2]
    
    # Remove end token if present
    motion_text = motion_text.replace('<|im_end|>', '').strip()
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # Reduced from 0.7
    font_thickness = 1
    text_color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)  # Black background
    
    # Split text into lines if too long (wrap at ~40 chars)
    words = motion_text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= 40:  # +1 for space
            current_line.append(word)
            current_length += len(word) + 1
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
    if current_line:
        lines.append(' '.join(current_line))
    
    # Calculate text sizes and positions
    padding = 8  # Slightly reduced padding to match smaller text
    line_spacing = 4  # Slightly reduced spacing to match smaller text
    total_height = 0
    
    # Get size of each line
    line_sizes = []
    for line in lines:
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        line_sizes.append((text_width, text_height))
        total_height += text_height + line_spacing
    
    # Calculate starting y position (near bottom of image)
    y_pos = h - total_height - padding
    
    # Draw background and text for each line
    for line, (text_width, text_height) in zip(lines, line_sizes):
        # Calculate text position
        x_pos = (w - text_width) // 2  # Center text
        
        # Draw background rectangle
        bg_pts = np.array([[x_pos - padding, y_pos - padding],
                          [x_pos + text_width + padding, y_pos + text_height + padding]])
        cv2.rectangle(img_with_text, bg_pts[0], bg_pts[1], bg_color, -1)
        
        # Draw text
        cv2.putText(img_with_text, line, (x_pos, y_pos + text_height), 
                    font, font_scale, text_color, font_thickness)
        
        y_pos += text_height + line_spacing
    
    return img_with_text


def eval_single_task(cfg: GenerateConfig, model, robot_task_name: str, log_file) -> None:
    """Evaluate a single robot task"""
    cfg.task_suite_name = robot_task_name
    cfg.unnorm_key = cfg.task_suite_name

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family in ["openvla", "prismatic"]:
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize environment
    env_meta = get_env_meta(ROBOT_TASK_DATA_PATHS[cfg.task_suite_name])
    env = create_env(env_meta, seed=cfg.seed)

    # Get task description
    if "pn_p_counter_to_cab" in cfg.task_suite_name:
        task_description = "pick up the carrot on the counter and put it in the cabinet"
    elif "pn_p_cab_to_counter" in cfg.task_suite_name:
        task_description = "pick up the carrot in the cabinet and put it on the counter"
    elif "pn_p_counter_to_microwave" in cfg.task_suite_name:
        task_description = "pick up the carrot on the counter and put it in the microwave"
    elif "pn_p_counter_to_sink" in cfg.task_suite_name:
        task_description = "pick up the carrot on the counter and put it in the sink"
    elif "pn_p_sink_to_counter" in cfg.task_suite_name:
        task_description = "pick up the carrot in the sink and put it on the counter"
    elif "pn_p_microwave_to_counter" in cfg.task_suite_name:
        task_description = "pick up the carrot in the microwave and put it on the counter"
    else:
        raise ValueError(f"Unknown task suite name: {cfg.task_suite_name}")

    # Get robot name
    if "panda" in cfg.task_suite_name:
        robot_name = "panda"
    elif "sawyer" in cfg.task_suite_name:
        robot_name = "sawyer"
    elif "ur5e" in cfg.task_suite_name:
        robot_name = "ur5e"
    elif "kinova3" in cfg.task_suite_name:
        robot_name = "kinova3"
    else:
        raise ValueError(f"Unknown robot name: {cfg.task_suite_name}")

    print(f"Robocasa task suite: {cfg.task_suite_name}")
    log_file.write(f"Robocasa task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Split aux_task_types by "->"
    if cfg.aux_task_types is not None:
        cfg.aux_task_types = cfg.aux_task_types.split("->")

    # Initialize tracking for single task
    total_episodes = 0
    total_successes = 0

    # Run trials for the task
    for trial_idx in range(cfg.num_trials_per_task):
        print(f"\nTrial {trial_idx + 1}/{cfg.num_trials_per_task}")
        log_file.write(f"\nTrial {trial_idx + 1}/{cfg.num_trials_per_task}\n")

        # Reset environment
        env.reset()

        # Setup
        t = 0
        replay_images = []
        replay_images_with_bbox = []  # Separate list for bbox images
        replay_wrist_images = []  # Add list for wrist images
        bbox_color_map = {}  # Reset color map for each new video
        has_bbox_predictions = False
        has_ee_pose_predictions = False
        has_motion_predictions = False

        if "pn_p_sink_to_counter" in cfg.task_suite_name:
            max_steps = 420
        elif "pn_p_counter_to_cab" in cfg.task_suite_name:
            max_steps = 360
        elif "pn_p_counter_to_sink" in cfg.task_suite_name:
            max_steps = 350
        else:
            max_steps = 400

        print(f"Starting episode {total_episodes+1}...")
        log_file.write(f"Starting episode {total_episodes+1}...\n")
        
        # Create progress bar for steps
        pbar = tqdm.tqdm(total=max_steps + cfg.num_steps_wait, desc='Environment steps')
        
        while t < max_steps + cfg.num_steps_wait:
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_robocasa_dummy_action(robot_name))
                t += 1
                pbar.update(1)
                continue

            # Get preprocessed image
            img = get_libero_image(obs, resize_size, flip_image=False, key="robot0_agentview_right_image", is_robocasa=True)

            # Save preprocessed image for replay video
            replay_images.append(img)

            # use_wrist_image
            if cfg.use_wrist_image:
                wrist_img = get_libero_image(obs, resize_size, key="robot0_eye_in_hand_image", flip_image=False, is_robocasa=True)
                replay_wrist_images.append(wrist_img)

            # buffering #obs_history images, optionally
            image_history = replay_images[-cfg.obs_history :]
            if len(image_history) < cfg.obs_history:
                image_history.extend([replay_images[-1]] * (cfg.obs_history - len(image_history)))

            # same but for optional wrist images
            if cfg.use_wrist_image:
                wrist_image_history = replay_wrist_images[-cfg.obs_history :]
                if len(wrist_image_history) < cfg.obs_history:
                    wrist_image_history.extend([replay_wrist_images[-1]] * (cfg.obs_history - len(wrist_image_history)))
                # interleaved images [... image_t, wrist_t ...]
                image_history = [val for tup in zip(image_history, wrist_image_history) for val in tup]

            # Prepare observations dict
            observation = {
                "full_image": image_history,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }

            # Query model to get action
            output = get_action(
                cfg,
                model,
                observation,
                task_description,
                processor=processor,
            )

            action = output['action']
            
            # Save original image
            replay_images.append(img)
            
            # Process image with visualizations
            current_img = img.copy()
            
            # Draw bounding boxes if available
            if 'bbox' in output:
                has_bbox_predictions = True
                current_img = draw_bbox_on_image(current_img, output['bbox'], bbox_color_map)
            
            # Draw trajectory if available
            if 'ee_pose_2D' in output:
                has_ee_pose_predictions = True
                current_img = draw_trajectory_on_image(current_img, output['ee_pose_2D'])
            
            # Draw motion text if available
            if 'low_level_motion' in output:
                has_motion_predictions = True
                current_img = draw_motion_text_on_image(current_img, output['low_level_motion'])
            
            replay_images_with_bbox.append(current_img)

            # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
            action = normalize_gripper_action(action, binarize=True)

            # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
            # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
            if cfg.model_family in ["openvla", "prismatic"]:
                action = invert_gripper_action(action)

            # Execute action in environment
            obs, reward, done, info = env.step(pad_action_robocasa(action.tolist(), robot_name))
            if info["is_success"]["task"]:
                done = True
                reward = 1
                break
            t += 1
            pbar.update(1)

        pbar.close()

        total_episodes += 1
        if done:
            total_successes += 1

        # Save videos and log results
        save_rollout_video(
            replay_images, total_episodes, success=done, task_description=cfg.task_suite_name, log_file=log_file
        )
        
        # Add wrist camera video saving if enabled
        if cfg.use_wrist_image:
            save_rollout_video(
                replay_wrist_images, total_episodes, success=done, 
                task_description=f"{cfg.task_suite_name}_wrist", log_file=log_file
            )
        
        if has_bbox_predictions or has_ee_pose_predictions or has_motion_predictions:
            viz_suffix = "_with_" + "_".join(
                x for x in ["bbox", "ee_pose", "motion"] 
                if (x == "bbox" and has_bbox_predictions) or 
                   (x == "ee_pose" and has_ee_pose_predictions) or
                   (x == "motion" and has_motion_predictions)
            )
            save_rollout_video(
                replay_images_with_bbox, total_episodes, success=done,
                task_description=f"{cfg.task_suite_name}{viz_suffix}", log_file=log_file
            )

        # Log current results
        success_rate = float(total_successes) / float(total_episodes) * 100
        print(f"Success rate: {success_rate:.1f}%")
        log_file.write(f"Success rate: {success_rate:.1f}%\n")

        # Log to wandb after each episode (only for first trial)
        if cfg.use_wandb:
            log_dict = {
                f"{cfg.task_suite_name}/success_rate": success_rate,
                f"{cfg.task_suite_name}/total_successes": total_successes,
                f"{cfg.task_suite_name}/total_episodes": total_episodes,
                f"{cfg.task_suite_name}/episode_success": 1.0 if done else 0.0,
            }
            
            # Only log videos for the first trial
            if trial_idx == 0:
                group = "success" if done else "failure"
                log_dict[f"{cfg.task_suite_name}/{group}/trial_0"] = wandb.Video(
                    np.array(replay_images).transpose(0, 3, 1, 2)
                )
                
                if cfg.use_wrist_image:
                    log_dict[f"{cfg.task_suite_name}_wrist/{group}/trial_0"] = wandb.Video(
                        np.array(replay_wrist_images).transpose(0, 3, 1, 2)
                    )
                    
                if has_bbox_predictions or has_ee_pose_predictions or has_motion_predictions:
                    viz_suffix = "_with_" + "_".join(
                        x for x in ["bbox", "ee_pose", "motion"] 
                        if (x == "bbox" and has_bbox_predictions) or 
                           (x == "ee_pose" and has_ee_pose_predictions) or
                           (x == "motion" and has_motion_predictions)
                    )
                    log_dict[f"{cfg.task_suite_name}{viz_suffix}/{group}/trial_0"] = wandb.Video(
                        np.array(replay_images_with_bbox).transpose(0, 3, 1, 2)
                    )
            
            wandb.log(log_dict)

        log_file.flush()


@draccus.wrap()
def eval_robocasa(cfg: GenerateConfig) -> None:

    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    for robot_task_name in cfg.robot_task_names: 
        assert robot_task_name in ROBOT_TASK_DATA_PATHS, f"Robot task name {robot_task_name} not in {ROBOT_TASK_DATA_PATHS}!"

    ObsUtils.OBS_KEYS_TO_MODALITIES = {
        'robot0_agentview_right_image': 'rgb', 
        'robot0_eye_in_hand_image': 'rgb', 
        'mean': 'low_dim', 
        'scale': 'low_dim', 
        'logits': 'low_dim'
    }

    # Load model
    model = get_model(cfg)

    # After loading your model, add:
    model = patch_model_for_generation(model)

    if cfg.use_wrist_image:
        model.vision_backbone.image_sequence_len *= 2

    # Move log file initialization before the task loop
    run_id = f"{cfg.prefix}EVAL-{cfg.run_id}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    try:
        # Initialize Weights & Biases logging
        if cfg.use_wandb:
            wandb.init(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                name=run_id,
            )

        # [OpenVLA] Set action un-normalization key
        for robot_task_name in cfg.robot_task_names:
            eval_single_task(cfg, model, robot_task_name, log_file)

        # Save local log file
        log_file.close()

        # Final wandb logging not needed since we're logging throughout
        if cfg.use_wandb:
            wandb.save(local_log_filepath)

    except ZeroDivisionError as e:
        print(f"Caught exception: {e}")
        log_file.write(f"Caught exception: {e}\n")
    finally:
        # Close log file in finally block to ensure it's closed properly
        log_file.close()
        
        # Final wandb logging if needed
        if cfg.use_wandb:
            wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_robocasa()

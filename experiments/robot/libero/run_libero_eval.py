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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

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
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
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

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite.
    #                                       Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
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


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:

    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family in ["openvla", "prismatic"]:
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"{cfg.prefix}EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    cfg.task_suite_name = '_'.join(cfg.task_suite_name.split("_")[:2])
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Split aux_task_types by "->"
    if cfg.aux_task_types is not None:
        cfg.aux_task_types = cfg.aux_task_types.split("->")

    # Initialize tracking for all tasks
    task_episodes = [0] * num_tasks_in_suite
    task_successes = [0] * num_tasks_in_suite
    total_episodes, total_successes = 0, 0

    # Run trials in an interleaved pattern
    for trial_idx in range(cfg.num_trials_per_task):
        # Iterate through each task for this trial
        for task_id in tqdm.tqdm(range(num_tasks_in_suite)): #num_tasks_in_suite
            # Get task
            task = task_suite.get_task(task_id)
            task_name = task.language

            # Get default LIBERO initial states
            initial_states = task_suite.get_task_init_states(task_id)

            # Initialize LIBERO environment and task description
            env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

            print(f"\nTask {task_id}: {task_name} (Trial {trial_idx + 1}/{cfg.num_trials_per_task})")
            log_file.write(f"\nTask {task_id}: {task_name} (Trial {trial_idx + 1}/{cfg.num_trials_per_task})\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[trial_idx])

            # Setup
            t = 0
            replay_images = []
            replay_images_with_bbox = []  # Separate list for bbox images
            bbox_color_map = {}  # Reset color map for each new video
            has_bbox_predictions = False  # Flag to track if we've seen any bbox predictions
            has_ee_pose_predictions = False  # Flag to track if we've seen any ee_pose predictions
            has_motion_predictions = False  # Flag to track if we've seen any motion predictions
            
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400 # 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes[task_id]+1}...")
            log_file.write(f"Starting episode {task_episodes[task_id]+1}...\n")
            # Create progress bar for steps
            pbar = tqdm.tqdm(total=max_steps + cfg.num_steps_wait, desc='Environment steps')
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        pbar.update(1)
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # buffering #obs_history images, optionally
                    image_history = replay_images[-cfg.obs_history :]
                    if len(image_history) < cfg.obs_history:
                        image_history.extend([replay_images[-1]] * (cfg.obs_history - len(image_history)))

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
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
                    
                    replay_images_with_bbox.append(current_img)  # Now includes bbox, trajectory, and motion text

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family in ["openvla", "prismatic"]:
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        break    # Just break the loop, we'll count success after
                    t += 1
                    pbar.update(1)

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break
            pbar.close()

            task_episodes[task_id] += 1
            total_episodes += 1
            if done:
                # Count success only once, here
                task_successes[task_id] += 1
                total_successes += 1

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )
            
            # Save visualization video if we had any predictions
            if has_bbox_predictions or has_ee_pose_predictions or has_motion_predictions:
                viz_suffix = "_with_" + "_".join(
                    x for x in ["bbox", "ee_pose", "motion"] 
                    if (x == "bbox" and has_bbox_predictions) or 
                       (x == "ee_pose" and has_ee_pose_predictions) or
                       (x == "motion" and has_motion_predictions)
                )
                save_rollout_video(
                    replay_images_with_bbox, total_episodes, success=done,
                    task_description=f"{task_description}{viz_suffix}", log_file=log_file
                )

            # If using wandb, log videos
            if cfg.use_wandb and (task_successes[task_id] < 10 or task_episodes[task_id] - task_successes[task_id] < 10):
                group = "success" if done else "failure"
                idx = task_successes[task_id] if done else task_episodes[task_id] - task_successes[task_id]
                log_dict = {
                    f"{task_description}/{group}/{idx}": wandb.Video(np.array(replay_images).transpose(0, 3, 1, 2))
                }
                if has_bbox_predictions or has_ee_pose_predictions or has_motion_predictions:
                    viz_suffix = "_with_" + "_".join(
                        x for x in ["bbox", "ee_pose", "motion"] 
                        if (x == "bbox" and has_bbox_predictions) or 
                           (x == "ee_pose" and has_ee_pose_predictions) or
                           (x == "motion" and has_motion_predictions)
                    )
                    log_dict[f"{task_description}{viz_suffix}/{group}/{idx}"] = wandb.Video(
                        np.array(replay_images_with_bbox).transpose(0, 3, 1, 2)
                    )
                wandb.log(log_dict)

            # Log current results for this task and overall progress
            task_success_rate = float(task_successes[task_id]) / float(task_episodes[task_id]) * 100
            total_success_rate = float(total_successes) / float(total_episodes) * 100
            
            print(f"Task {task_id} success rate: {task_success_rate:.1f}%")
            print(f"Overall success rate: {total_success_rate:.1f}%")
            log_file.write(f"Task {task_id} success rate: {task_success_rate:.1f}%\n")
            log_file.write(f"Overall success rate: {total_success_rate:.1f}%\n")

            # Log to wandb after each episode
            if cfg.use_wandb:
                wandb.log({
                    f"success_rate/{task_description}": task_success_rate / 100,
                    "success_rate/total": total_success_rate / 100,
                    f"num_episodes/{task_description}": task_episodes[task_id],
                    "num_episodes/total": total_episodes,
                })

        # Remove the per-trial summary since we're already logging after each episode
        log_file.flush()

    # Save local log file
    log_file.close()

    # Final wandb logging not needed since we're logging throughout
    if cfg.use_wandb:
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()

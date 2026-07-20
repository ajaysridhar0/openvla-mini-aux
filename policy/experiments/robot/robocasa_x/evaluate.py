"""Evaluate a BARX policy in a RoboCasa-X simulation environment."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Tuple
from termcolor import colored

import draccus
import numpy as np
import tqdm
import re

import wandb
import cv2
from matplotlib import colors
import random


# Make both the root ``barx`` package and ``policy/experiments`` importable
# when this file is invoked directly.
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
POLICY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(POLICY_ROOT))

from barx.names import (
    canonicalize_prediction_keys,
    internal_representation_name,
    representations_for_method,
)
from barx.benchmark import (
    EMBODIMENTS,
    OBJECT_INSTANCE_SPLIT,
    TASK_BY_ENVIRONMENT,
    evaluation_scene_config,
)
from experiments.robot.robocasa_x.utils import (
    get_robocasa_dummy_action, 
    pad_action_robocasa,
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
import robocasa
from robocasa.utils.robomimic.robomimic_env_utils import create_env
import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils
from robocasa.utils.controller_utils import load_robocasa_controller_config
import robosuite
import h5py
import json


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "prismatic"                    # Model family
    hf_token: str = Path(".hf_token")                       # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    random_llm_weights: bool = False                 # Randomly initialize LLM weights (for ablation studies)

    center_crop: bool = False                        # Center crop? (if trained w/ random crop image aug)
    obs_history: int = 1                             # Number of images to pass in from history
    use_wrist_image: bool = False                    # Use wrist images (doubles the number of input images)
    unnorm_key: str = None

    #################################################################################################################
    # ROBOCASA environment-specific parameters
    #################################################################################################################
    robot: str = None                              
    embodiment: str = None
    task: str = None
    camera: str = None
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 100                   # Number of rollouts per task
    max_steps: int = 600
    use_distractors: bool = True
    obj_groups: str = "obj_set1"
    generative_textures: bool = True
    controller: str = None
    gripper_types: str = "default"

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
    start_seed: int = 1000

    aux_task_types: Optional[str] = None     # Auxiliary task types to query before action prediction
    inference_representation: Optional[str] = None  # Optional paper-named representation predicted before actions
    method: Optional[str] = None             # Deprecated compatibility option for older evaluation commands
    aux_context_freq: int = 1
    obj_xinit_range: float = None
    obj_yinit_range: float = None
    act_horizon: int = 1
    rollout_dir: str = None
    camera_width: int = 320
    camera_height: int = 180
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


def extract_step_k(s):
    match = re.search(r'step-(\d{6})-epoch', s)
    if match:
        step_num = int(match.group(1))
        rounded_k = round(step_num / 1000, 1)
 
        return f"{rounded_k}k".replace(".", "_")
    return None


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

def get_env_config(cfg):
    embodiment = EMBODIMENTS[cfg.embodiment]
    expected = (embodiment.robot, embodiment.gripper, embodiment.camera)
    actual = (cfg.robot, cfg.gripper_types, cfg.camera)
    if actual != expected:
        raise ValueError(
            f"BARX embodiment {cfg.embodiment!r} requires robot, gripper, and "
            f"camera {expected}, received {actual}. Use scripts/evaluate.py to "
            "select them automatically."
        )

    controller_config = load_robocasa_controller_config(
        controller=cfg.controller,
        robot=cfg.robot,
    )

    if cfg.obj_xinit_range and cfg.obj_yinit_range:
        obj_init_range = [cfg.obj_xinit_range, cfg.obj_yinit_range]
    else:
        obj_init_range = None

    camera_names = [cfg.camera]
    if cfg.use_wrist_image:
        camera_names.append("robot0_eye_in_hand")


    # Create argument configuration
    config = {
        "env_name": cfg.task,
        "robots": [cfg.robot],
        "controller_configs": controller_config,
        "use_distractors": cfg.use_distractors,
        "render_camera": cfg.camera,
        "camera_names": camera_names,
        "obj_init_range": obj_init_range,
        "camera_widths": cfg.camera_width,
        "camera_heights": cfg.camera_height,
        "gripper_types": cfg.gripper_types,
    }

    if cfg.generative_textures is True:
        config["generative_textures"] = "100p"

    task_name = TASK_BY_ENVIRONMENT[cfg.task]
    config.update(evaluation_scene_config(task_name, cfg.embodiment))

    ### update config for kitchen envs ###
    if "pnp" in cfg.task.lower() and cfg.obj_groups is not None:
        config.update({"obj_groups": cfg.obj_groups})

    config["translucent_robot"] = False

    # by default use obj instance split A
    config["obj_instance_split"] = OBJECT_INSTANCE_SPLIT

    return config

def robocasa_img_transform(img):
    img = np.transpose(img, (1, 2, 0))
    img = (255*img).astype(np.uint8)
    return img

def eval_single_task(cfg: GenerateConfig, model, log_file) -> None:
    """Evaluate a single robot task"""
    # if cfg.unnorm_key is None:    
    #     cfg.unnorm_key = f"{cfg.robot}_{cfg.task}"

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family in ["openvla", "prismatic"]:
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize environment
    env_config = get_env_config(cfg)

    env = create_env(
        **env_config,
        env_type=1,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
        use_camera_obs=False,
        rng=np.random.default_rng(cfg.start_seed),
    )

    # env = create_env(env_config, cfg.start_seed)

    log_file.write(f"Robocasa task: {cfg.task}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Initialize tracking for single task
    total_episodes = 0
    total_successes = 0
    current_seed = cfg.start_seed

    # Run trials for the task
    for trial_idx in range(cfg.num_trials_per_task):
        print(f"\nTrial {trial_idx + 1}/{cfg.num_trials_per_task}")
        log_file.write(f"\nTrial {trial_idx + 1}/{cfg.num_trials_per_task}\n")

        # Reset environment
        env.env.rng = np.random.default_rng(current_seed)
        env.reset()

        
        ep_meta = env.env.get_ep_meta()
        lang = ep_meta.get("lang", None)
        if lang is not None:
            print(colored(f"Instruction: {lang}", "green"))

        # print the style and layout ids
        print(colored(f"Style ID: {ep_meta['style_id']}", "blue"))
        print(colored(f"Layout ID: {ep_meta['layout_id']}", "blue"))

        # Setup
        t = 0
        replay_images = []
        replay_images_with_bbox = []  # Separate list for bbox images
        replay_wrist_images = []  # Add list for wrist images
        bbox_color_map = {}  # Reset color map for each new video
        has_bbox_predictions = False
        has_ee_pose_predictions = False
        has_motion_predictions = False

        print(f"Starting episode {total_episodes+1}...")
        log_file.write(f"Starting episode {total_episodes+1}...\n")
        
        # Create progress bar for steps
        pbar = tqdm.tqdm(total=cfg.max_steps + cfg.num_steps_wait, desc='Environment steps')

        action_buffer = []
        
        while t < cfg.max_steps + cfg.num_steps_wait:
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_robocasa_dummy_action())
                t += 1
                pbar.update(1)
                continue

            # Get preprocessed image
            cam_key = f"{cfg.camera}_image"
            # img = get_libero_image(obs, resize_size, flip_image=False, key=cam_key, is_robocasa=True)
            img = obs[cam_key]
            img = robocasa_img_transform(img)

            # Save preprocessed image for replay video
            replay_images.append(img)

            # use_wrist_image
            if cfg.use_wrist_image:
                # wrist_img = get_libero_image(obs, resize_size, key="robot0_eye_in_hand_image", flip_image=False, is_robocasa=True)
                wrist_img = obs["robot0_eye_in_hand_image"]
                wrist_img = robocasa_img_transform(img)
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

            ep_meta = env.env.get_ep_meta()
            lang = ep_meta["lang"]

            if len(action_buffer) == 0:
                # Query model to get action
                output = canonicalize_prediction_keys(get_action(
                    cfg,
                    model,
                    observation,
                    lang,
                    aux_task_types=cfg.aux_task_types,
                    processor=processor,
                ))

                actions = output['action']
                if len(actions.shape) == 1:
                    actions = [actions]
                for act in actions[:cfg.act_horizon]:
                    action_buffer.append(act)
            
            action = action_buffer[0]
            action_buffer = action_buffer[1:]
                
            # Save original image
            replay_images.append(img)
            
            # Process image with visualizations
            current_img = img.copy()
            
            # Draw bounding boxes if available
            if 'bounding_box' in output:
                has_bbox_predictions = True
                try:
                    current_img = draw_bbox_on_image(current_img, output['bounding_box'], bbox_color_map)
                except Exception as e:
                    print(f"Error drawing bounding boxes: {e}")
                    print(f"Output: {output['bounding_box']}")
            
            # Draw trajectory if available
            if 'end_effector_trace' in output:
                has_ee_pose_predictions = True
                try:
                    current_img = draw_trajectory_on_image(current_img, output['end_effector_trace'])
                except Exception as e:
                    print(f"Error drawing trajectory: {e}")
                    print(f"Output: {output['end_effector_trace']}")
            
            # Draw motion text if available
            if 'language_motion' in output:
                has_motion_predictions = True
                try:
                    current_img = draw_motion_text_on_image(current_img, output['language_motion'])
                except Exception as e:
                    print(f"Error drawing motion text: {e}")
                    print(f"Output: {output['language_motion']}")
            
            replay_images_with_bbox.append(current_img)

            # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
            action = normalize_gripper_action(action, binarize=True)
            # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
            # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
            if cfg.model_family in ["openvla", "prismatic"]:
                # data for this task is always open gripper, which causes bug in openvla code where
                # gripper dim is always set to 0 (after normalizing) in the data, so do not flip
                if "faucet" not in cfg.task.lower():
                    action = invert_gripper_action(action)
            # Execute action in environment
            env_action = pad_action_robocasa(action.tolist())
            obs, reward, done, info = env.step(env_action)
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
        current_seed += 1

        # Save videos and log results
        save_rollout_video(
            replay_images, total_episodes, success=done, task_description=cfg.task, log_file=log_file, rollout_dir=cfg.rollout_dir
        )
        
        # Add wrist camera video saving if enabled
        if cfg.use_wrist_image:
            save_rollout_video(
                replay_wrist_images, total_episodes, success=done, 
                task_description=f"{cfg.task}_wrist", log_file=log_file, rollout_dir=cfg.rollout_dir
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
                task_description=f"{cfg.task}{viz_suffix}", log_file=log_file, rollout_dir=cfg.rollout_dir
            )

        # Log current results
        success_rate = float(total_successes) / float(total_episodes) * 100
        print(f"Success rate: {success_rate:.1f}%")
        log_file.write(f"Success rate: {success_rate:.1f}%\n")

        # Log to wandb after each episode (only for first trial)
        if cfg.use_wandb:
            log_dict = {
                f"{cfg.task}/success_rate": success_rate,
                f"{cfg.task}/total_successes": total_successes,
                f"{cfg.task}/total_episodes": total_episodes,
                f"{cfg.task}/episode_success": 1.0 if done else 0.0,
            }
            
            # Only log videos for the first trial
            if trial_idx == 0:
                group = "success" if done else "failure"
                log_dict[f"{cfg.task}/{group}/trial_0"] = wandb.Video(
                    np.array(replay_images).transpose(0, 3, 1, 2)
                )
                
                if cfg.use_wrist_image:
                    log_dict[f"{cfg.task}_wrist/{group}/trial_0"] = wandb.Video(
                        np.array(replay_wrist_images).transpose(0, 3, 1, 2)
                    )
                    
                if has_bbox_predictions or has_ee_pose_predictions or has_motion_predictions:
                    viz_suffix = "_with_" + "_".join(
                        x for x in ["bbox", "ee_pose", "motion"] 
                        if (x == "bbox" and has_bbox_predictions) or 
                           (x == "ee_pose" and has_ee_pose_predictions) or
                           (x == "motion" and has_motion_predictions)
                    )
                    log_dict[f"{cfg.task}{viz_suffix}/{group}/trial_0"] = wandb.Video(
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

    step_str = extract_step_k(cfg.pretrained_checkpoint)
    cfg.rollout_dir = cfg.rollout_dir.replace("STEP", step_str)
        
    # Set random seed
    set_seed_everywhere(cfg.seed)

    ObsUtils.OBS_KEYS_TO_MODALITIES = {
        f"{cfg.camera}_image": 'rgb', 
        'robot0_eye_in_hand_image': 'rgb', 
        'mean': 'low_dim', 
        'scale': 'low_dim', 
        'logits': 'low_dim'
    }

    specified_inference_options = sum(
        option is not None for option in (cfg.inference_representation, cfg.method, cfg.aux_task_types)
    )
    if specified_inference_options > 1:
        raise ValueError(
            "Specify only one of --inference_representation, --method, or --aux_task_types."
        )
    if cfg.inference_representation is not None:
        cfg.aux_task_types = [internal_representation_name(cfg.inference_representation)]
    if cfg.method is not None:
        cfg.aux_task_types = representations_for_method(cfg.method)
    elif cfg.aux_task_types is not None:
        cfg.aux_task_types = [
            internal_representation_name(name)
            for name in cfg.aux_task_types.split("->")
            if name
        ]

    # Load model
    model = get_model(cfg)

    # After loading your model, add:
    model = patch_model_for_generation(model)

    if cfg.use_wrist_image:
        model.vision_backbone.image_sequence_len *= 2

    # Move log file initialization before the task loop
    # run_id = f"{cfg.prefix}EVAL-{cfg.run_id}-{DATE_TIME}"
    # if cfg.run_id_note is not None:
    #     run_id += f"--{cfg.run_id_note}"
    # os.makedirs(cfg.local_log_dir, exist_ok=True)
    # local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    # log_file = open(local_log_filepath, "w")

    os.makedirs(cfg.rollout_dir, exist_ok=True)
    log_path = os.path.join(cfg.rollout_dir, "log.txt")
    log_file = open(log_path, "w")
    print(f"Logging to local log file: {log_path}")


    try:
        # Initialize Weights & Biases logging
        if cfg.use_wandb:
            wandb.init(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                name=run_id,
            )

        # [OpenVLA] Set action un-normalization key
        eval_single_task(cfg, model, log_file)

        # Save local log file
        log_file.close()

        # Final wandb logging not needed since we're logging throughout
        if cfg.use_wandb:
            wandb.save(log_path)

    except ZeroDivisionError as e:
        print(f"Caught exception: {e}")
        log_file.write(f"Caught exception: {e}\n")
    finally:
        # Close log file in finally block to ensure it's closed properly
        log_file.close()
        
        # Final wandb logging if needed
        if cfg.use_wandb:
            wandb.save(local_path)


if __name__ == "__main__":
    eval_robocasa()

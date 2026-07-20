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
from termcolor import colored

import draccus
import numpy as np
from tqdm import tqdm
import re

import wandb
import cv2
from matplotlib import colors
import random
from sentence_transformers import SentenceTransformer


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
from experiments.robot.libero.language_motion_converter import LiberoLanguageMotionEpisodeConverter
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
from robosuite.controllers import load_composite_controller_config
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
    # pretrained_checkpoint: Union[str, Path] = "runs/aux--panda+mg_pnp/checkpoints/step-001000-epoch-03-loss=0.1831.pt"     # Pretrained checkpoint path
    pretrained_checkpoint: Union[str, Path] = "runs/aux--mg_pnp/checkpoints/step-250000-epoch-23-loss=0.1868.pt"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    random_llm_weights: bool = False                 # Randomly initialize LLM weights (for ablation studies)

    center_crop: bool = False                        # Center crop? (if trained w/ random crop image aug)
    obs_history: int = 1                             # Number of images to pass in from history
    use_wrist_image: bool = False                    # Use wrist images (doubles the number of input images)
    unnorm_key: str = "mg_pnp"

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id: str = "robocasa_xembod_eval"
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    prefix: str = ''
    data_path: str = '/iliad/u/jenseng/xembod/robocasa_xembod/data/human/PandaOmron/PnPCounterToSink/demo_gentex_im320_heldout.hdf5'

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "prismatic"        # Name of W&B project to log to (use default!)
    wandb_entity: Optional[str] = None          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    start_seed: int = 1000

    aux_context_freq: int = 1
    act_horizon: int = 1
    rollout_dir: str = 'experiments/aux_reps/PnPCounterToSink/Panda/aux--mg_pnp/'
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
        rounded_k = round(step_num / 1000)
        return f"{rounded_k}k"
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


def calculate_iou(box1, box2):
    x1_max, y1_max = np.maximum(box1[:2], box2[:2])
    x2_min, y2_min = np.minimum(box1[2:], box2[2:])
    intersection = np.maximum(0, x2_min - x1_max) * np.maximum(0, y2_min - y1_max)

    if intersection == 0:
        return 0
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    iou = intersection / union
    return iou

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

    
    data = h5py.File(cfg.data_path, "r")
    

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    trace_stride = 8
    trace_window = 40

    metrics = {
        "ee_pose_2D": [],
        "bbox": [],
        "low_level_motion": []
    }

    embedding_model = SentenceTransformer("paraphrase-mpnet-base-v2")
    lm_converter = LiberoLanguageMotionEpisodeConverter()


    for i, key in enumerate(data['data']):
        images = []
        demo = data['data'][key]
        obses = demo['obs']['agentview_rgb'][:]
        bboxes = demo['aux_info']['bboxes_2d']
        obj_name = list(bboxes.keys())[0]
        bboxes = bboxes[obj_name][:]
        eef_pts = demo['aux_info']['eef_normalized_image_pts'][:]

        ep_meta_str = demo.attrs['ep_meta']
        ep_meta = json.loads(ep_meta_str)
        lang = ep_meta['lang']

        actions = demo["actions"][()]
        states = demo["obs"]["ee_states"][()]
        gripper_states = demo["obs"]["gripper_states"][()]

        episode = []
        for action, state, gripper in zip(actions, states, gripper_states):
            episode.append(
                {
                    "observation": {
                        "state": np.asarray(np.concatenate((state, gripper), axis=-1), np.float32),
                    },
                    "action": np.asarray(action[:7], dtype=np.float32), # only for panda
                }
            )

        lms = lm_converter.convert(episode)

        np.asarray(np.concatenate((states[i], gripper_states[i]), axis=-1), np.float32)
        
        print(f"Demo {i + 1}/{len(data['data'])}")
        log_file.write(f"\nDemo {i + 1}/{len(data['data'])}\n")
        
        horizon = 8
        for i in tqdm(range(0, len(obses), 8)):
            # buffering #obs_history images, optionally
            
            # Prepare observations dict
            observation = {
                "full_image": [obses[i]],
            }

            aux_task_labels = {
                "ee_pose_2D": eef_pts[np.arange(i + 1, min(i + trace_window + 1, len(obses)), trace_stride)],
                "bbox": bboxes[i]
            }

            reps = {}
            
            for aux_task_types in ("ee_pose_2D", "bbox", "low_level_motion"):
                output = get_action(
                    cfg,
                    model,
                    observation,
                    lang,
                    aux_task_types=[aux_task_types],
                    processor=processor,
                )
                
                try:
                    if aux_task_types == "ee_pose_2D":
                        label = np.round(aux_task_labels["ee_pose_2D"], 2)
                        pred = np.array(output["ee_pose_2D"][:len(label)])
                        metric = np.mean(np.sum(np.abs(label - pred), axis=-1))

                    elif aux_task_types == "bbox":
                        label = np.round(aux_task_labels["bbox"], 2)
                        pred = np.array(next(iter(output["bbox"].values())))
                        metric = calculate_iou(pred, label)

                    elif aux_task_types == "low_level_motion":
                        label = lms[i]
                        pred = output["low_level_motion"].replace('<|im_end|>', '').strip()
                        label_embedding = embedding_model.encode(label, show_progress_bar=False)
                        pred_embedding = embedding_model.encode(pred, show_progress_bar=False)
                        metric = embedding_model.similarity(pred_embedding, label_embedding).item()

                    metrics[aux_task_types].append(metric)
                except:
                    print(f"failed to compute metric for {aux_task_types} at timestep {i + 1}/{len(obses)}")
            
            # # Process image with visualizations
            # current_img = img.copy()
            
            # # Draw bounding boxes if available
            # if 'bbox' in output:
            #     has_bbox_predictions = True
            #     try:
            #         current_img = draw_bbox_on_image(current_img, output['bbox'], bbox_color_map)
            #     except Exception as e:
            #         print(f"Error drawing bounding boxes: {e}")
            #         print(f"Output: {output['bbox']}")
            
            # # Draw trajectory if available
            # if 'ee_pose_2D' in output:
            #     has_ee_pose_predictions = True
            #     try:
            #         current_img = draw_trajectory_on_image(current_img, output['ee_pose_2D'])
            #     except Exception as e:
            #         print(f"Error drawing trajectory: {e}")
            #         print(f"Output: {output['ee_pose_2D']}")
            
            # # Draw motion text if available
            # if 'low_level_motion' in output:
            #     has_motion_predictions = True
            #     try:
            #         current_img = draw_motion_text_on_image(current_img, output['low_level_motion'])
            #     except Exception as e:
            #         print(f"Error drawing motion text: {e}")
            #         print(f"Output: {output['low_level_motion']}")
            
            # replay_images_with_bbox.append(current_img)


        # Save videos and log results
        # save_rollout_video(
        #     replay_images, total_episodes, success=done, task_description=cfg.task, log_file=log_file, rollout_dir=cfg.rollout_dir
        # )
        
        # if has_bbox_predictions or has_ee_pose_predictions or has_motion_predictions:
        #     viz_suffix = "_with_" + "_".join(
        #         x for x in ["bbox", "ee_pose", "motion"] 
        #         if (x == "bbox" and has_bbox_predictions) or 
        #            (x == "ee_pose" and has_ee_pose_predictions) or
        #            (x == "motion" and has_motion_predictions)
        #     )
        #     save_rollout_video(
        #         replay_images_with_bbox, total_episodes, success=done,
        #         task_description=f"{cfg.task}{viz_suffix}", log_file=log_file, rollout_dir=cfg.rollout_dir
        #     )

        # Log current results
        for rep, metric in metrics.items():
            print(f"{rep} metric: {np.mean(metric)}")
            log_file.write(f"{rep} metric: {np.mean(metric)}\n")

        log_file.flush()
    
    data.close()


@draccus.wrap()
def eval_robocasa(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    step_str = extract_step_k(cfg.pretrained_checkpoint)
        
    # Set random seed
    set_seed_everywhere(cfg.seed)

    # if "sink" in cfg.task.lower():
    #     cfg.camera = "fixed_sink"
    # elif "cab" in cfg.task.lower():
    #     cfg.camera = "fixed_cab"
    # else:
    #     cfg.camera = "robot0_agentview_right"

    # ObsUtils.OBS_KEYS_TO_MODALITIES = {
    #     f"{cfg.camera}_image": 'rgb', 
    #     'robot0_eye_in_hand_image': 'rgb', 
    #     'mean': 'low_dim', 
    #     'scale': 'low_dim', 
    #     'logits': 'low_dim'
    # }

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

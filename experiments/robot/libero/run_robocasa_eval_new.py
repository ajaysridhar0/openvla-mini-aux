"""Run BARX policy rollouts in the pinned RoboCasa evaluation environment."""

import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import draccus
import numpy as np
import tqdm
from termcolor import colored

# Append the repository root so the script also works when invoked from this directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))
import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils  # noqa: E402
from robocasa.utils.robomimic.robomimic_env_utils import create_env  # noqa: E402
from robosuite.controllers import load_composite_controller_config  # noqa: E402

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_robocasa_dummy_action,
    pad_action_robocasa,
    patch_model_for_generation,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor  # noqa: E402
from experiments.robot.robot_utils import (  # noqa: E402
    DATE_TIME,
    get_action,
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
    model_family: str = "prismatic"
    hf_token: Union[str, Path] = Path(".hf_token")
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    random_llm_weights: bool = False

    center_crop: bool = False
    obs_history: int = 1
    use_wrist_image: bool = False
    unnorm_key: Optional[str] = None

    #################################################################################################################
    # RoboCasa environment-specific parameters
    #################################################################################################################
    robot: Optional[str] = None
    task: Optional[str] = None
    num_steps_wait: int = 10
    num_trials_per_task: int = 100
    max_steps: int = 600
    use_distractors: bool = True
    obj_groups: Optional[str] = "obj_set1"
    generative_textures: bool = True
    controller: Optional[str] = None
    gripper_types: str = "default"

    #################################################################################################################
    # Output and runtime parameters
    #################################################################################################################
    run_id: str = "robocasa_xembod_eval"
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    prefix: str = ""

    use_wandb: bool = False
    wandb_project: str = "prismatic"
    wandb_entity: Optional[str] = None

    seed: int = 7
    start_seed: int = 1000

    aux_task_types: Optional[str] = None
    aux_context_freq: int = 1
    obj_xinit_range: Optional[float] = None
    obj_yinit_range: Optional[float] = None
    act_horizon: int = 1
    rollout_dir: Optional[str] = None
    camera_width: int = 320
    camera_height: int = 180
    save_videos: bool = True
    max_videos: int = 10
    result_filename: str = "result.json"
    trials_filename: str = "trials.jsonl"
    protocol_version: str = "barx-robocasa-v1"
    camera: Optional[str] = None
    # fmt: on


END_TEXT = "<|im_end|>"


def extract_step_k(checkpoint: Union[str, Path]) -> Optional[str]:
    """Extract the historical ``2k``-style rollout tag from a checkpoint name."""
    match = re.search(r"step-(\d{6})-epoch", str(checkpoint))
    if match is None:
        return None
    step_num = int(match.group(1))
    rounded_k = round(step_num / 1000, 1)
    return f"{rounded_k}k".replace(".", "_")


def checkpoint_tag(checkpoint: Union[str, Path]) -> str:
    """Return a filesystem-safe tag for paths, Hub IDs, and nonstandard filenames."""
    step_tag = extract_step_k(checkpoint)
    if step_tag is not None:
        return step_tag

    checkpoint_name = str(checkpoint).rstrip("/").rsplit("/", maxsplit=1)[-1]
    checkpoint_name = Path(checkpoint_name).stem
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", checkpoint_name).strip("-._")
    return safe_name or "checkpoint"


def camera_for_robot(robot: str) -> str:
    """Resolve the robot-relative third-person camera used by BARX."""
    robot_name = robot.lower()
    for robot_prefix in ("panda", "kinova", "ur5e", "iiwa", "jaco"):
        if robot_prefix in robot_name:
            return f"{robot_prefix}_agentview_left"
    return "robot0_agentview_left"


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append one durable trial record."""
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, sort_keys=True) + "\n")
        output_file.flush()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically publish the current aggregate result."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    os.replace(temporary_path, path)


def public_config(cfg: GenerateConfig) -> Dict[str, Any]:
    """Serialize evaluation settings without recording credential locations."""
    config = asdict(cfg)
    config.pop("hf_token", None)
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    return config


def build_result(cfg: GenerateConfig, trial_records: List[Dict[str, Any]], status: str) -> Dict[str, Any]:
    """Build a machine-readable aggregate while retaining incomplete or failed trials."""
    completed_records = [record for record in trial_records if record.get("error") is None]
    total_successes = sum(bool(record.get("success")) for record in completed_records)
    completed_trials = len(completed_records)
    success_rate = 100.0 * total_successes / completed_trials if completed_trials else 0.0
    return {
        "status": status,
        "protocol_version": cfg.protocol_version,
        "config": public_config(cfg),
        "requested_trials": cfg.num_trials_per_task,
        "recorded_trials": len(trial_records),
        "completed_trials": completed_trials,
        "total_successes": total_successes,
        "success_rate": success_rate,
        "seeds": [record["seed"] for record in trial_records],
        "trials": trial_records,
    }


def draw_bbox_on_image(img: np.ndarray, bbox_dict: Dict[str, Any], color_map: Dict[str, Any]) -> np.ndarray:
    """Draw normalized bounding boxes with stable per-object colors."""
    img_with_bbox = img.copy()
    height, width = img.shape[:2]

    for obj_name, bbox in bbox_dict.items():
        if obj_name not in color_map:
            color_map[obj_name] = tuple(random.random() for _ in range(3))

        color = color_map[obj_name]
        x1, y1, x2, y2 = bbox
        x1, x2 = int(x1 * width), int(x2 * width)
        y1, y2 = int(y1 * height), int(y2 * height)
        display_color = tuple(int(component * 255) for component in color)

        cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), display_color, 2)
        cv2.putText(img_with_bbox, obj_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, display_color, 1)

    return img_with_bbox


def draw_trajectory_on_image(img: np.ndarray, trajectory_points: Any) -> np.ndarray:
    """Draw normalized end-effector trajectory points."""
    img_with_traj = img.copy()
    height, width = img.shape[:2]
    pixel_points = [(int(x * width), int(y * height)) for x, y in trajectory_points]

    for start, end in zip(pixel_points[:-1], pixel_points[1:]):
        cv2.line(img_with_traj, start, end, (0, 255, 0), 2)
    for point in pixel_points:
        cv2.circle(img_with_traj, point, 3, (255, 0, 0), -1)
    return img_with_traj


def draw_motion_text_on_image(img: np.ndarray, motion_text: str) -> np.ndarray:
    """Draw low-level motion text as a centered subtitle."""
    img_with_text = img.copy()
    height, width = img.shape[:2]
    motion_text = motion_text.replace(END_TEXT, "").strip()

    words = motion_text.split()
    lines = []
    current_line = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 <= 40:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
    if current_line:
        lines.append(" ".join(current_line))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    padding = 8
    line_spacing = 4
    line_sizes = [cv2.getTextSize(line, font, font_scale, font_thickness)[0] for line in lines]
    total_height = sum(text_height + line_spacing for _, text_height in line_sizes)
    y_pos = height - total_height - padding

    for line, (text_width, text_height) in zip(lines, line_sizes):
        x_pos = (width - text_width) // 2
        cv2.rectangle(
            img_with_text,
            (x_pos - padding, y_pos - padding),
            (x_pos + text_width + padding, y_pos + text_height + padding),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            img_with_text,
            line,
            (x_pos, y_pos + text_height),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )
        y_pos += text_height + line_spacing

    return img_with_text


def get_env_config(cfg: GenerateConfig) -> Dict[str, Any]:
    """Build the exact BARX RoboCasa rollout environment configuration."""
    controller_config = load_composite_controller_config(controller=cfg.controller, robot=cfg.robot)
    obj_init_range = None
    if cfg.obj_xinit_range is not None and cfg.obj_yinit_range is not None:
        obj_init_range = [cfg.obj_xinit_range, cfg.obj_yinit_range]

    camera_names = [cfg.camera]
    if cfg.use_wrist_image:
        camera_names.append("robot0_eye_in_hand")

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
    if cfg.generative_textures:
        config["generative_textures"] = "100p"

    if "PnP" in cfg.task or "Mug" in cfg.task:
        layouts = [4, 7, 8]
        if "Panda" in cfg.robot and "Panda" in cfg.gripper_types and cfg.task == "PnPSinkToCounter":
            styles = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]
        else:
            styles = list(range(12))
        bad_combinations = {(8, 3), (8, 5), (8, 6), (8, 9)}
        config["layout_and_style_ids"] = [
            (layout, style) for layout in layouts for style in styles if (layout, style) not in bad_combinations
        ]
    elif cfg.task == "TurnOnSinkFaucet":
        config["layout_ids"] = -1
        config["style_ids"] = [0, 1, 2, 3, 4, 7, 8, 10, 11]
    else:
        raise NotImplementedError(f"Unsupported RoboCasa evaluation task: {cfg.task}")

    if "pnp" in cfg.task.lower() and cfg.obj_groups is not None:
        config["obj_groups"] = cfg.obj_groups
    config["translucent_robot"] = False
    config["obj_instance_split"] = "A"
    return config


def robocasa_img_transform(img: np.ndarray) -> np.ndarray:
    """Convert RoboCasa CHW float observations to HWC uint8 frames."""
    if img.ndim != 3:
        raise ValueError(f"Expected a 3-D image, got shape {img.shape}")
    if img.shape[0] in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))
    if np.issubdtype(img.dtype, np.floating):
        img = img * 255
    return np.clip(img, 0, 255).astype(np.uint8)


def task_succeeded(info: Dict[str, Any]) -> bool:
    """Use RoboCasa's explicit task signal instead of the generic done flag."""
    return bool(info.get("is_success", {}).get("task", False))


def close_environment(env: Any) -> None:
    """Close either the RoboMimic wrapper or its underlying RoboCasa environment."""
    close_method = getattr(env, "close", None)
    if callable(close_method):
        close_method()
        return
    base_env = getattr(env, "env", None)
    close_method = getattr(base_env, "close", None)
    if callable(close_method):
        close_method()


def normalize_model_output(raw_output: Any) -> Dict[str, Any]:
    """Normalize Prismatic and OpenVLA model returns to one dictionary shape."""
    if isinstance(raw_output, dict):
        if "action" not in raw_output:
            raise KeyError("Model output dictionary is missing the action field")
        return raw_output
    return {"action": raw_output}


def should_save_video(cfg: GenerateConfig, episode_index: int) -> bool:
    if not cfg.save_videos:
        return False
    return cfg.max_videos < 0 or episode_index <= cfg.max_videos


def run_trial(
    cfg: GenerateConfig,
    env: Any,
    model: Any,
    processor: Any,
    trial_index: int,
    seed: int,
    log_file: Any,
) -> Dict[str, Any]:
    """Run one deterministic rollout and return its record and replay frames."""
    env.env.rng = np.random.default_rng(seed)
    obs = env.reset()
    ep_meta = env.env.get_ep_meta()
    language_instruction = ep_meta.get("lang", "")
    style_id = ep_meta.get("style_id")
    layout_id = ep_meta.get("layout_id")
    if isinstance(style_id, np.generic):
        style_id = style_id.item()
    if isinstance(layout_id, np.generic):
        layout_id = layout_id.item()

    print(f"\nTrial {trial_index + 1}/{cfg.num_trials_per_task}")
    log_file.write(f"\nTrial {trial_index + 1}/{cfg.num_trials_per_task}\n")
    if language_instruction:
        print(colored(f"Instruction: {language_instruction}", "green"))
    print(colored(f"Style ID: {style_id}", "blue"))
    print(colored(f"Layout ID: {layout_id}", "blue"))

    replay_images = []
    replay_wrist_images = []
    replay_visualizations = []
    bbox_color_map = {}
    visualization_flags = {"bbox": False, "ee_pose": False, "motion": False}
    action_buffer = []
    output: Dict[str, Any] = {}
    success = False
    simulator_done = False
    control_steps = 0

    progress = tqdm.tqdm(total=cfg.max_steps + cfg.num_steps_wait, desc="Environment steps")
    try:
        for _ in range(cfg.num_steps_wait):
            obs, _, simulator_done, info = env.step(get_robocasa_dummy_action(cfg.robot))
            success = task_succeeded(info)
            progress.update(1)
            if success:
                break

        while control_steps < cfg.max_steps and not success:
            image = robocasa_img_transform(obs[f"{cfg.camera}_image"])
            replay_images.append(image)

            if cfg.use_wrist_image:
                wrist_image = robocasa_img_transform(obs["robot0_eye_in_hand_image"])
                replay_wrist_images.append(wrist_image)

            image_history = list(replay_images[-cfg.obs_history :])
            if len(image_history) < cfg.obs_history:
                image_history.extend([replay_images[-1]] * (cfg.obs_history - len(image_history)))

            if cfg.use_wrist_image:
                wrist_history = list(replay_wrist_images[-cfg.obs_history :])
                if len(wrist_history) < cfg.obs_history:
                    wrist_history.extend([replay_wrist_images[-1]] * (cfg.obs_history - len(wrist_history)))
                image_history = [frame for pair in zip(image_history, wrist_history) for frame in pair]

            observation = {
                "full_image": image_history,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }

            if not action_buffer:
                output = normalize_model_output(
                    get_action(
                        cfg,
                        model,
                        observation,
                        language_instruction,
                        aux_task_types=cfg.aux_task_types,
                        processor=processor,
                    )
                )
                actions = np.asarray(output["action"])
                if actions.ndim == 1:
                    actions = actions[None, :]
                if actions.shape[0] == 0:
                    raise ValueError("Model returned an empty action chunk")
                action_buffer.extend(np.asarray(action).copy() for action in actions[: cfg.act_horizon])

            action = action_buffer.pop(0)
            visualization = image.copy()
            if "bbox" in output:
                visualization_flags["bbox"] = True
                visualization = draw_bbox_on_image(visualization, output["bbox"], bbox_color_map)
            if "ee_pose_2D" in output:
                visualization_flags["ee_pose"] = True
                visualization = draw_trajectory_on_image(visualization, output["ee_pose_2D"])
            if "low_level_motion" in output:
                visualization_flags["motion"] = True
                visualization = draw_motion_text_on_image(visualization, output["low_level_motion"])
            replay_visualizations.append(visualization)

            action = normalize_gripper_action(np.asarray(action).copy(), binarize=True)
            if cfg.model_family in ("openvla", "prismatic") and "faucet" not in cfg.task.lower():
                action = invert_gripper_action(action)
            env_action = pad_action_robocasa(action.tolist(), cfg.robot)
            obs, _, simulator_done, info = env.step(env_action)
            success = task_succeeded(info)
            control_steps += 1
            progress.update(1)
    finally:
        progress.close()

    return {
        "record": {
            "trial_index": trial_index,
            "seed": seed,
            "success": success,
            "control_steps": control_steps,
            "settling_steps": cfg.num_steps_wait,
            "simulator_done": bool(simulator_done),
            "layout_id": layout_id,
            "style_id": style_id,
            "language_instruction": language_instruction,
            "error": None,
        },
        "replay_images": replay_images,
        "replay_wrist_images": replay_wrist_images,
        "replay_visualizations": replay_visualizations,
        "visualization_flags": visualization_flags,
    }


def eval_single_task(
    cfg: GenerateConfig,
    model: Any,
    log_file: Any,
    trials_path: Path,
    result_path: Path,
    wandb_module: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate one model/task pair and persist every completed trial."""
    if cfg.model_family in ("openvla", "prismatic"):
        if cfg.unnorm_key not in model.norm_stats:
            raise KeyError(f"Action un-normalization key {cfg.unnorm_key!r} is not present in model norm_stats")

    processor = get_processor(cfg) if cfg.model_family == "openvla" else None
    env = create_env(
        **get_env_config(cfg),
        env_type=1,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
        use_camera_obs=False,
        rng=np.random.default_rng(cfg.start_seed),
    )
    log_file.write(f"Robocasa task: {cfg.task}\n")

    trial_records = []
    try:
        for trial_index in range(cfg.num_trials_per_task):
            seed = cfg.start_seed + trial_index
            try:
                trial_output = run_trial(cfg, env, model, processor, trial_index, seed, log_file)
            except Exception as exc:
                error_record = {
                    "trial_index": trial_index,
                    "seed": seed,
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                trial_records.append(error_record)
                append_jsonl(trials_path, error_record)
                write_json(result_path, build_result(cfg, trial_records, status="failed"))
                raise

            record = trial_output["record"]
            trial_records.append(record)
            append_jsonl(trials_path, record)

            episode_index = len(trial_records)
            if should_save_video(cfg, episode_index) and trial_output["replay_images"]:
                save_rollout_video(
                    trial_output["replay_images"],
                    episode_index,
                    success=record["success"],
                    task_description=cfg.task,
                    log_file=log_file,
                    rollout_dir=cfg.rollout_dir,
                )
                if cfg.use_wrist_image:
                    save_rollout_video(
                        trial_output["replay_wrist_images"],
                        episode_index,
                        success=record["success"],
                        task_description=f"{cfg.task}_wrist",
                        log_file=log_file,
                        rollout_dir=cfg.rollout_dir,
                    )

                flags = trial_output["visualization_flags"]
                if any(flags.values()):
                    suffix = "_with_" + "_".join(name for name, enabled in flags.items() if enabled)
                    save_rollout_video(
                        trial_output["replay_visualizations"],
                        episode_index,
                        success=record["success"],
                        task_description=f"{cfg.task}{suffix}",
                        log_file=log_file,
                        rollout_dir=cfg.rollout_dir,
                    )

            current_result = build_result(cfg, trial_records, status="running")
            write_json(result_path, current_result)
            print(f"Success rate: {current_result['success_rate']:.1f}%")
            log_file.write(f"Success rate: {current_result['success_rate']:.1f}%\n")
            log_file.flush()

            if cfg.use_wandb:
                if wandb_module is None:
                    raise RuntimeError("W&B logging was requested but W&B was not initialized")
                wandb_module.log(
                    {
                        f"{cfg.task}/success_rate": current_result["success_rate"],
                        f"{cfg.task}/total_successes": current_result["total_successes"],
                        f"{cfg.task}/total_episodes": current_result["completed_trials"],
                        f"{cfg.task}/episode_success": float(record["success"]),
                    }
                )
    finally:
        close_environment(env)

    final_result = build_result(cfg, trial_records, status="completed")
    write_json(result_path, final_result)
    return final_result


def validate_config(cfg: GenerateConfig) -> None:
    checkpoint = str(cfg.pretrained_checkpoint)
    if not checkpoint:
        raise ValueError("pretrained_checkpoint must be provided")
    if "image_aug" in checkpoint and not cfg.center_crop:
        raise ValueError("center_crop must be enabled for a checkpoint trained with image augmentation")
    if cfg.load_in_8bit and cfg.load_in_4bit:
        raise ValueError("load_in_8bit and load_in_4bit cannot both be enabled")
    if not cfg.robot or not cfg.task or not cfg.unnorm_key:
        raise ValueError("robot, task, and unnorm_key must be provided")
    if cfg.num_trials_per_task <= 0 or cfg.max_steps <= 0:
        raise ValueError("num_trials_per_task and max_steps must be positive")
    if cfg.num_steps_wait < 0:
        raise ValueError("num_steps_wait cannot be negative")
    if cfg.obs_history <= 0 or cfg.act_horizon <= 0:
        raise ValueError("obs_history and act_horizon must be positive")
    if cfg.max_videos < -1:
        raise ValueError("max_videos must be -1 or non-negative")


@draccus.wrap()
def eval_robocasa(cfg: GenerateConfig) -> None:
    """Load one checkpoint and run the configured deterministic RoboCasa trials."""
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    cfg.camera = camera_for_robot(cfg.robot)

    if cfg.rollout_dir is None:
        cfg.rollout_dir = str(
            Path("experiments/rollouts") / cfg.task / cfg.robot / checkpoint_tag(cfg.pretrained_checkpoint) / "act"
        )
    cfg.rollout_dir = cfg.rollout_dir.replace("STEP", checkpoint_tag(cfg.pretrained_checkpoint))
    rollout_dir = Path(cfg.rollout_dir)
    rollout_dir.mkdir(parents=True, exist_ok=True)

    result_path = rollout_dir / cfg.result_filename
    trials_path = rollout_dir / cfg.trials_filename
    log_path = rollout_dir / "log.txt"
    trials_path.write_text("", encoding="utf-8")
    write_json(result_path, build_result(cfg, [], status="starting"))

    ObsUtils.OBS_KEYS_TO_MODALITIES = {
        f"{cfg.camera}_image": "rgb",
        "robot0_eye_in_hand_image": "rgb",
        "mean": "low_dim",
        "scale": "low_dim",
        "logits": "low_dim",
    }

    if cfg.aux_task_types is None or cfg.aux_task_types.lower() in ("", "none", "null"):
        cfg.aux_task_types = None
    else:
        cfg.aux_task_types = cfg.aux_task_types.split("->")

    run_id = f"{cfg.prefix}EVAL-{cfg.run_id}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"

    log_file = log_path.open("w", encoding="utf-8")
    print(f"Logging to local log file: {log_path}")
    wandb_module = None
    wandb_run = None
    try:
        if cfg.use_wandb:
            import wandb

            wandb_module = wandb
            wandb_run = wandb_module.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

        model = patch_model_for_generation(get_model(cfg))
        if cfg.use_wrist_image:
            model.vision_backbone.image_sequence_len *= 2

        eval_single_task(cfg, model, log_file, trials_path, result_path, wandb_module=wandb_module)
        if cfg.use_wandb:
            wandb_module.save(str(log_path))
    except Exception as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}")
        log_file.write(f"Evaluation failed: {type(exc).__name__}: {exc}\n")
        log_file.flush()
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            result = build_result(cfg, [], status="failed")
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        write_json(result_path, result)
        raise
    finally:
        log_file.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    eval_robocasa()

"""Replay RoboCasa simulator states into BARX RGB and auxiliary HDF5 observations."""

import argparse
import json
import os
import re
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import transforms3d
from notebooks.utils import get_aabbs_for_body, get_corners
from robocasa.utils.robomimic import robomimic_dataset_utils as DatasetUtils
from robocasa.utils.robomimic import robomimic_env_utils as EnvUtils
from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera
from tqdm import tqdm


def object_for_auxiliary_labels(env: Any) -> tuple:
    """Return the language object name and simulator body used for 2-D boxes."""
    env_name = env.name.lower()
    if "pnp" in env_name or "flip" in env_name:
        return env.env.get_obj_lang(), "obj_main"
    if "faucet" in env_name:
        pattern = re.compile(r"^sink_(.+?)_group_handle$")
        body_name = next((name for name in env.env.sim.model.body_names if pattern.match(name)), None)
        if body_name is None:
            raise RuntimeError("Could not locate the sink handle body")
        return "handle", body_name
    raise ValueError(f"Unsupported BARX task for auxiliary labels: {env.name}")


def extract_trajectory(env: Any, initial_state: Dict[str, Any], states: np.ndarray, actions: np.ndarray, cameras):
    """Replay one trajectory and collect the exact BARX intermediate schema."""
    env.reset()
    env.reset_to(initial_state)
    ep_meta = json.loads(initial_state["ep_meta"])
    ep_meta["cam_configs"] = deepcopy(env.env._cam_configs)
    initial_state["ep_meta"] = json.dumps(ep_meta, indent=2)
    object_name, body_name = object_for_auxiliary_labels(env)

    trajectory = {
        "obs": {
            key: []
            for key in (
                "agentview_rgb",
                "eye_in_hand_rgb",
                "ee_pos",
                "ee_ori",
                "ee_states",
                "gripper_states",
                "joint_states",
            )
        },
        "rewards": [],
        "dones": [],
        "actions": np.asarray(actions),
        "states": np.asarray(states),
        "initial_state_dict": initial_state,
        "aux_info": {"bboxes_2d": {object_name: []}, "eef_normalized_image_pts": []},
    }
    for timestep, state in enumerate(states):
        obs = deepcopy(env.reset_to({"states": state}))
        height, width = env.env.camera_heights[0], env.env.camera_widths[0]
        transform = get_camera_transform_matrix(env.env.sim, cameras[0], height, width)
        project = partial(
            project_points_from_world_to_camera,
            world_to_camera_transform=transform,
            camera_height=height,
            camera_width=width,
        )
        euler = transforms3d.euler.quat2euler(obs["robot0_eef_quat"], axes="sxyz")
        trajectory["obs"]["agentview_rgb"].append(obs[f"{cameras[0]}_image"])
        trajectory["obs"]["eye_in_hand_rgb"].append(obs[f"{cameras[1]}_image"])
        trajectory["obs"]["ee_pos"].append(obs["robot0_eef_pos"])
        trajectory["obs"]["ee_ori"].append(euler)
        trajectory["obs"]["ee_states"].append(np.concatenate([obs["robot0_eef_pos"], euler]))
        trajectory["obs"]["gripper_states"].append(obs["robot0_gripper_qpos"][:2])
        trajectory["obs"]["joint_states"].append(obs["robot0_joint_vel"])
        trajectory["rewards"].append(env.get_reward())
        trajectory["dones"].append(np.uint8(env.is_success()["task"] or timestep == len(states) - 1))

        eef_pixel = project(obs["robot0_eef_pos"])
        trajectory["aux_info"]["eef_normalized_image_pts"].append([eef_pixel[1] / width, eef_pixel[0] / height])
        try:
            bounds = get_aabbs_for_body(env.env, body_name)[body_name]
            corners = get_corners(bounds[None])[0]
            pixels = project(corners)
            minimum, maximum = np.min(pixels, axis=0), np.max(pixels, axis=0)
            box = [minimum[1] / width, minimum[0] / height, maximum[1] / width, maximum[0] / height]
        except Exception as exc:
            print(f"Warning: bbox projection failed at step {timestep}: {exc}")
            box = [0.0, 0.0, 0.0, 0.0]
        trajectory["aux_info"]["bboxes_2d"][object_name].append(box)

    trajectory["obs"] = {key: np.asarray(value) for key, value in trajectory["obs"].items()}
    trajectory["rewards"] = np.asarray(trajectory["rewards"])
    trajectory["dones"] = np.asarray(trajectory["dones"], dtype=np.uint8)
    trajectory["aux_info"]["eef_normalized_image_pts"] = np.asarray(
        trajectory["aux_info"]["eef_normalized_image_pts"], dtype=np.float32
    )
    trajectory["aux_info"]["bboxes_2d"] = {
        key: np.asarray(value, dtype=np.float32) for key, value in trajectory["aux_info"]["bboxes_2d"].items()
    }
    return trajectory


def write_demo(output_group, source_demo, demo_id: str, trajectory, no_compress: bool) -> int:
    demo_group = output_group.create_group(demo_id)
    for key in ("actions", "states", "rewards", "dones"):
        demo_group.create_dataset(key, data=trajectory[key])
    obs_group = demo_group.create_group("obs")
    compression = None if no_compress else "gzip"
    for key, value in trajectory["obs"].items():
        obs_group.create_dataset(key, data=value, compression=compression)
    aux_group = demo_group.create_group("aux_info")
    bbox_group = aux_group.create_group("bboxes_2d")
    for key, value in trajectory["aux_info"]["bboxes_2d"].items():
        bbox_group.create_dataset(key, data=value)
    aux_group.create_dataset("eef_normalized_image_pts", data=trajectory["aux_info"]["eef_normalized_image_pts"])
    if "action_dict" in source_demo:
        source_demo.file.copy(source_demo["action_dict"], demo_group, name="action_dict")
    demo_group.attrs["model_file"] = trajectory["initial_state_dict"]["model"]
    demo_group.attrs["ep_meta"] = trajectory["initial_state_dict"]["ep_meta"]
    demo_group.attrs["num_samples"] = len(trajectory["actions"])
    return len(trajectory["actions"])


def render_dataset(args: argparse.Namespace) -> None:
    source_path = args.dataset.resolve()
    output_path = source_path.with_name(args.output_name)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=str(source_path))
    if args.generative_textures:
        env_meta["env_kwargs"]["generative_textures"] = "100p"
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=[args.camera, args.wrist_camera],
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=False,
    )

    try:
        with h5py.File(source_path, "r") as source, h5py.File(temporary_path, "w") as output:
            output_data = output.create_group("data")
            demo_ids = sorted(source["data"], key=lambda name: int(name.rsplit("_", maxsplit=1)[-1]))
            if args.max_demos is not None:
                demo_ids = demo_ids[: args.max_demos]
            total = 0
            for demo_index, demo_id in enumerate(tqdm(demo_ids, desc="Rendering demos")):
                source_demo = source["data"][demo_id]
                states = source_demo["states"][()]
                initial_state = {
                    "states": states[0],
                    "model": source_demo.attrs["model_file"],
                    "ep_meta": source_demo.attrs["ep_meta"],
                }
                env.env.rng = np.random.default_rng(demo_index)
                trajectory = extract_trajectory(
                    env, initial_state, states, source_demo["actions"][()], [args.camera, args.wrist_camera]
                )
                total += write_demo(output_data, source_demo, demo_id, trajectory, args.no_compress)
            if "mask" in source:
                source.copy("mask", output)
            output_data.attrs["total"] = total
            output_data.attrs["env_args"] = json.dumps(env.serialize(), indent=2)
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    print(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--camera", required=True, help="Robot-relative third-person camera.")
    parser.add_argument("--wrist-camera", default="robot0_eye_in_hand")
    parser.add_argument("--camera-height", type=int, default=180)
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--output-name", default="demo_gentex_im320.hdf5")
    parser.add_argument("--no-generative-textures", action="store_false", dest="generative_textures", default=True)
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument("--max-demos", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    render_dataset(parse_args())

"""Construction of BARX paper evaluation environments."""

from __future__ import annotations

from barx.benchmark import (
    EMBODIMENTS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    OBJECT_GROUP,
    OBJECT_INSTANCE_SPLIT,
    TASKS,
    evaluation_scene_config,
)


def build_environment_config(
    task: str,
    embodiment: str,
    *,
    use_wrist_image: bool = False,
) -> dict[str, object]:
    """Return the exact RoboCasa kwargs used for BARX paper evaluation."""

    from robocasa.utils.controller_utils import load_robocasa_controller_config

    task_spec = TASKS[task]
    embodiment_spec = EMBODIMENTS[embodiment]
    camera_names = [embodiment_spec.camera]
    if use_wrist_image:
        camera_names.append("robot0_eye_in_hand")

    config: dict[str, object] = {
        "env_name": task_spec.environment,
        "robots": [embodiment_spec.robot],
        "controller_configs": load_robocasa_controller_config(
            robot=embodiment_spec.robot
        ),
        "use_distractors": True,
        "render_camera": embodiment_spec.camera,
        "camera_names": camera_names,
        "camera_widths": IMAGE_WIDTH,
        "camera_heights": IMAGE_HEIGHT,
        "gripper_types": embodiment_spec.gripper,
        "generative_textures": "100p",
        "translucent_robot": False,
        "obj_instance_split": OBJECT_INSTANCE_SPLIT,
    }
    if task.startswith("pnp_"):
        config["obj_groups"] = OBJECT_GROUP
    config.update(evaluation_scene_config(task, embodiment))
    return config


def initialize_observation_utils(environment_config: dict[str, object]) -> None:
    """Register the image keys emitted by a BARX RoboCasa environment."""

    import robocasa.utils.robomimic.robomimic_obs_utils as obs_utils

    camera_names = environment_config["camera_names"]
    if isinstance(camera_names, str):
        camera_names = [camera_names]
    obs_utils.initialize_obs_utils_with_obs_specs(
        {
            "obs": {
                "low_dim": [],
                "rgb": [f"{camera_name}_image" for camera_name in camera_names],
            }
        }
    )

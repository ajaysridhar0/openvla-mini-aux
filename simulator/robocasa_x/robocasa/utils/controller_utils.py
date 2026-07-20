"""RoboCasa-X controller configuration without patching robosuite."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from robosuite.controllers import load_composite_controller_config


_CONTROLLER_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "controllers"
    / "barx_omron.json"
)


def is_omron_robot(robot: Optional[str]) -> bool:
    """Return whether a robot uses the RoboCasa-X Omron mobile base."""
    return bool(robot and "omron" in robot.lower())


def load_robocasa_controller_config(
    controller: Optional[str] = None,
    robot: Optional[str] = None,
) -> dict:
    """Load the unified RoboCasa-X controller or an upstream controller.

    Every Omron embodiment uses the same action order:
    ``arm(6), gripper(1), base(3), torso(1), mode(1)``. Existing dataset
    metadata may still carry the historical per-embodiment ordering; callers
    that replay such metadata should pass it through unchanged.
    """
    if controller is None and is_omron_robot(robot):
        controller = str(_CONTROLLER_PATH)
    return load_composite_controller_config(controller=controller, robot=robot)

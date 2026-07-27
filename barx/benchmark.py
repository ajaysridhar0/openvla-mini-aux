"""Frozen task and embodiment protocol for the BARX simulation benchmark."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbodimentSpec:
    paper_name: str
    robot: str
    gripper: str
    camera: str


@dataclass(frozen=True)
class TaskSpec:
    paper_name: str
    environment: str
    max_steps: int


EMBODIMENTS = {
    "iiwa": EmbodimentSpec(
        paper_name="IIWA",
        robot="IIWAOmron",
        gripper="default",
        camera="barx_iiwa_agentview",
    ),
    "kinova3": EmbodimentSpec(
        paper_name="Kinova3",
        robot="Kinova3Omron",
        gripper="default",
        camera="barx_kinova3_agentview",
    ),
    "ur5e": EmbodimentSpec(
        paper_name="UR5e",
        robot="UR5eOmron",
        gripper="default",
        camera="barx_ur5e_agentview",
    ),
    "panda": EmbodimentSpec(
        paper_name="Panda",
        robot="PandaOmron",
        gripper="Robotiq85Gripper",
        camera="barx_panda_agentview",
    ),
    "panda_og": EmbodimentSpec(
        paper_name="Panda-OG",
        robot="PandaOmron",
        gripper="PandaGripper",
        camera="barx_panda_agentview",
    ),
    "jaco": EmbodimentSpec(
        paper_name="Jaco",
        robot="JacoOmron",
        gripper="default",
        camera="barx_jaco_agentview",
    ),
}

TASKS = {
    "pnp_counter_to_sink": TaskSpec(
        paper_name="PnP Counter to Sink",
        environment="XPnPCounterToSink",
        max_steps=600,
    ),
    "pnp_sink_to_counter": TaskSpec(
        paper_name="PnP Sink to Counter",
        environment="XPnPSinkToCounter",
        max_steps=650,
    ),
    "turn_on_sink_faucet": TaskSpec(
        paper_name="Turn On Sink Faucet",
        environment="XTurnOnSinkFaucet",
        max_steps=500,
    ),
    "flip_mug_upright": TaskSpec(
        paper_name="Flip Mug Upright",
        environment="XFlipMugUpright",
        max_steps=500,
    ),
}

TASK_BY_ENVIRONMENT = {spec.environment: name for name, spec in TASKS.items()}

EVALUATION_EPISODES = 100
EVALUATION_START_SEED = 1000
EVALUATION_GLOBAL_SEED = 7
ACTION_HORIZON = 8
SETTLE_STEPS = 10
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 180
MIN_TARGET_VISIBLE_PIXELS = 25
OBJECT_GROUP = "obj_set1"
OBJECT_INSTANCE_SPLIT = "A"
OBJECT_CATEGORIES = frozenset(
    {
        "apple",
        "banana",
        "can",
        "carrot",
        "cucumber",
        "lemon",
        "orange",
        "sponge",
    }
)

_PNP_LAYOUTS = (4, 7, 8)
_PNP_STYLES = tuple(range(12))
_PNP_EXCLUDED_PAIRS = {(8, 3), (8, 5), (8, 6), (8, 9)}
_FAUCET_STYLES = (0, 1, 2, 3, 4, 7, 8, 10, 11)


def evaluation_scene_config(task: str, embodiment: str) -> dict[str, object]:
    """Return the exact scene distribution used by the paper evaluator."""

    if task in {"pnp_counter_to_sink", "pnp_sink_to_counter", "flip_mug_upright"}:
        styles = _PNP_STYLES
        # Style 4 was outside the historical Panda-OG condition set for this
        # task because the Franka Hand could not reach the sampled goal.
        if task == "pnp_sink_to_counter" and embodiment == "panda_og":
            styles = tuple(style for style in styles if style != 4)
        pairs = [
            (layout, style)
            for layout in _PNP_LAYOUTS
            for style in styles
            if (layout, style) not in _PNP_EXCLUDED_PAIRS
        ]
        return {"layout_and_style_ids": pairs}

    if task == "turn_on_sink_faucet":
        return {"layout_ids": -1, "style_ids": list(_FAUCET_STYLES)}

    raise KeyError(f"Unknown BARX task: {task}")

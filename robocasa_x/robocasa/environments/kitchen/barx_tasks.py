"""RoboCasa task variants used by the BARX simulation benchmark.

The classes in this module intentionally isolate the task and embodiment
changes described in the BARX paper. The original RoboCasa task classes remain
registered under their original names and retain their original defaults.
"""

from __future__ import annotations

from string import digits

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.utils.transform_utils import quat2mat

from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen, OU
from robocasa.environments.kitchen.single_stage.kitchen_pnp import (
    PnPCounterToSink,
    PnPSinkToCounter,
)
from robocasa.environments.kitchen.single_stage.kitchen_sink import (
    ManipulateSinkFaucet,
)
from robocasa.utils import camera_utils as CamUtils


# Values used by the released data generation and paper evaluation code.
BARX_ROBOT_POSITION_OFFSETS = {
    "IIWAOmron": [0.4, -0.1, 0.0],
    "UR5eOmron": [0.4, -0.1, 0.2],
    "PandaOmron": [0.4, -0.06, 0.2],
    "Kinova3Omron": [0.4, 0.0, 0.1],
    "JacoOmron": [0.4, 0.0, 0.2],
}

BARX_ROBOT_INITIAL_QPOS = {
    "PandaOmron": [
        -0.0322107,
        -0.87780094,
        0.04876096,
        -2.59347324,
        0.0378028,
        1.71482307,
        0.77941175,
    ],
    "IIWAOmron": [
        0.01115337,
        -0.08399538,
        0.01529279,
        -1.22063772,
        0.00151392,
        2.00516437,
        0.02812827,
    ],
    "Kinova3Omron": [
        -0.05554795,
        -0.17629911,
        -0.0708673,
        1.4654714,
        -0.01258805,
        1.8501173,
        -1.69937106,
    ],
    "UR5eOmron": [
        -0.38866802,
        -2.02017087,
        1.83610264,
        -1.38660839,
        -1.57096943,
        -1.95933365,
    ],
    "SawyerOmron": [
        -0.08429812,
        -1.6291641,
        -0.4312483,
        2.20450107,
        -0.02849383,
        0.98969552,
        -1.89593991,
    ],
    "JacoOmron": [
        3.1776479,
        2.95208179,
        -0.02913301,
        1.20341971,
        -0.0055724,
        4.53421693,
        3.14988761,
    ],
}

BARX_FLIP_MUG_EXCLUDED_INSTANCES = ("mug_0", "mug_7", "mug_10", "mug_12")


class BARXTaskMixin:
    """Apply BARX embodiment alignment without changing RoboCasa defaults."""

    _barx_lock_omron_joints = True

    def __init__(self, *args, robot_pos_offsets=None, robot_init_qpos=None, **kwargs):
        position_offsets = dict(BARX_ROBOT_POSITION_OFFSETS)
        if robot_pos_offsets is not None:
            position_offsets.update(robot_pos_offsets)

        initial_qpos = dict(BARX_ROBOT_INITIAL_QPOS)
        if robot_init_qpos is not None:
            initial_qpos.update(robot_init_qpos)

        super().__init__(
            *args,
            robot_pos_offsets=position_offsets,
            robot_init_qpos=initial_qpos,
            **kwargs,
        )

    def get_obj_lang(self, *args, **kwargs):
        """Match the paper instructions by removing asset-number suffixes."""

        language = super().get_obj_lang(*args, **kwargs)
        return language.translate({ord(char): None for char in digits}).strip()

    def _load_model(self):
        """Apply the paper's zero-height Omron torso initialization."""

        super()._load_model()
        for robot in self.robots:
            robot_name = robot.robot_model.__class__.__name__
            if robot_name in BARX_ROBOT_INITIAL_QPOS:
                robot.init_torso_qpos = np.array([0.0])

    def set_cameras(self):
        """Install canonical paper cameras only for X task variants."""

        super().set_cameras()
        self._cam_configs.update(CamUtils.BARX_CAMERA_CONFIGS)


class XPnPCounterToSink(BARXTaskMixin, PnPCounterToSink):
    """BARX variation of RoboCasa's *PnP Counter to Sink* task."""

    def __init__(self, *args, obj_init_range=None, **kwargs):
        self.barx_obj_init_range = obj_init_range or [0.30, 0.40]
        super().__init__(*args, **kwargs)

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        cfgs[0]["placement"]["size"] = self.barx_obj_init_range

        if not self.use_distractors:
            return cfgs[:1]

        if self.obj_groups != "all":
            cfgs[1]["exclude_obj_groups"] = self.obj_groups
        return cfgs

    def _check_success(self):
        obj_in_sink = OU.obj_inside_of(self, "obj", self.sink, partial_check=True)
        gripper_obj_far = OU.gripper_obj_far(self, th=0.1)
        return obj_in_sink and gripper_obj_far


class XPnPSinkToCounter(BARXTaskMixin, PnPSinkToCounter):
    """BARX variation of RoboCasa's *PnP Sink to Counter* task."""

    def __init__(self, *args, obj_init_range=None, **kwargs):
        self.barx_obj_init_range = obj_init_range or [0.25, 0.25]
        super().__init__(*args, **kwargs)

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        cfgs[0]["placement"]["size"] = self.barx_obj_init_range

        # BARX places the goal receptacle in the same compact region used by
        # the paper's data generation code.
        cfgs[1]["placement"]["sample_region_kwargs"] = {"ref": self.sink}
        cfgs[1]["placement"]["size"] = (0.30, 0.30)

        if not self.use_distractors:
            return cfgs[:2]

        cfgs[2]["exclude_obj_groups"] = ("container",)
        cfgs[2]["placement"]["sample_region_kwargs"] = {"ref": self.sink}
        return cfgs

    def _check_success(self):
        obj_in_receptacle = OU.check_obj_in_receptacle(self, "obj", "container")
        receptacle_on_counter = self.check_contact(
            self.objects["container"], self.counter
        )
        gripper_obj_far = OU.gripper_obj_far(self, th=0.1)
        return obj_in_receptacle and receptacle_on_counter and gripper_obj_far


class XTurnOnSinkFaucet(BARXTaskMixin, ManipulateSinkFaucet):
    """BARX variation of RoboCasa's *Turn On Sink Faucet* task."""

    def __init__(self, *args, **kwargs):
        super().__init__(behavior="turn_on", *args, **kwargs)


class XFlipMugUpright(BARXTaskMixin, Kitchen):
    """The *Flip Mug Upright* task introduced by BARX."""

    def __init__(self, *args, obj_init_range=None, **kwargs):
        self.obj_groups = "mug"
        self.barx_obj_init_range = obj_init_range or [0.30, 0.40]
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", {"id": FixtureType.SINK})
        self.counter = self.register_fixture_ref(
            "counter", {"id": FixtureType.COUNTER, "ref": self.sink}
        )
        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"flip the {self.get_obj_lang()} on the counter upright"
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = [
            {
                "name": "obj",
                "obj_groups": self.obj_groups,
                "exclude_obj_instances": BARX_FLIP_MUG_EXCLUDED_INSTANCES,
                "graspable": True,
                "washable": True,
                "placement": {
                    "fixture": self.counter,
                    "sample_region_kwargs": {"ref": self.sink},
                    "size": self.barx_obj_init_range,
                    "pos": ("ref", -1.0),
                    "rotation": {
                        "y": np.pi / 2,
                        "z": [3 * np.pi / 4, 5 * np.pi / 4],
                    },
                },
            }
        ]

        if self.use_distractors:
            cfgs.extend(
                [
                    {
                        "name": "distr_counter",
                        "obj_groups": "all",
                        "exclude_obj_groups": self.obj_groups,
                        "placement": {
                            "fixture": self.counter,
                            "sample_region_kwargs": {"ref": self.sink},
                            "size": (0.30, 0.30),
                            "pos": ("ref", -1.0),
                            "offset": (0.0, 0.30),
                        },
                    },
                    {
                        "name": "distr_sink",
                        "obj_groups": "all",
                        "washable": True,
                        "placement": {
                            "fixture": self.sink,
                            "size": (0.25, 0.25),
                            "pos": (0.0, 1.0),
                        },
                    },
                ]
            )

        return cfgs

    def _check_success(self):
        obj_on_counter = OU.check_obj_fixture_contact(self, "obj", self.counter)
        gripper_obj_far = OU.gripper_obj_far(self, th=0.1)

        quaternion = T.convert_quat(
            np.asarray(self.sim.data.body_xquat[self.obj_body_id["obj"]]),
            to="xyzw",
        )
        local_up = quat2mat(quaternion)[:, 2]
        upright = np.dot(local_up, np.array([0.0, 0.0, 1.0])) > 0.95
        return obj_on_counter and gripper_obj_far and upright

from robocasa.environments.kitchen.kitchen import *
from robosuite.utils.transform_utils import quat2mat


class FlipMugUpright(Kitchen):
    """
    Class encapsulating the atomic flip mug task

    Args:
        obj_groups (str): Object groups to sample the target object from.
    """

    def __init__(self, *args, **kwargs):
        self.obj_groups = "mug"
        super().__init__(
            robot_pos_offsets={
                "IIWAOmron": [0.4, -0.1, 0], # final
                "UR5eOmron": [0.4, -0.1, 0.2], # final
                "PandaOmron": [0.4, -0.06, 0.2], # final
                "Kinova3Omron": [0.4, 0, 0.1], # final
                "JacoOmron": [0.4, 0, 0.2], # final
            }, 
            *args, 
            **kwargs
        )

        if self.obj_init_range is None:
            self.obj_init_range = [0.30, 0.40]

    def _get_placement_region_kwargs(self, fixture, ref=None):
        """
        Helper method to get placement region kwargs based on robot type.
        For UR5eOmron, restricts placements to the left side of fixtures.
        """
        robot_class_name = self.robots[0].robot_model.__class__.__name__
        
        # Base kwargs
        kwargs = {}
        if ref is not None:
            kwargs["ref"] = ref
        
        return kwargs
        
    def _setup_kitchen_references(self):
        """
        Setup the kitchen references:
        """
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref(
            "sink",
            dict(id=FixtureType.SINK),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.sink),
        )
        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        """
        Get the episode metadata for the counter to sink pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"flip the {obj_lang} on the counter upright"
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the counter to sink pick and place task.
        Puts the target object in the front area of the counter. Puts a distractor object on the counter
        and the sink.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=None,
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=self._get_placement_region_kwargs(
                        self.counter,
                        ref=self.sink
                    ),
                    size=self.obj_init_range,
                    pos=("ref", -1.0),
                    rotation={"y": np.pi/2, "z": [3 * np.pi/4, 5 * np.pi/4]}
                ),
            )
        )

        # distractors
        if self.use_distractors:
            cfgs.append(
                dict(
                    name="distr_counter",
                    obj_groups="all",
                    exclude_obj_groups=self.obj_groups if self.obj_groups != "all" else None,
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=self._get_placement_region_kwargs(
                            self.counter,
                            ref=self.sink
                        ),
                        size=(0.30, 0.30),
                        pos=("ref", -1.0),
                        offset=(0.0, 0.30),
                    ),
                )
            )
            cfgs.append(
                dict(
                    name="distr_sink",
                    obj_groups="all",
                    washable=True,
                    placement=dict(
                        fixture=self.sink,
                        size=(0.25, 0.25),
                        pos=(0.0, 1.0),
                    ),
                )
            )

        return cfgs

    def _check_success(self):
        """
        Check if the counter to sink pick and place task is successful.
        Checks if the object is inside the sink and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj_on_counter = OU.check_obj_fixture_contact(self, "obj", self.counter)
        gripper_obj_far = OU.gripper_obj_far(self, th=0.1)

        quat = T.convert_quat(np.array(self.sim.data.body_xquat[self.obj_body_id["obj"]]), to="xyzw")
        rot = quat2mat(quat)

        local_up = rot[:, 2]
        global_up = np.array([0, 0, 1])

        cos_angle = np.dot(local_up, global_up)
        upright = cos_angle > 0.95

        return obj_on_counter and gripper_obj_far and upright

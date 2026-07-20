from robosuite.models.robots import IIWA, Jaco
from robosuite.robots import register_robot_class

import numpy as np


@register_robot_class("WheeledRobot")
class IIWAOmron(IIWA):
    """
    Variant of Panda robot with mobile base. Currently serves as placeholder class.
    """

    @property
    def default_base(self):
        return "OmronMobileBase"

    @property
    def default_arms(self):
        return {"right": "IIWA"}

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.6, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }
    

@register_robot_class("WheeledRobot")
class JacoOmron(Jaco):
    """
    Variant of Panda robot with mobile base. Currently serves as placeholder class.
    """

    @property
    def default_base(self):
        return "OmronMobileBase"

    @property
    def default_arms(self):
        return {"right": "Jaco"}

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.6, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }
    
    @property
    def default_gripper(self):
        return {"right": "Robotiq85Gripper"}
    

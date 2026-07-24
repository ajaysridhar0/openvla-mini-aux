# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""Register the MimicGen configurations used by BARX."""

from mimicgen.configs.config import (  # noqa: F401
    MG_Config,
    config_factory,
    get_all_registered_configs,
)
from mimicgen.configs.robocasa.single_stage.config_mug import *  # noqa: F403
from mimicgen.configs.robocasa.single_stage.config_pnp import *  # noqa: F403
from mimicgen.configs.robocasa.single_stage.config_sink import *  # noqa: F403
from mimicgen.configs.task_spec import MG_TaskSpec  # noqa: F401

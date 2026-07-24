# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""BARX's minimal, simulator-compatible MimicGen distribution."""

__version__ = "1.0.1+barx.1"

# Register only the RoboCasa interfaces used by BARX. The upstream package
# imports legacy RoboSuite environments at module import time; those tasks are
# unrelated to BARX and target an older RoboSuite API.
from mimicgen.env_interfaces.robocasa.single_stage.mg_mug import (  # noqa: F401, E402
    MG_FlipMugUpright,
)
from mimicgen.env_interfaces.robocasa.single_stage.mg_pnp import (  # noqa: F401, E402
    MG_PnPCounterToSink,
    MG_PnPSinkToCounter,
)
from mimicgen.env_interfaces.robocasa.single_stage.mg_sink import (  # noqa: F401, E402
    MG_TurnOnSinkFaucet,
)

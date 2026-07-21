import os

assets_root = os.path.join(os.path.dirname(__file__), "assets")

# Importing these classes runs robosuite's registration decorators. RoboCasa
# environments refer to them by name, so this side effect must happen during a
# normal ``import robocasa`` rather than relying on callers to know the module.
from robocasa.models.compositional_robots import IIWAOmron, JacoOmron  # noqa: E402, F401

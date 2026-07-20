"""Public compatibility utilities for BARX simulation experiments."""

from .action_space import canonicalize_action, robocasa_action, robocasa_noop_action
from .benchmark import EMBODIMENTS, TASKS, evaluation_scene_config
from .names import (
    METHOD_TRANSFORMS,
    canonicalize_prediction_keys,
    internal_representation_name,
    representations_for_method,
    resolve_transform_spec,
)

__all__ = [
    "METHOD_TRANSFORMS",
    "EMBODIMENTS",
    "TASKS",
    "canonicalize_action",
    "canonicalize_prediction_keys",
    "internal_representation_name",
    "evaluation_scene_config",
    "representations_for_method",
    "resolve_transform_spec",
    "robocasa_action",
    "robocasa_noop_action",
]

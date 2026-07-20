"""Public compatibility utilities for BARX simulation experiments."""

from .action_space import canonicalize_action, robocasa_action, robocasa_noop_action
from .names import (
    METHOD_TRANSFORMS,
    canonicalize_prediction_keys,
    internal_representation_name,
    representations_for_method,
    resolve_transform_spec,
)

__all__ = [
    "METHOD_TRANSFORMS",
    "canonicalize_action",
    "canonicalize_prediction_keys",
    "internal_representation_name",
    "representations_for_method",
    "resolve_transform_spec",
    "robocasa_action",
    "robocasa_noop_action",
]

"""
barx_aliases.py

Backward-compatible BARX alias wrapper.

The public release naming has moved to RoboCasa-X, but some internal imports and
older user scripts still reference `barx_aliases`. Re-export the RoboCasa-X alias
tables and canonicalizers here so those call sites continue to work.
"""

from prismatic.vla.robocasa_x_aliases import (
    ROBOCASA_X_ACTION_TOKENIZER_ALIASES,
    ROBOCASA_X_DATA_MIX_ALIASES,
    ROBOCASA_X_DATASET_NAME_ALIASES,
    ROBOCASA_X_RELEASED_TFDS_DATASET_ALIASES,
    canonicalize_robocasa_x_dataset_list,
    canonicalize_robocasa_x_dataset_name,
    canonicalize_robocasa_x_dataset_statistics_map,
    canonicalize_robocasa_x_subset_percentages,
)


BARX_ACTION_TOKENIZER_ALIASES = ROBOCASA_X_ACTION_TOKENIZER_ALIASES
BARX_DATA_MIX_ALIASES = ROBOCASA_X_DATA_MIX_ALIASES
BARX_DATASET_NAME_ALIASES = ROBOCASA_X_DATASET_NAME_ALIASES
BARX_RELEASED_TFDS_DATASET_ALIASES = ROBOCASA_X_RELEASED_TFDS_DATASET_ALIASES


def canonicalize_barx_dataset_name(name: str) -> str:
    return canonicalize_robocasa_x_dataset_name(name)


def canonicalize_barx_dataset_statistics_map(dataset_statistics_map):
    return canonicalize_robocasa_x_dataset_statistics_map(dataset_statistics_map)


def canonicalize_barx_dataset_list(dataset_names):
    return canonicalize_robocasa_x_dataset_list(dataset_names)


def canonicalize_barx_subset_percentages(subset_percentages):
    return canonicalize_robocasa_x_subset_percentages(subset_percentages)

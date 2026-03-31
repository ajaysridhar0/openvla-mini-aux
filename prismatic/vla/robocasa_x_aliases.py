"""
robocasa_x_aliases.py

Public RoboCasa-X aliases for released dataset names and training CLI identifiers.

These aliases let the codebase and docs use the released RoboCasa-X naming while
preserving backwards compatibility with the temporary BARX release names and the
historical TFDS / tokenizer ids still used internally.
"""

from typing import Dict, List, Optional


ROBOCASA_X_DATA_MIX_ALIASES: Dict[str, str] = {}
ROBOCASA_X_DATASET_NAME_ALIASES: Dict[str, str] = {}
ROBOCASA_X_ACTION_TOKENIZER_ALIASES: Dict[str, str] = {}
ROBOCASA_X_RELEASED_TFDS_DATASET_ALIASES: Dict[str, str] = {}


_ROBOCASA_X_TASKS = {
    "pnp": {
        "raw_suffix": "pnp",
        "xp3k_prior": "mg_pnp",
        "xp900_prior": "mg_pnp_lite",
    },
    "turn-on-sink-faucet": {
        "raw_suffix": "turn_on_sink",
        "xp3k_prior": "mg_turn_on_sink",
        "xp900_prior": "mg_turn_on_sink_lite",
    },
    "flip-mug-upright": {
        "raw_suffix": "flip_mug",
        "xp3k_prior": "mg_flip_mug",
        "xp900_prior": "mg_flip_mug_lite",
    },
}

_ROBOCASA_X_ROBOTS = {
    "panda": "panda",
    "panda-og": "panda_og",
    "jaco": "jaco",
}


def _released_tfds_name(public_name: str) -> str:
    return public_name.replace("-", "_")


def _register_public_alias(alias_name: str, released_tfds_name: str, config_dataset_name: str, tokenizer_name: str) -> None:
    ROBOCASA_X_DATA_MIX_ALIASES[alias_name] = released_tfds_name
    ROBOCASA_X_DATASET_NAME_ALIASES[alias_name] = released_tfds_name
    ROBOCASA_X_RELEASED_TFDS_DATASET_ALIASES[released_tfds_name] = config_dataset_name
    ROBOCASA_X_ACTION_TOKENIZER_ALIASES[f"{alias_name}-vq-extra-action-tokenizer"] = tokenizer_name


for prefix in ("robocasa-x", "barx"):
    for task_slug, task_info in _ROBOCASA_X_TASKS.items():
        xp3k_alias = f"{prefix}-xp3k-{task_slug}"
        xp900_alias = f"{prefix}-xp900-{task_slug}"
        xp3k_tfds_name = _released_tfds_name(f"robocasa-x-xp3k-{task_slug}")
        xp900_tfds_name = _released_tfds_name(f"robocasa-x-xp900-{task_slug}")

        _register_public_alias(
            xp3k_alias,
            xp3k_tfds_name,
            task_info["xp3k_prior"],
            f"{task_info['xp3k_prior']}_vq_extra_action_tokenizer",
        )
        _register_public_alias(
            xp900_alias,
            xp900_tfds_name,
            task_info["xp900_prior"],
            f"{task_info['xp900_prior']}_vq_extra_action_tokenizer",
        )

        for robot_alias, robot_name in _ROBOCASA_X_ROBOTS.items():
            target_alias = f"{prefix}-target-{robot_alias}-{task_slug}"
            target_tfds_name = _released_tfds_name(f"robocasa-x-target-{robot_alias}-{task_slug}")
            target_dataset_name = f"{robot_name}_{task_info['raw_suffix']}"

            same_prior_alias = f"{prefix}-sp900-{robot_alias}-{task_slug}"
            same_prior_tfds_name = _released_tfds_name(f"robocasa-x-sp900-{robot_alias}-{task_slug}")
            same_prior_dataset_name = f"mg_{robot_name}_{task_info['raw_suffix']}"

            _register_public_alias(
                target_alias,
                target_tfds_name,
                target_dataset_name,
                f"{target_dataset_name}_vq_extra_action_tokenizer",
            )
            _register_public_alias(
                same_prior_alias,
                same_prior_tfds_name,
                same_prior_dataset_name,
                f"{same_prior_dataset_name}_vq_extra_action_tokenizer",
            )

            ROBOCASA_X_DATA_MIX_ALIASES[f"{prefix}-xp3k-{robot_alias}-{task_slug}-mix"] = (
                f"{prefix}-xp3k-{robot_alias}-{task_slug}-mix"
            )
            ROBOCASA_X_DATA_MIX_ALIASES[f"{prefix}-xp900-{robot_alias}-{task_slug}-mix"] = (
                f"{prefix}-xp900-{robot_alias}-{task_slug}-mix"
            )
            ROBOCASA_X_DATA_MIX_ALIASES[f"{same_prior_alias}-mix"] = f"{same_prior_alias}-mix"


# Backwards compatibility for users who downloaded the earlier BARX TFDS payload names.
for task_slug, task_info in _ROBOCASA_X_TASKS.items():
    ROBOCASA_X_RELEASED_TFDS_DATASET_ALIASES[f"barx_xp3k_{task_slug.replace('-', '_')}"] = task_info["xp3k_prior"]
    ROBOCASA_X_RELEASED_TFDS_DATASET_ALIASES[f"barx_xp900_{task_slug.replace('-', '_')}"] = task_info["xp900_prior"]
    for robot_alias, robot_name in _ROBOCASA_X_ROBOTS.items():
        raw_suffix = task_info["raw_suffix"]
        ROBOCASA_X_RELEASED_TFDS_DATASET_ALIASES[
            f"barx_sp900_{robot_alias.replace('-', '_')}_{task_slug.replace('-', '_')}"
        ] = f"mg_{robot_name}_{raw_suffix}"


def canonicalize_robocasa_x_dataset_name(name: str) -> str:
    """Return the released TFDS dataset / stats name for a public RoboCasa-X alias."""
    return ROBOCASA_X_DATASET_NAME_ALIASES.get(name, name)


def canonicalize_robocasa_x_dataset_statistics_map(
    dataset_statistics_map: Optional[Dict[str, str]]
) -> Optional[Dict[str, str]]:
    """Normalize RoboCasa-X public aliases to the released TFDS dataset names."""
    if dataset_statistics_map is None:
        return None

    return {
        canonicalize_robocasa_x_dataset_name(dataset_name): canonicalize_robocasa_x_dataset_name(stats_name)
        for dataset_name, stats_name in dataset_statistics_map.items()
    }


def canonicalize_robocasa_x_dataset_list(dataset_names: Optional[List[str]]) -> Optional[List[str]]:
    """Normalize a list of RoboCasa-X dataset names, preserving order."""
    if dataset_names is None:
        return None

    return [canonicalize_robocasa_x_dataset_name(dataset_name) for dataset_name in dataset_names]


def canonicalize_robocasa_x_subset_percentages(
    subset_percentages: Optional[Dict[str, float]]
) -> Optional[Dict[str, float]]:
    """Normalize subset percentage keys to the released TFDS dataset names used by RLDS loading."""
    if subset_percentages is None:
        return None

    return {
        canonicalize_robocasa_x_dataset_name(dataset_name): subset_fraction
        for dataset_name, subset_fraction in subset_percentages.items()
    }

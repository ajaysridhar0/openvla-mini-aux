"""Load and validate the small BARX preset registry."""

from pathlib import Path
from typing import Any, Dict

import yaml

PRESET_PATH = Path(__file__).parent / "configs" / "presets.yaml"


class ConfigError(ValueError):
    """Raised when a user selects an invalid preset combination."""


def load_presets(path: Path = PRESET_PATH) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as preset_file:
        presets = yaml.safe_load(preset_file)
    if presets.get("version") != 1:
        raise ConfigError(f"Unsupported preset version in {path}")
    return presets


def select(mapping: Dict[str, Any], name: str, kind: str) -> Dict[str, Any]:
    try:
        return mapping[name]
    except KeyError as exc:
        choices = ", ".join(sorted(mapping))
        raise ConfigError(f"Unknown {kind} {name!r}; choose one of: {choices}") from exc


def require_target_robot(robot_name: str, robot: Dict[str, Any]) -> None:
    if not robot["target"]:
        raise ConfigError(
            f"{robot_name!r} is a source robot. Training target splits support only panda, panda-og, and jaco."
        )

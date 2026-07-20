"""Canonical action adapters used by BARX conversion and evaluation.

The policy and released RLDS datasets use seven values:
``[dx, dy, dz, droll, dpitch, dyaw, gripper]``. RoboCasa-X uses a
12-value mobile-manipulator command whose gripper index differs for the
Panda mobile base. Keeping that difference here reproduces the original two
converter implementations without exposing two public datasets or APIs.
"""

from __future__ import annotations

from typing import Iterable


_PANDA_NAMES = {
    "panda",
    "pandaog",
    "pandaoggripperomron",
    "pandaogomron",
    "pandaomron",
}


def _compact_name(embodiment: str) -> str:
    return "".join(character for character in embodiment.lower() if character.isalnum())


def is_panda_embodiment(embodiment: str) -> bool:
    """Return whether ``embodiment`` uses RoboCasa-X's Panda action layout."""
    if not embodiment:
        raise ValueError("An embodiment is required to disambiguate raw RoboCasa-X actions.")
    return _compact_name(embodiment) in _PANDA_NAMES


def canonicalize_action(raw_action: Iterable[float], embodiment: str) -> list[float]:
    """Convert a recorded RoboCasa-X action to the canonical seven values.

    This exactly preserves the slicing used by the published experiments:

    - Panda and Panda-OG 12-D actions use the first seven values.
    - Other 12-D mobile-manipulator actions use the first six and penultimate
      value.
    - Legacy 11-D non-mobile actions use the final value as the gripper.
    - Already canonical 7-D actions pass through unchanged.
    """
    action = list(raw_action)
    if len(action) == 7:
        return action.copy()
    if len(action) < 7:
        raise ValueError(f"Expected at least 7 action values, received {len(action)}.")

    if is_panda_embodiment(embodiment):
        return action[:7]
    if len(action) == 11:
        gripper = action[-1]
    else:
        gripper = action[-2]
    return action[:6] + [gripper]


def robocasa_action(action: Iterable[float], embodiment: str) -> list[float]:
    """Expand a canonical action into the original 12-D RoboCasa-X layout."""
    canonical = list(action)
    if len(canonical) != 7:
        raise ValueError(f"Expected a canonical 7-D action, received {len(canonical)} values.")

    if is_panda_embodiment(embodiment):
        expanded = canonical + [0.0] * 4 + [-1.0]
    else:
        expanded = canonical[:6] + [0.0] * 4 + canonical[6:] + [-1.0]
    assert len(expanded) == 12
    return expanded


def robocasa_noop_action(embodiment: str) -> list[float]:
    """Return the original RoboCasa-X no-op command for an embodiment."""
    return robocasa_action([0.0] * 6 + [-1.0], embodiment)

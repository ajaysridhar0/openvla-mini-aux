"""Action adapters for the normalized BARX simulation-data format.

Released HDF5 files and every RoboCasa-X embodiment use the same 12 values:
``[arm(6), gripper(1), base(3), torso(1), mode(1)]``. Policies and RLDS
datasets use the first seven values.
"""

from __future__ import annotations

from typing import Iterable


def canonicalize_action(raw_action: Iterable[float]) -> list[float]:
    """Return the seven policy-controlled values from a normalized action."""
    action = list(raw_action)
    if len(action) == 7:
        return action.copy()
    if len(action) == 12:
        return action[:7]
    raise ValueError(f"Expected a 7-D policy or 12-D BARX action, received {len(action)} values.")


def robocasa_action(action: Iterable[float]) -> list[float]:
    """Expand a policy action into the normalized 12-D RoboCasa-X layout."""
    canonical = list(action)
    if len(canonical) != 7:
        raise ValueError(f"Expected a canonical 7-D action, received {len(canonical)} values.")

    expanded = canonical + [0.0] * 4 + [-1.0]
    assert len(expanded) == 12
    return expanded


def robocasa_noop_action() -> list[float]:
    """Return a no-op command in the normalized RoboCasa-X layout."""
    return robocasa_action([0.0] * 6 + [-1.0])

"""Logical BARX raw-data subsets shared by download and RLDS conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

PAPER_SET_NAMES = {
    "xp_900": "XP-900",
    "xp_3k": "XP-3K",
    "sp_900": "SP-900",
    "target_50": "target-50",
}
TASK_NAMES = {
    "pnp": {
        "PnP Counter to Sink",
        "PnP Sink to Counter",
    },
    "turn_on_sink": {"Turn On Sink Faucet"},
    "flip_mug": {"Flip Mug Upright"},
}
TARGET_NAMES = {
    "panda": "Panda",
    "panda_og": "Panda-OG",
    "jaco": "Jaco",
}

_DATASET_SLUGS = {
    "xp_900": "xp900",
    "xp_3k": "xp3k",
    "sp_900": "sp900",
    "target_50": "target",
}
_TASK_SLUGS = {
    "pnp": "pnp",
    "turn_on_sink": "turn-on-sink-faucet",
    "flip_mug": "flip-mug-upright",
}
_TARGET_SLUGS = {
    "panda": "panda",
    "panda_og": "panda-og",
    "jaco": "jaco",
}


@dataclass(frozen=True)
class RawSubset:
    """One raw HDF5 view corresponding to one public RLDS repository."""

    dataset: str
    task: str
    target: str | None = None

    def __post_init__(self) -> None:
        validate_selection(self.dataset, self.task, self.target)

    @property
    def slug(self) -> str:
        parts = ["robocasa-x", _DATASET_SLUGS[self.dataset]]
        if self.target is not None:
            parts.append(_TARGET_SLUGS[self.target])
        parts.append(_TASK_SLUGS[self.task])
        return "-".join(parts)

    @property
    def rlds_repo_id(self) -> str:
        return f"ajaysri/{self.slug}"


def validate_selection(dataset: str, task: str, target: str | None) -> None:
    """Validate one paper-facing dataset/task/target selection."""

    if dataset not in PAPER_SET_NAMES:
        raise ValueError(f"Unknown paper dataset: {dataset}")
    if task not in TASK_NAMES:
        raise ValueError(f"Unknown paper task: {task}")
    requires_target = dataset in {"sp_900", "target_50"}
    if requires_target and target not in TARGET_NAMES:
        raise ValueError(f"{dataset} requires target in {sorted(TARGET_NAMES)}")
    if not requires_target and target is not None:
        raise ValueError(f"{dataset} does not accept a target embodiment")


def all_subsets() -> tuple[RawSubset, ...]:
    """Return the 24 logical HDF5 subsets in public RLDS collection order."""

    subsets = []
    for dataset in ("xp_3k", "xp_900"):
        for task in ("pnp", "turn_on_sink", "flip_mug"):
            subsets.append(RawSubset(dataset, task))
    for dataset in ("sp_900", "target_50"):
        for target in ("panda", "panda_og", "jaco"):
            for task in ("pnp", "turn_on_sink", "flip_mug"):
                subsets.append(RawSubset(dataset, task, target))
    return tuple(subsets)


def selected_rows(
    rows: Iterable[Mapping[str, str]], subset: RawSubset
) -> list[Mapping[str, str]]:
    """Select master-manifest rows belonging to one logical subset."""

    paper_set = PAPER_SET_NAMES[subset.dataset]
    target_name = TARGET_NAMES.get(subset.target) if subset.target else None
    selected = []
    for row in rows:
        memberships = set(row["paper_sets"].split(";"))
        if paper_set not in memberships or row["task"] not in TASK_NAMES[subset.task]:
            continue
        if target_name is not None and row["embodiment"] != target_name:
            continue
        selected.append(row)
    if not selected:
        raise ValueError(f"No manifest entries matched {subset.slug}")
    return selected

"""Paper-facing BARX names and backward-compatible schema aliases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


# Public names from the paper map to the identifiers stored in the original
# datasets and understood by the training code. Do not change the values: they
# are part of the released RLDS/checkpoint compatibility contract.
REPRESENTATION_ALIASES = {
    "bounding_box": "bbox",
    "language_motion": "low_level_motion",
    "end_effector_trace": "ee_pose_2D",
}
INTERNAL_TO_PAPER_REPRESENTATION = {value: key for key, value in REPRESENTATION_ALIASES.items()}

METHOD_TRANSFORMS = {
    "no_reps": "action",
    "bounding_box": "bounding_box->,action",
    "language_motion": "language_motion->,action",
    "end_effector_trace": "end_effector_trace->,action",
    "joint_reps": "bounding_box->,language_motion->,end_effector_trace->,action",
    "ecot": "bounding_box->end_effector_trace->language_motion->,action",
}


def internal_representation_name(name: str) -> str:
    """Resolve one paper-facing representation name to its stored identifier."""
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    return REPRESENTATION_ALIASES.get(normalized, name.strip())


def resolve_transform_spec(specification: str) -> str:
    """Resolve a paper method or transform expression to the legacy exact form."""
    normalized = specification.strip().lower().replace("-", "_").replace(" ", "_")
    specification = METHOD_TRANSFORMS.get(normalized, specification)
    for paper_name, internal_name in REPRESENTATION_ALIASES.items():
        specification = specification.replace(paper_name, internal_name)
    return specification.replace(" ", "")


def representations_for_method(method: str) -> list[str]:
    """Return ordered internal representation queries for a paper method."""
    transform_spec = resolve_transform_spec(method)
    representations = []
    for chain in transform_spec.split(","):
        representations.extend(item for item in chain.split("->") if item and item != "action")
    return representations


def canonicalize_prediction_keys(prediction: Mapping[str, Any]) -> dict[str, Any]:
    """Add paper-facing keys while retaining original prediction keys."""
    canonical = dict(prediction)
    for internal_name, paper_name in INTERNAL_TO_PAPER_REPRESENTATION.items():
        if internal_name in prediction:
            canonical.setdefault(paper_name, prediction[internal_name])
    return canonical

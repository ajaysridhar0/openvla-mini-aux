"""Materialization for RAW dataset w/ images / actions / instructions. (no tokens)"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

from prismatic.vla.datasets.datasets import EpisodicRLDSDataset, RLDSDataset


def _canonicalize_robocasa_x_dataset_name(name: str) -> str:
    if name.startswith("robocasa-x-") and not name.endswith("-mix"):
        return name.replace("-", "_")
    return name


def _canonicalize_robocasa_x_dataset_statistics_map(
    dataset_statistics_map: Optional[Dict[str, str]]
) -> Optional[Dict[str, str]]:
    if dataset_statistics_map is None:
        return None

    return {
        _canonicalize_robocasa_x_dataset_name(dataset_name): _canonicalize_robocasa_x_dataset_name(stats_name)
        for dataset_name, stats_name in dataset_statistics_map.items()
    }


def _canonicalize_robocasa_x_subset_percentages(
    subset_percentages: Optional[Dict[str, float]]
) -> Optional[Dict[str, float]]:
    if subset_percentages is None:
        return None

    return {
        _canonicalize_robocasa_x_dataset_name(dataset_name): subset_fraction
        for dataset_name, subset_fraction in subset_percentages.items()
    }


@dataclass
class RLDSActionBatchTransform:
    include_images: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        batch = dict(instruction=lang, action=action, dataset_name=dataset_name)
        if self.include_images:
            batch["image"] = rlds_batch["observation"]["image_primary"][0]

        return batch


def get_vla_action_dataset(
    data_root_dir: Path,
    data_mix: str,
    default_image_resolution: Tuple[int, int, int],
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    future_action_window_size: int = 0,
    include_images: bool = True,
    dataset_statistics_map: Optional[Dict] = None,
    subset_percentages: Optional[Dict[str, float]] = None,
):
    """Only get the image / action / instruction, don't do any tokenization."""

    dataset_statistics_map = _canonicalize_robocasa_x_dataset_statistics_map(dataset_statistics_map)
    subset_percentages = _canonicalize_robocasa_x_subset_percentages(subset_percentages)

    # TODO new batch transform
    batch_transform = RLDSActionBatchTransform(include_images=include_images)

    # Build RLDS Iterable Dataset & Return
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
        # did not add support for below kwargs with episodic dataset
        future_action_window_size=future_action_window_size,
        dataset_statistics_map=dataset_statistics_map,
        subset_percentages=subset_percentages,
    )

    return dataset

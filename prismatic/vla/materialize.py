"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type, List, Optional, Dict

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.models.backbones.vision.base_vision import WrapSequenceImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla import ActionTokenizer, ACTION_TOKENIZERS
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSDataset
from prismatic.vla.datasets.datasets import (
    AUX_TASK_QA_FUNCTIONS,
    RLDSBatchTransform,
    RLDSAuxTransform,
    ChainedTransform,
)


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


def _canonicalize_robocasa_x_dataset_list(dataset_names: Optional[List[str]]) -> Optional[List[str]]:
    if dataset_names is None:
        return None

    return [_canonicalize_robocasa_x_dataset_name(dataset_name) for dataset_name in dataset_names]


def _canonicalize_robocasa_x_subset_percentages(
    subset_percentages: Optional[Dict[str, float]]
) -> Optional[Dict[str, float]]:
    if subset_percentages is None:
        return None

    return {
        _canonicalize_robocasa_x_dataset_name(dataset_name): subset_fraction
        for dataset_name, subset_fraction in subset_percentages.items()
    }


def _parse_transform_spec(transform_types: str) -> List[str]:
    """Parse the CLI transform string.

    Syntax is intentionally compact:
    - `action` means direct action supervision only.
    - `bbox` means aux-only supervision for that task.
    - `bbox->` means predict `bbox`, then predict the action.
    - `bbox->obj_pose->ee_pose_2D` means predict those aux tasks in order, then action.
    - Commas separate independently sampled transform modes.
    """
    parsed_transform_types = [transform_type.strip() for transform_type in transform_types.split(",") if transform_type.strip()]
    if len(parsed_transform_types) == 0:
        raise ValueError("Must specify at least one transform in `transform_types`.")
    return parsed_transform_types


def _parse_transform_weights(transform_weights: Optional[str], num_transforms: int) -> List[float]:
    """Parse optional transform weights and validate they align with the transform spec."""
    if transform_weights is None:
        return [1 / num_transforms] * num_transforms

    parsed_transform_weights = [float(weight.strip()) for weight in transform_weights.split(",") if weight.strip()]
    if len(parsed_transform_weights) != num_transforms:
        raise ValueError(
            f"Expected {num_transforms} transform weights for `{num_transforms}` transform specs, "
            f"but found {len(parsed_transform_weights)}."
        )
    if abs(sum(parsed_transform_weights) - 1.0) >= 1e-6:
        raise ValueError("Transform weights must sum to 1.0.")
    return parsed_transform_weights


def _validate_aux_transform_names(aux_task_types: List[str], transform_type: str) -> None:
    valid_aux_task_types = sorted(AUX_TASK_QA_FUNCTIONS.keys())
    invalid_aux_task_types = [aux_task_type for aux_task_type in aux_task_types if aux_task_type not in AUX_TASK_QA_FUNCTIONS]
    if invalid_aux_task_types:
        raise ValueError(
            f"Invalid aux transform(s) {invalid_aux_task_types} in `{transform_type}`. "
            f"Valid aux transforms: {valid_aux_task_types}."
        )


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    action_tokenizer: str = "action_tokenizer",
    future_action_window_size: int = 0,
    future_obj_pose_window_size: int = 0,
    future_2D_trace_window_size: int = 0,
    obj_pose_stride: int = 1,
    ee_pose_2D_stride: int = 1,
    image_window_size: int = 1,
    transform_types: str = "action",
    transform_weights: str = None,
    use_wrist_image: bool = False,
    normalize_data: bool = True,
    past_obj_pose_window_size: int = 0,
    past_2D_trace_window_size: int = 0,
    subset_percentages: Optional[Dict[str, float]] = None,
    global_subset_fraction: Optional[float] = None,
    dataset_statistics_map: Optional[Dict] = None,
    non_action_datasets: Optional[List] = None,
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""

    dataset_statistics_map = _canonicalize_robocasa_x_dataset_statistics_map(dataset_statistics_map)
    subset_percentages = _canonicalize_robocasa_x_subset_percentages(subset_percentages)
    non_action_datasets = _canonicalize_robocasa_x_dataset_list(non_action_datasets)
    action_tokenizer: ActionTokenizer = ACTION_TOKENIZERS[action_tokenizer](tokenizer)

    # get the future action window needed from the tokenizer
    future_action_window_size = max(action_tokenizer.required_future_horizon, future_action_window_size)

    load_camera_views = ("primary", "wrist") if use_wrist_image else ("primary",)

    # get the observation history from the image_transform (only needed if its a WrapSequence transform)
    if isinstance(image_transform, WrapSequenceImageTransform):
        if use_wrist_image:
            # expects groupings of two in image sequence len
            assert image_transform.sequence_len % 2 == 0, "With wrist images, image transform must expect 2N images!"
            image_window_size = max(image_transform.sequence_len // 2, image_window_size)
        else:
            image_window_size = max(image_transform.sequence_len, image_window_size)

    transform_base_args = {
        "tokenizer": tokenizer,
        "image_transform": image_transform,
        "prompt_builder_fn": prompt_builder_fn,
        "predict_stop_token": predict_stop_token,
        "image_window_size": image_window_size,
        "use_wrist_image": use_wrist_image,
    }

    batch_transforms = []
    non_action_transforms = []
    parsed_transform_types = _parse_transform_spec(transform_types)
    parsed_transform_weights = _parse_transform_weights(transform_weights, len(parsed_transform_types))

    for transform_type, weight in zip(parsed_transform_types, parsed_transform_weights):
        non_action_rlds_transform = None
        if "->" in transform_type:
            chained_transforms = [aux_task_type for aux_task_type in transform_type.split("->") if aux_task_type]
            if len(chained_transforms) == 0:
                raise ValueError(
                    f"Invalid chained transform `{transform_type}`. "
                    "Use `<aux>` for aux-only or `<aux1>->...-><auxN>` for chained aux-to-action supervision."
                )
            if "action" in chained_transforms:
                raise ValueError(
                    f"Invalid chained transform `{transform_type}`. "
                    "Do not include `action` inside a chain; chained transforms append action supervision automatically."
                )
            _validate_aux_transform_names(chained_transforms, transform_type)
            rlds_transform = ChainedTransform(
                **transform_base_args,
                action_tokenizer=action_tokenizer,
                aux_task_types=chained_transforms,
            )

            non_action_chained_transforms = chained_transforms.copy()

            if "low_level_motion" in non_action_chained_transforms:  # requires action for annotation
                non_action_chained_transforms.remove("low_level_motion")

            if non_action_chained_transforms:
                non_action_rlds_transform = RLDSAuxTransform(
                    **transform_base_args,
                    # For aux-only datasets, reuse the first chained aux task as a representative aux target.
                    aux_task_type=non_action_chained_transforms[0],
                )

        elif transform_type == "action":
            rlds_transform = RLDSBatchTransform(
                **transform_base_args, 
                action_tokenizer=action_tokenizer, 
            )
        else:
            _validate_aux_transform_names([transform_type], transform_type)
            rlds_transform = RLDSAuxTransform(
                **transform_base_args, 
                aux_task_type=transform_type, 
            )

            if transform_type != "low_level_motion":  # requires action for annotation
                non_action_rlds_transform = rlds_transform

        batch_transforms.append((rlds_transform, weight))
        if non_action_rlds_transform is not None:
            non_action_transforms.append((non_action_rlds_transform, weight))

    if non_action_transforms:
        sum_weights = sum([weight for transform, weight in non_action_transforms])
        non_action_transforms = [(transform, weight / sum_weights) for transform, weight in non_action_transforms]
    
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    # Build RLDS Iterable Dataset
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transforms,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
        future_action_window_size=future_action_window_size,
        future_obj_pose_window_size=future_obj_pose_window_size,
        future_2D_trace_window_size=future_2D_trace_window_size,
        past_obj_pose_window_size=past_obj_pose_window_size,
        past_2D_trace_window_size=past_2D_trace_window_size,
        obj_pose_stride=obj_pose_stride,
        ee_pose_2D_stride=ee_pose_2D_stride,
        image_window_size=image_window_size,
        load_camera_views=load_camera_views,
        normalize_data=normalize_data,
        subset_percentages=subset_percentages,
        global_subset_fraction=global_subset_fraction,
        dataset_statistics_map=dataset_statistics_map,
        non_action_datasets=non_action_datasets,
        non_action_transforms=non_action_transforms,
    )

    return dataset, action_tokenizer, collator

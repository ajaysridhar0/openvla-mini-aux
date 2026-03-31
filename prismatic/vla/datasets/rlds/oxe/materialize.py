"""
materialize.py

Factory class for initializing Open-X Embodiment dataset kwargs and other parameters; provides and exports functions for
clear control flow.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from prismatic.overwatch import initialize_overwatch
from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS, ActionEncoding
from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
from prismatic.vla.datasets.rlds.oxe.transforms import libero_dataset_transform

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def make_oxe_dataset_kwargs(
    dataset_name: str,
    data_root_dir: Path,
    load_camera_views: Tuple[str] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
) -> Dict[str, Any]:
    """Generates config (kwargs) for given dataset from Open-X Embodiment."""
    config_dataset_name = dataset_name
    dataset_kwargs = deepcopy(OXE_DATASET_CONFIGS[config_dataset_name])
    if dataset_kwargs["action_encoding"] not in [ActionEncoding.EEF_POS, ActionEncoding.EEF_R6]:
        raise ValueError(f"Cannot load `{dataset_name}`; only EEF_POS & EEF_R6 actions supported!")

    # [Contract] For EEF_POS & EEF_R6 actions, only the last action dimension (gripper) is absolute!
    # Normalize all action dimensions *except* the gripper
    if dataset_kwargs["action_encoding"] is ActionEncoding.EEF_POS:
        dataset_kwargs["absolute_action_mask"] = [False] * 6 + [True]
        dataset_kwargs["action_normalization_mask"] = [True] * 6 + [False]
    elif dataset_kwargs["action_encoding"] is ActionEncoding.EEF_R6:
        dataset_kwargs["absolute_action_mask"] = [False] * 9 + [True]
        dataset_kwargs["action_normalization_mask"] = [True] * 9 + [False]
    dataset_kwargs["action_proprio_normalization_type"] = action_proprio_normalization_type

    # Adjust Loaded Camera Views
    if len(missing_keys := (set(load_camera_views) - set(dataset_kwargs["image_obs_keys"]))) > 0:
        raise ValueError(f"Cannot load `{dataset_name}`; missing camera views `{missing_keys}`")

    # Filter
    dataset_kwargs["image_obs_keys"] = {
        k: v for k, v in dataset_kwargs["image_obs_keys"].items() if k in load_camera_views
    }
    dataset_kwargs["depth_obs_keys"] = {
        k: v for k, v in dataset_kwargs["depth_obs_keys"].items() if k in load_camera_views
    }

    # Eliminate Unnecessary Keys
    dataset_kwargs.pop("state_encoding")
    dataset_kwargs.pop("action_encoding")
    if not load_depth:
        dataset_kwargs.pop("depth_obs_keys")
    if not load_proprio:
        dataset_kwargs.pop("state_obs_keys")

    # Load Language
    if load_language:
        dataset_kwargs["language_key"] = "language_instruction"

    # Specify Standardization Transform
    dataset_kwargs["standardize_fn"] = OXE_STANDARDIZATION_TRANSFORMS.get(
        config_dataset_name, libero_dataset_transform
    )

    # Add any aux arguments
    if "aux_kwargs" in dataset_kwargs:
        dataset_kwargs.update(dataset_kwargs.pop("aux_kwargs"))

    return {"name": dataset_name, "data_dir": str(data_root_dir), **dataset_kwargs}


def get_oxe_dataset_kwargs_and_weights(
    data_root_dir: Path,
    mixture_spec: List[Tuple[str, float]],
    load_camera_views: Tuple[str] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
    subset_percentages: Optional[Dict[str, float]] = None,
    global_subset_fraction: Optional[float] = None,
) -> Tuple[Dict[str, Any], List[float]]:
    """
    Generates dataset kwargs for a given dataset mix from the Open X-Embodiment dataset. The returned kwargs
    (per-dataset configs) and weights can be passed directly to `make_interleaved_dataset`.

    :param data_root_dir: Base directory containing RLDS/TFDS-formatted datasets (from Open-X)
    :param mixture_spec: List of (dataset_name, sampling_weight) from `oxe.mixtures.OXE_NAMED_MIXTURES`
    :param load_camera_views: Camera views to load; see `oxe.dataset_configs.py` for available views.
    :param load_depth: Load depth information in addition to camera RGB.
    :param load_proprio: Load proprioceptive state.
    :param load_language: Load language instructions.
    :param action_proprio_normalization_type: Normalization scheme to use for proprioceptive actions.
    :param subset_percentages: Optional dictionary mapping dataset names to percentage of data to use (0.0-1.0)
                               Example: {"dataset_name": 0.25} to use 25% of dataset_name
    :param global_subset_fraction: Optional float specifying a fraction to use for ALL datasets (0.0-1.0)
                                   Example: 0.25 to use 25% of all datasets
                                   Note: This is ignored if subset_percentages is provided

    return: Tuple of (per_dataset_kwargs, sampling_weights)
    """
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight in mixture_spec:
        if d_name in included_datasets:
            overwatch.warning(f"Skipping Duplicate Dataset: `{(d_name, d_weight)}`")
            continue

        included_datasets.add(d_name)
        filtered_mixture_spec.append((d_name, d_weight))

    # Assemble Dataset Config (kwargs) and Weights
    per_dataset_kwargs, sampling_weights = [], []
    
    # If global_subset_fraction is provided and subset_percentages is not, apply it to all datasets
    global_subset = None
    if subset_percentages is None and global_subset_fraction is not None:
        if 0.0 < global_subset_fraction < 1.0:
            global_subset = global_subset_fraction
            overwatch.info(f"Applying global subset fraction {global_subset:.1%} to all datasets")
    
    for d_name, d_weight in filtered_mixture_spec:
        try:
            dataset_kwargs = make_oxe_dataset_kwargs(
                d_name,
                data_root_dir,
                load_camera_views,
                load_depth,
                load_proprio,
                load_language,
                action_proprio_normalization_type,
            )
            
            # Apply subsetting if specified for this dataset
            if subset_percentages and d_name in subset_percentages:
                subset_pct = subset_percentages[d_name]
                if 0.0 < subset_pct < 1.0:
                    dataset_kwargs["subset_percentage"] = subset_pct
                    overwatch.info(f"Using {subset_pct:.1%} of dataset '{d_name}'")
            # Apply global subset if no specific subset is defined for this dataset
            elif global_subset is not None:
                dataset_kwargs["subset_percentage"] = global_subset
                overwatch.info(f"Using {global_subset:.1%} of dataset '{d_name}' (global setting)")
            
            per_dataset_kwargs.append(dataset_kwargs)
            sampling_weights.append(d_weight)

        except ValueError as e:
            overwatch.warning(f"Skipping `{d_name}` due to Error: {e}")

    return per_dataset_kwargs, sampling_weights

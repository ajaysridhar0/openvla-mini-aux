"""
load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import HfFileSystem, hf_hub_download

from prismatic.conf import ModelConfig
from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from prismatic.models.vlas import OpenVLA
from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ACTION_TOKENIZERS, ActionTokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === HF Hub Repository ===
HF_HUB_REPO = "TRI-ML/prismatic-vlms"
VLA_HF_HUB_REPO = "openvla/openvla-dev"

# Public Hugging Face repository names use hyphens where the historical
# Prismatic registry ID uses ``+``. Checkpoint configs may contain either.
BASE_VLM_ID_ALIASES = {
    "prism-qwen25-extra-dinosiglip-224px-0_5b-stage-finetune-x7":
        "prism-qwen25-extra-dinosiglip-224px+0_5b+stage-finetune+x7",
}


# === Available Models ===
def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def available_model_names() -> List[str]:
    return list(GLOBAL_REGISTRY.items())


def get_model_description(model_id_or_name: str) -> str:
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }; check `prismatic.available_model_names()`")

    # Print Description & Return
    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))

    return description


# === Load Pretrained Model ===
def load(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    image_sequence_len: Optional[int] = None,
    random_llm_weights: bool = False,
    vla_id: Optional[str] = None,
) -> PrismaticVLM:
    """
    Loads a pretrained PrismaticVLM from either local disk or the HuggingFace Hub.
    
    Args:
        ...
        random_llm_weights: If True, initialize LLM weights randomly instead of loading pretrained weights
        vla_id: Optional VLA ID to check for special backbone overrides
    """
    if os.path.isdir(model_id_or_path):
        overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")

        # Get paths for `config.json` and pretrained checkpoint
        config_json, checkpoint_pt = run_dir / "config.json", run_dir / "checkpoints" / "latest-checkpoint.pt"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert checkpoint_pt.exists(), f"Missing checkpoint for `{run_dir = }`"
    else:
        if model_id_or_path not in GLOBAL_REGISTRY:
            raise ValueError(f"Couldn't find `{model_id_or_path = }; check `prismatic.available_model_names()`")

        overwatch.info(f"Downloading `{(model_id := GLOBAL_REGISTRY[model_id_or_path]['model_id'])} from HF Hub")
        with overwatch.local_zero_first():
            config_json = hf_hub_download(repo_id=HF_HUB_REPO, filename=f"{model_id}/config.json", cache_dir=cache_dir)
            checkpoint_pt = hf_hub_download(
                repo_id=HF_HUB_REPO, filename=f"{model_id}/checkpoints/latest-checkpoint.pt", cache_dir=cache_dir
            )

    # Load Model Config from `config.json`
    with open(config_json, "r") as f:
        model_cfg = json.load(f)["model"]

    # Check if we're loading for a ResNet-based VLA
    if vla_id is not None and "resnet" in vla_id.lower():
        if "resnet50" in vla_id.lower():
            overwatch.info(f"Detected ResNet VLA ID: {vla_id}. Overriding vision backbone to: resnet50-224px")
            model_cfg["vision_backbone_id"] = "resnet50-224px"
        elif "resnet101" in vla_id.lower():
            overwatch.info(f"Detected ResNet VLA ID: {vla_id}. Overriding vision backbone to: resnet101-224px")
            model_cfg["vision_backbone_id"] = "resnet101-224px"
    
    # Also check if the checkpoint itself is from a ResNet model
    checkpoint_path = str(checkpoint_pt)
    if "resnet" in checkpoint_path.lower():
        if "resnet50" in checkpoint_path.lower():
            overwatch.info(f"Detected ResNet checkpoint. Using ResNet50 backbone.")
            model_cfg["vision_backbone_id"] = "resnet50-224px"
        elif "resnet101" in checkpoint_path.lower():
            overwatch.info(f"Detected ResNet checkpoint. Using ResNet101 backbone.")
            model_cfg["vision_backbone_id"] = "resnet101-224px"

    # = Load Individual Components necessary for Instantiating a VLM =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg['model_id']}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg['vision_backbone_id']}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg['llm_backbone_id']}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg['arch_specifier']}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    if image_sequence_len is None:
        if hasattr(model_cfg, "image_sequence_len"):
            image_sequence_len = model_cfg.image_sequence_len
        else:
            image_sequence_len = 1

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
        image_sequence_len,
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(
        f"{'Randomly initializing' if random_llm_weights else 'Loading Pretrained'} "
        f"LLM [bold]{model_cfg['llm_backbone_id']}[/] via HF Transformers"
    )
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=not load_for_training,
        pretrained=not random_llm_weights,
    )

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLM [bold blue]{model_cfg['model_id']}[/] from Checkpoint")
    vlm = PrismaticVLM.from_pretrained(
        checkpoint_pt,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg["arch_specifier"],
        freeze_weights=not load_for_training,
    )

    return vlm

# === Load Pretrained VLA Model ===
def load_vla(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    step_to_load: Optional[int] = None,
    model_type: str = "pretrained",
    image_sequence_len: Optional[int] = None,
    random_llm_weights: bool = False,  # Add this new parameter
    aux_context_freq: int = 1,
) -> OpenVLA:
    """
    Loads a pretrained OpenVLA from either local disk or the HuggingFace Hub.
    
    Args:
        ...
        random_llm_weights: If True, initialize LLM weights randomly instead of loading pretrained weights
    """

    # TODO (siddk, moojink) :: Unify semantics with `load()` above; right now, `load_vla()` assumes path points to
    #   checkpoint `.pt` file, rather than the top-level run directory!
    if os.path.isfile(model_id_or_path):
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(model_id_or_path))}`")

        # [Validate] Checkpoint Path should look like `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt`
        assert (checkpoint_pt.suffix == ".pt") and (checkpoint_pt.parent.name == "checkpoints"), "Invalid checkpoint!"
        run_dir = checkpoint_pt.parents[1]

        # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
        config_json, dataset_statistics_json = run_dir / "config.json", run_dir / "dataset_statistics.json"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir = }`"

    # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`VLA_HF_HUB_REPO`)
    else:
        # Search HF Hub Repo via fsspec API
        overwatch.info(f"Checking HF for `{(hf_path := str(model_id_or_path))}`")
        if not (tmpfs := HfFileSystem()).exists(hf_path):
            raise ValueError(f"Couldn't find valid HF Hub Path `{hf_path = }`")

        # Identify Checkpoint to Load (via `step_to_load`)
        step_to_load = f"{step_to_load:06d}" if step_to_load is not None else None
        valid_ckpts = tmpfs.glob(f"{hf_path}/checkpoints/step-{step_to_load if step_to_load is not None else ''}*.pt")
        if (len(valid_ckpts) == 0) or (step_to_load is not None and len(valid_ckpts) != 1):
            raise ValueError(f"Couldn't find a valid checkpoint to load from HF Hub Path `{hf_path}/checkpoints/")

        # Call to `glob` will sort steps in ascending order (if `step_to_load` is None); just grab last element
        target_ckpt = Path(valid_ckpts[-1]).name

        overwatch.info(f"Downloading Model `{model_id_or_path}` Config & Checkpoint `{target_ckpt}`")
        with overwatch.local_zero_first():
            relpath = model_id_or_path
            repo_id = relpath.as_posix()
            config_json = hf_hub_download(
                repo_id=repo_id, filename=f"{('config.json')!s}", cache_dir=cache_dir
            )
            dataset_statistics_json = hf_hub_download(
                repo_id=repo_id, filename=f"{('dataset_statistics.json')!s}", cache_dir=cache_dir
            )
            checkpoint_pt = hf_hub_download(
                repo_id=repo_id, filename=f"checkpoints/{target_ckpt}", cache_dir=cache_dir
            )

    # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
    with open(config_json, "r") as f:
        vla_cfg = json.load(f)["vla"]
        base_vlm = vla_cfg["base_vlm"]

    # Override vision backbone for our ResNet config (based on the vla_id)
    # This ensures we use the ResNet backbone even when loading from a DINO+SigLIP checkpoint
    vla_id = vla_cfg.get("vla_id", "")
    checkpoint_path = str(checkpoint_pt)
    
    # Check if this is a ResNet checkpoint by looking at the path
    is_resnet_checkpoint = "resnet" in checkpoint_path.lower()
    
    # Only override if it's a ResNet VLA ID but not a ResNet checkpoint
    if "resnet50" in vla_id.lower() and not is_resnet_checkpoint:
        vla_cfg["vision_backbone_id"] = "resnet50-224px"
        overwatch.info(f"Detected ResNet VLA ID. Overriding vision backbone to: resnet50-224px")
    elif "resnet101" in vla_id.lower() and not is_resnet_checkpoint:
        vla_cfg["vision_backbone_id"] = "resnet101-224px"
        overwatch.info(f"Detected ResNet VLA ID. Overriding vision backbone to: resnet101-224px")
    elif is_resnet_checkpoint:
        # If it's a ResNet checkpoint, make sure we use the ResNet backbone
        if "resnet50" in checkpoint_path.lower():
            vla_cfg["vision_backbone_id"] = "resnet50-224px"
            overwatch.info(f"Detected ResNet checkpoint. Using ResNet50 backbone.")
        elif "resnet101" in checkpoint_path.lower():
            vla_cfg["vision_backbone_id"] = "resnet101-224px"
            overwatch.info(f"Detected ResNet checkpoint. Using ResNet101 backbone.")

    # if base vlm is a folder, load its config.json (only works for native format!)
    # this might happen if you start a run who's base vlm is from a folder instead of from hf
    if os.path.isdir(base_vlm):
        with open(Path(base_vlm) / "config.json", "r") as f:
            base_cfg = json.load(f)["model"]
            base_vlm = base_cfg["model_id"]

    base_vlm_id = Path(str(base_vlm)).name
    base_vlm_id = BASE_VLM_ID_ALIASES.get(base_vlm_id, base_vlm_id)
    overwatch.info(f"Base vlm: {base_vlm} (registry ID: {base_vlm_id})")
    model_cfg = ModelConfig.get_choice_class(base_vlm_id)()

    # Load Dataset Statistics for Action Denormalization
    # TODO (ajaysri) :: Make this dynamic
    with open(dataset_statistics_json, "r") as f:
        norm_stats = json.load(f)

    # = Load Individual Components necessary for Instantiating a VLA (via base VLM components) =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg.model_id}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg.vision_backbone_id}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg.llm_backbone_id}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg.arch_specifier}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    if image_sequence_len is None:
        if hasattr(model_cfg, "image_sequence_len"):
            image_sequence_len = model_cfg.image_sequence_len
        else:
            image_sequence_len = 1

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{vla_cfg.get('vision_backbone_id', model_cfg.vision_backbone_id)}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        vla_cfg.get('vision_backbone_id', model_cfg.vision_backbone_id),
        model_cfg.image_resize_strategy,
        image_sequence_len,
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(
        f"{'Randomly initializing' if random_llm_weights else 'Loading Pretrained'} "
        f"LLM [bold]{model_cfg.llm_backbone_id}[/] via HF Transformers"
    )
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg.llm_backbone_id,
        llm_max_length=model_cfg.llm_max_length,
        hf_token=hf_token,
        inference_mode=not load_for_training,
        pretrained=not random_llm_weights,  # Add this parameter to control weight initialization
    )

    # Create Action Tokenizer
    ac_tokenizer = vla_cfg["action_tokenizer"] if "action_tokenizer" in vla_cfg else "action_tokenizer"
    action_tokenizer: ActionTokenizer = ACTION_TOKENIZERS[ac_tokenizer](llm_backbone.get_tokenizer())

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLA [bold blue]{model_cfg.model_id}[/] from Checkpoint")
    vla = OpenVLA.from_pretrained(
        checkpoint_pt,
        model_cfg.model_id,
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg.arch_specifier,
        freeze_weights=not load_for_training,
        norm_stats=norm_stats,
        action_tokenizer=action_tokenizer,
        aux_context_freq=aux_context_freq,
    )

    return vla

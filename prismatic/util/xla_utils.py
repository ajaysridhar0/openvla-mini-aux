"""
xla_utils.py

General XLA + XLA FSDP utilities.
"""

from functools import partial
from typing import Callable, List, Set, Type

import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy

from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# Note =>> ViT throws FSDP Error because of modified forward pass (https://github.com/pytorch/pytorch/issues/109385)
#          We are still wrapping each ViT Block (and ViTs are small), so shouldn't matter too much in practice
XLA_EXCLUDE_WRAP_MODULES = {nn.ModuleList, nn.ModuleDict, VisionTransformer}


# === XLA FSDP Wrapping Utilities ===
def _xla_module_wrap_policy(
    module: nn.Module, recurse: bool, unwrapped_params: int, module_classes: Set[Type[nn.Module]]
) -> bool:
    """Wraps every module that is an instance of the types specified in `module_classes`."""
    if recurse:
        return True

    return isinstance(module, tuple(module_classes)) and not isinstance(module, tuple(XLA_EXCLUDE_WRAP_MODULES))


def _xla_or_policy(module: nn.Module, recurse: bool, unwrapped_params: int, policies: List[Callable[..., bool]]) -> bool:
    return any(policy(module=module, recurse=recurse, unwrapped_params=unwrapped_params) for policy in policies)


# === XLA Imports (Shielded) ===
if overwatch.use_tpu:
    from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy as xla_transformer_auto_wrap_policy

    TORCH_NATIVE_TO_XLA_WRAP_FN_MAP = {
        transformer_auto_wrap_policy: xla_transformer_auto_wrap_policy,
        _module_wrap_policy: _xla_module_wrap_policy,
        _or_policy: _xla_or_policy,
    }


def convert_to_xla_wrapping_policy(native_wrapping_policy: Callable[..., bool]) -> Callable[..., bool]:
    """Converts a Torch Native FSDP Wrapping Policy to an XLA Wrapping Policy (different naming convention...)."""
    assert isinstance(native_wrapping_policy, partial), "Conversion assumes `native_wrapping_policy` is a partial!"
    fn, kwargs = native_wrapping_policy.func, native_wrapping_policy.keywords

    # Check for nested policies --> recurse!
    if "policies" in kwargs:
        kwargs["policies"] = [convert_to_xla_wrapping_policy(policy) for policy in kwargs["policies"]]

    return partial(TORCH_NATIVE_TO_XLA_WRAP_FN_MAP[fn], **kwargs)

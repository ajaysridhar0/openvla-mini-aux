"""
fsdp_xla.py

Core class definition for a strategy implementing PyTorch XLA Fully Sharded Data Parallel Training (with support for
fine-grained control over wrapping policies and mixed precision per component).

Inherits from `FSDPStrategy` and implements all XLA-specific semantics.
"""

import glob
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLAMetrics
from prismatic.training.strategies.fsdp import FSDPStrategy

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === XLA Imports (Shielded) ===
if overwatch.use_tpu:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP
    from torch_xla.distributed.fsdp import checkpoint_module, consolidate_sharded_model_checkpoints
    from torch_xla.distributed.fsdp.wrap import lambda_auto_wrap_policy, recursive_wrap

    from prismatic.util.xla_utils import convert_to_xla_wrapping_policy


class FSDPXLAStrategy(FSDPStrategy):

    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        assert overwatch.use_tpu, "Cannot set up `FSDPXLAStrategy` if not on TPU!"

        # Note =>> XLA requires adding Gradient Checkpointing wrappers *before* FSDP default wrapping!
        if self.enable_gradient_checkpointing:
            # Similar assumption to base FSDPStrategy =>> only checkpoint LLM blocks!
            def check_fn(submodule: nn.Module) -> bool:
                return isinstance(submodule, self.llm_transformer_layer_cls)

            # Apply (recursive) wrapping policy!
            gradient_checkpointing_wrap_policy = partial(lambda_auto_wrap_policy, lambda_fn=check_fn)
            recursive_wrap(
                self.vlm,
                gradient_checkpointing_wrap_policy,
                checkpoint_module,
                ignored_modules=set(),
                ignored_params=set(),
                only_wrap_children=True,
            )

        # Iteratively Assemble FSDP Wrapping Policy by fetching the wrapping policies for each backbone/constituent
        #   =>> XLA uses different naming convention; convert!
        vlm_fsdp_wrapping_policy = self.vlm.get_fsdp_wrapping_policy()
        vlm_fsdp_wrapping_policy = convert_to_xla_wrapping_policy(vlm_fsdp_wrapping_policy)

        # Assemble Mixed Precision Parameters =>> XLA FSDP expects these directly in FSDP initializer!
        if self.enable_mixed_precision_training and self.mixed_precision_dtype == torch.bfloat16:
            reduce_buffer_dtype = torch.bfloat16 if not self.reduce_in_full_precision else torch.float32
            compute_dtype = torch.bfloat16
        else:
            reduce_buffer_dtype, compute_dtype = torch.float32, torch.float32

        # <FSDP> => note that FSDP will automatically take care of device placement (similar to `autocast`)
        self.vlm = XLAFSDP(
            self.vlm,
            auto_wrap_policy=vlm_fsdp_wrapping_policy,
            compute_dtype=compute_dtype,
            buffer_dtype=reduce_buffer_dtype,
            fp32_reduce_scatter=(compute_dtype != torch.float32 and reduce_buffer_dtype == torch.float32),
        )

        # Barrier =>> Sharding takes a minute?
        self.barrier("Finished Sharding")

        # Create Optimizer & LR Scheduler
        #   =>> TODO (kpertsch) :: Try `syncfree` Optimizers for improved AMP-XLA
        #                          See: https://github.com/pytorch/xla/blob/master/torch_xla/amp/syncfree/adamw.py
        num_training_steps, num_warmup_steps = self.create_optimizer(n_train_examples)

        # Finalize Setup =>> Log!
        overwatch.info(
            "XLA FSDP Full-Shard Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {overwatch.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone XLA FSDP Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> Use XLA FSDP Mixed Precision = {self.enable_mixed_precision_training}\n"
            f"                 |-> Compute Precision = {compute_dtype}\n"
            f"                 |-> Reduction & Buffer Precision = {reduce_buffer_dtype}\n\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            f"         |-> LR Scheduler Type = {self.lr_scheduler_type}\n"
            f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n"
        )

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
        keep_latest_only: bool = True,
    ) -> None:
        """Save a checkpoint to the `run_dir` only containing the state_dicts for trainable parameters by default."""
        assert isinstance(self.vlm, XLAFSDP), "FSDPXLAStrategy.save_checkpoint assumes VLM is already wrapped in FSDP!"

        # Saving vision_backbone weights crashes training -- remove for now
        # TODO(karl): figure out how to save vision backbones properly
        raw_vlm_state_dict = self.vlm.state_dict()
        vlm_state_dict = {}
        for key, param in raw_vlm_state_dict.items():
            if "vision_backbone" not in key:
                vlm_state_dict[key] = param

        # Specify Checkpoint Path / Naming
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_prefix = checkpoint_dir / f"step-{global_step:06d}"
        checkpoint_postfix = f"-rank-{overwatch.rank():08d}-of-{overwatch.world_size():08d}.pt"

        # Save with Shard Metadata
        ckpt = {
            "model": vlm_state_dict,
            "shard_metadata": self.vlm.get_shard_metadata(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        xm.save(ckpt, str(checkpoint_prefix) + str(checkpoint_postfix), master_only=False)
        self.barrier("Saved checkpoint shards")

        # Consolidate Checkpoints -- will be saved as `{checkpoint_prefix}_consolidated.pt`
        if overwatch.is_rank_zero():
            consolidate_sharded_model_checkpoints(ckpt_prefix=str(checkpoint_prefix), ckpt_suffix="-rank-*-of-*.pt")

        self.barrier("Finished checkpoint consolidation")

        # Optionally delete previous checkpoints to save memory
        if keep_latest_only and global_step > self.save_interval:
            prev_checkpoint_prefix = checkpoint_dir / f"step-{global_step - self.save_interval:06d}"
            prev_checkpoint_path = str(prev_checkpoint_prefix) + str(checkpoint_postfix)
            prev_consolidated_path = str(prev_checkpoint_prefix) + "_consolidated.pt"
            if os.path.exists(prev_checkpoint_path):
                os.remove(prev_checkpoint_path)
            if overwatch.is_rank_zero() and os.path.exists(prev_consolidated_path):
                os.remove(prev_consolidated_path)
            self.barrier("Finished deleting previous checkpoint")

    def load_checkpoint(self, resume_dir: Path, metrics: Union[Metrics, VLAMetrics]):
        assert isinstance(self.vlm, XLAFSDP), "FSDPXLAStrategy.load_checkpoint assumes VLM is already wrapped in FSDP!"
        # Get latest checkpoint
        checkpoint_files = glob.glob(str(resume_dir / "step-*-rank-*-of-*.pt"))
        latest_step = int(os.path.basename(sorted(checkpoint_files)[-1]).split("-")[1])
        ckpt_postfix = f"step-{latest_step:06d}-rank-{overwatch.rank():08d}-of-{overwatch.world_size():08d}.pt"

        # Load model, optimizer and LR schedule state dicts
        overwatch.info(f"Loading checkpoint for step {latest_step}...")
        ckpt = torch.load(resume_dir / ckpt_postfix, map_location="cpu")

        # Vision backbone weights currently not saved for XLA devices
        model_state_dict = ckpt["model"]
        for key, param in self.vlm.state_dict().items():
            if "vision_backbone" in key:
                model_state_dict[key] = param
        self.vlm.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

        # Set step in metrics
        metrics.commit(global_step=latest_step)
        self.barrier(f"Loaded checkpoint for step {latest_step} from {resume_dir}.")

    @staticmethod
    def wrap_dataloader(dataloader: DataLoader) -> DataLoader:
        return pl.MpDeviceLoader(dataloader, xm.xla_device())

    def finish_vla_train_step(
        self,
        metrics: VLAMetrics,
        progress: tqdm,
        log_info: Dict[str, Any],
    ) -> None:
        # Wrap Logging / Push to Trackers in Step Closure (TPU =>> CPU Transfer)
        xm.add_step_closure(self.update_vla_logs_and_progress, (metrics, progress, log_info))

    @staticmethod
    def barrier(tag: Optional[str] = "Default Rendezvous") -> None:
        xm.rendezvous(f"Rendezvous: {tag}")

    @property
    def device_type(self) -> str:
        return "xla"

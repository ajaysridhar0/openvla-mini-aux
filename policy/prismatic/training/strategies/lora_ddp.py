"""
ddp.py

Core class definition for a strategy implementing Torch native Distributed Data Parallel Training; note that on most
GPU hardware and LLM backbones >= 5-7B parameters, DDP training will OOM, which is why we opt for FSDP.
"""

import shutil
from pathlib import Path
from typing import Optional, Callable

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers.optimization import get_constant_schedule, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers import Conv1D
from peft import LoraConfig, get_peft_model
import re

from prismatic.overwatch import initialize_overwatch
from prismatic.training.strategies.base_strategy import TrainingStrategy
from prismatic.models.vlms import PrismaticVLM

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

def get_specific_layer_names(model):
    # Create a list to store the layer names
    layer_names = []
    
    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        # Check if the module is an instance of the specified layers
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
            # model name parsing 

            # layer_names.append('.'.join(name.split('.')[4:]).split('.')[0])
            layer_names.append(name)
    
    return layer_names


class LoraDDPStrategy(TrainingStrategy):
    def __init__(
            self,
            vlm: PrismaticVLM,
            device_id: int,
            stage: str,
            epochs: int,
            max_steps: Optional[int],
            global_batch_size: int,
            per_device_batch_size: int,
            learning_rate: float,
            weight_decay: float,
            max_grad_norm: float,
            lr_scheduler_type: str,
            warmup_ratio: float,
            enable_gradient_checkpointing: bool = False,
            enable_mixed_precision_training: bool = True,
            reduce_in_full_precision: bool = False,
            mixed_precision_dtype: torch.dtype = torch.bfloat16,
            worker_init_fn: Optional[Callable[[int], None]] = None,
            lora_rank: int = 64,
            lora_alpha: int = 16,
            lora_dropout: float = 0.0,
            save_every_n_steps: Optional[int] = None,
        ) -> None:
            super().__init__(
                vlm=vlm,
                device_id=device_id,
                stage=stage,
                epochs=epochs,
                max_steps=max_steps,
                global_batch_size=global_batch_size,
                per_device_batch_size=per_device_batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                lr_scheduler_type=lr_scheduler_type,
                warmup_ratio=warmup_ratio,
                enable_gradient_checkpointing=False,
                enable_mixed_precision_training=enable_mixed_precision_training,
                reduce_in_full_precision=reduce_in_full_precision,
                mixed_precision_dtype=mixed_precision_dtype,
                worker_init_fn=worker_init_fn,
                save_every_n_steps=save_every_n_steps,
            )

            # target_modules = [name for name, _ in vlm.named_modules() if pattern.search(name)]
            target_modules = get_specific_layer_names(vlm)

            self.lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=min(lora_rank, lora_alpha),
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                init_lora_weights="gaussian",
            )
       
    @overwatch.rank_zero_only
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """Save a checkpoint to the `run_dir` only containing the state_dicts for trainable parameters by default."""
        assert isinstance(self.vlm, DDP), "save_checkpoint assumes VLM is already wrapped in DDP!"

        optimizer_state_dict = self.optimizer.state_dict()

        # Set Checkpoint Path =>> Embed *minimal* training statistics!
        checkpoint_dir = run_dir / "checkpoints"
        if train_loss is None:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf"
        else:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}"

        # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
        self.vlm.module.save_pretrained(checkpoint_path)
        torch.save({"optimizer": optimizer_state_dict}, checkpoint_path / "optimizer.pt")
        # shutil.copytree(checkpoint_path, checkpoint_dir / "latest-checkpoint")


    def run_setup(self, run_dir: Path, n_train_examples: int, vla_batch_size: int = None) -> None:
        # Move to Device =>> Note parameters are in full precision (*mixed precision* will only autocast as appropriate)
        overwatch.info("Placing Entire VLM (Vision Backbone, LLM Backbone, Projector Weights) on GPU", ctx_level=1)
        self.vlm.to(dtype=self.mixed_precision_dtype, device=self.device_id)
        self.vlm = get_peft_model(self.vlm, self.lora_config)
        self.vlm.print_trainable_parameters()

        trainable_params, total_params = self.vlm.get_nb_trainable_parameters()
        overwatch.info(f"{trainable_params / 1000000:.2f}M trainable LoRA parameters out of {total_params / 1000000:.2f}M total", ctx_level=1)

        # Wrap with Distributed Data Parallel
        #   => Note: By default, wrapping naively with DDP(self.vlm) will initialize a *separate* buffer on GPU that
        #            is the same size/dtype as the model parameters; this will *double* GPU memory!
        # - stackoverflow.com/questions/68949954/model-takes-twice-the-memory-footprint-with-distributed-data-parallel
        overwatch.info("Wrapping VLM with Distributed Data Parallel", ctx_level=1)
        self.vlm = DDP(self.vlm, device_ids=[self.device_id], gradient_as_bucket_view=True, find_unused_parameters=True)

        # Create Optimizer and LR Scheduler =>> note that most of the LR Schedulers we use require `max_steps/epochs`
        #   => Optimizer should only operate on parameters that are *unfrozen* / trainable!
        trainable_params = [param for param in self.vlm.parameters() if param.requires_grad]
        if self.max_steps is None:
            batch_size = vla_batch_size if vla_batch_size is not None else self.global_batch_size
            num_training_steps = (n_train_examples * self.epochs) // batch_size
        else:
            num_training_steps = self.max_steps

        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            if self.warmup_steps is not None:
                num_warmup_steps = self.warmup_steps
            else:
                num_warmup_steps = int(num_training_steps * self.warmup_ratio)
                
            assert self.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0

        elif self.lr_scheduler_type == "constant":
            num_warmup_steps = 0

            assert self.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            self.lr_scheduler = get_constant_schedule(self.optimizer)

        elif self.lr_scheduler_type == "linear-warmup+constant":
             # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            if self.warmup_steps is not None:
                num_warmup_steps = self.warmup_steps
            else:
                num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            # Default AdamW w/ specified LR & Linear Warmup / Cosine Decay & Weight Decay
            #   => Create Parameter Groups --> bias terms, normalization layer parameters shouldn't be decayed!
            decay, no_decay = [], []
            for name, param in self.vlm.named_parameters():
                if not param.requires_grad:
                    continue

                # Check on any parameters with fewer than 2 dimensions or with "bias" in the name
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)

            # Build Parameter Groups
            groups = [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

            # Create Optimizer & LR Scheduler
            self.optimizer = AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0

        else:
            raise ValueError(f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!")

        # Finalize Setup =>> Log
        overwatch.info(
            "DDP Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {overwatch.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> Use Native AMP = {self.enable_mixed_precision_training} ({self.mixed_precision_dtype})\n\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            f"         |-> LR Scheduler Type = {self.lr_scheduler_type}\n"
            f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n"
        )

    def clip_grad_norm(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.vlm.parameters(), max_norm=self.max_grad_norm)

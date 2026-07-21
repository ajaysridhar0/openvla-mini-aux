"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.datasets import AUX_TASK_QA_FUNCTIONS


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

AUX_TASK_NAMES = sorted(AUX_TASK_QA_FUNCTIONS.keys())
AUX_TASK_NAMES = ["action"] + AUX_TASK_NAMES


def find_qa_segments(labels):
    """Find start and end indices of non-ignored segments in labels tensor.
    
    Args:
        labels: Tensor of shape [B, L] or [L]
        
    Returns:
        List[List[Tuple[int, int]]]: List of segments for each batch item, 
            where each segment is (start_idx, end_idx)
    """
    # Handle single sequence case
    if labels.ndim == 1:
        labels = labels.unsqueeze(0)
        
    # Convert to numpy for easier processing if needed
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    batch_segments = []
    
    # Process each sequence in the batch
    for batch_idx in range(len(labels)):
        segments = []
        current_start = None
        
        for i in range(len(labels[batch_idx])):
            if labels[batch_idx][i] != -100:
                if current_start is None:
                    current_start = i
            elif current_start is not None:
                segments.append((current_start, i))
                current_start = None
                
        # Handle case where last segment extends to end
        if current_start is not None:
            segments.append((current_start, len(labels[batch_idx])))
            
        batch_segments.append(segments)
    
    # If input was 1D, return just the segments without batch dimension
    # if labels.shape[0] == 1:
    #     return batch_segments[0]
        
    return batch_segments


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
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
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        save_every_n_steps: Optional[int] = None,
        **_: str,
    ) -> None:
        self.vlm, self.device_id, self.stage = vlm, device_id, stage

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls
        self.vision_backbone = self.vlm.vision_backbone

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # how often to save checkpoints
        self.save_every_n_steps = save_every_n_steps
        if save_every_n_steps is not None:
            assert save_every_n_steps > 0

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss
                        
                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                            return
                        elif (
                            self.save_every_n_steps is not None
                            and (metrics.global_step + 1) % self.save_every_n_steps == 0
                        ):

                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()

    # === VLA Training ===

    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer: ActionTokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
        sequence_level_decoding: bool = False,
    ) -> None:
        """
        Run the VLA training loop for the given `dataset` and `collator`.
        
        Args:
            sequence_level_decoding: If True, maintains sequence structure when decoding action tokens,
                passing List[List[int]] to decode_token_ids_to_actions instead of flattening.
        """
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()

            # Zero Gradients (just in case)
            self.optimizer.zero_grad()

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            for batch in dataloader:
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                with torch.autocast(
                    "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    output: CausalLMOutputWithPast = self.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    # The model returns the aggregate loss used for backpropagation.
                    loss = output.loss
                    transform_types = batch["transform_type"].to(output.loss.device)

                # Commit Loss =>> Backward!
                metrics.commit(loss=loss)
                loss.backward()

                # === Compute Action Token Accuracy & L1 Loss ===

                # To compute action token accuracy, we need to identify the locations of the action tokens
                # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                # insert `self.vlm.vision_backbone.num_patches` at index 1.
                #
                # Computing `action_prediction_accuracy` is then pretty straightforward:
                #   1) Extract "aligned" predictions & labels
                #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!
                unique_transform_types = transform_types.unique(dim=0)
                # remove all rows that don't have 0
                unique_transform_types = unique_transform_types[(unique_transform_types == 0).any(dim=1)]
                num_unique_transform_types = len(unique_transform_types)
                for i in range(num_unique_transform_types + 1):
                    if i < num_unique_transform_types:
                        transform_type = unique_transform_types[i]
                        # filter out all but the current transform type
                        transform_type_mask = torch.all(transform_types == transform_type, dim=1)
                        transform_types_str_list = [AUX_TASK_NAMES[int(t)] for t in transform_type if t >= 0]
                        transform_type_str = " -> ".join(transform_types_str_list)
                    else:
                        transform_type_mask = (transform_types == 0).any(dim=1)
                        transform_type_str = "all"
                        
                    action_preds = output.logits[:, self.vision_backbone.num_patches : -1].argmax(dim=2)
                    action_preds = action_preds[transform_type_mask]
                    action_gt = batch["labels"][:, 1:].to(action_preds.device)
                    action_gt = action_gt[transform_type_mask]

                    # find qa segments
                    qa_segments = find_qa_segments(action_gt)
                    
                    if action_preds.numel() > 0 and action_gt.numel() > 0:
                        if transform_type_str != "all":
                            # Split the transform type to get individual prediction types
                            pred_types = transform_type_str.split(" -> ")
                            
                            # For each prediction type in the sequence, compute accuracy
                            for pred_idx, pred_type in enumerate(pred_types):
                                # Create mask for current prediction type using qa_segments
                                current_mask = torch.zeros_like(action_gt, dtype=torch.bool)
                                for batch_idx in range(len(qa_segments)):
                                    start, end = qa_segments[batch_idx][pred_idx]
                                    current_mask[batch_idx, start:end] = True
                                    
                                
                                # Compute accuracy for current prediction type
                                correct_preds = (action_preds == action_gt) & current_mask
                                current_accuracy = correct_preds.sum().float() / current_mask.sum().float()
                                
                                # Commit accuracy for this prediction type
                                metrics.commit_for_dataset(
                                    dataset_name=transform_type_str, 
                                    **{f"{pred_type}_accuracy": current_accuracy}
                                )
                                
                                # Per-dataset metrics (only on rank zero)
                                if overwatch.is_rank_zero():
                                    datasets = set(batch["dataset_names"])
                                    if len(datasets) > 1:
                                        for ds in datasets:
                                            ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                                            ds_mask = ds_mask[transform_type_mask.cpu()]
                                            ds_correct_preds = correct_preds[ds_mask]
                                            
                                            ds_current_mask = current_mask[ds_mask]
                                            ds_accuracy = ds_correct_preds.sum().float() / ds_current_mask.sum().float()
                                            metrics.commit_for_dataset(
                                                dataset_name=f"{ds.decode()}/{transform_type_str}",
                                                **{f"{pred_type}_accuracy": ds_accuracy}
                                            )
                            
                            # Compute L1 loss only if 'action' is one of the prediction types
                            if 'action' in pred_types:
                                action_idx = pred_types.index('action')
                                action_mask = torch.zeros_like(action_gt, dtype=torch.bool)
                                for batch_idx in range(len(qa_segments)):
                                    start, end = qa_segments[batch_idx][action_idx]
                                    action_mask[batch_idx, start:end-2] = True

                                if sequence_level_decoding:
                                    # Maintain sequence structure for each batch item
                                    pred_sequences = []
                                    gt_sequences = []
                                    for b in range(action_preds.size(0)):
                                        batch_mask = action_mask[b]
                                        pred_sequences.append(action_preds[b][batch_mask].tolist())
                                        gt_sequences.append(action_gt[b][batch_mask].tolist())
                                    
                                    continuous_actions_pred = torch.tensor(
                                        action_tokenizer.decode_token_ids_to_actions(pred_sequences)
                                    )
                                    continuous_actions_gt = torch.tensor(
                                        action_tokenizer.decode_token_ids_to_actions(gt_sequences)
                                    )
                                else:
                                    # Original flattened behavior
                                    continuous_actions_pred = torch.tensor(
                                        action_tokenizer.decode_token_ids_to_actions(action_preds[action_mask].cpu().numpy())
                                    )
                                    continuous_actions_gt = torch.tensor(
                                        action_tokenizer.decode_token_ids_to_actions(action_gt[action_mask].cpu().numpy())
                                    )

                                action_l1_loss = torch.nn.functional.l1_loss(
                                    continuous_actions_pred,
                                    continuous_actions_gt,
                                )
                                metrics.commit_for_dataset(dataset_name=transform_type_str, l1_loss=action_l1_loss)

                                # Per-dataset L1 loss
                                if overwatch.is_rank_zero():
                                    datasets = set(batch["dataset_names"])
                                    if len(datasets) > 1:
                                        for ds in datasets:
                                            ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                                            ds_mask = ds_mask[transform_type_mask.cpu()]
                                            ds_action_mask = action_mask[ds_mask]
                                            ds_preds = action_preds[ds_mask][ds_action_mask]
                                            ds_gt = action_gt[ds_mask][ds_action_mask]
                                            
                                            if ds_preds.numel() > 0:
                                                if sequence_level_decoding:
                                                    # Maintain sequence structure for dataset-specific predictions
                                                    ds_pred_sequences = []
                                                    ds_gt_sequences = []
                                                    for b in range(ds_preds.size(0)):
                                                        ds_pred_sequences.append(ds_preds[b].tolist())
                                                        ds_gt_sequences.append(ds_gt[b].tolist())
                                                    
                                                    ds_continuous_pred = torch.tensor(
                                                        action_tokenizer.decode_token_ids_to_actions(ds_pred_sequences)
                                                    )
                                                    ds_continuous_gt = torch.tensor(
                                                        action_tokenizer.decode_token_ids_to_actions(ds_gt_sequences)
                                                    )
                                                else:
                                                    # Original flattened behavior
                                                    ds_continuous_pred = torch.tensor(
                                                        action_tokenizer.decode_token_ids_to_actions(ds_preds.cpu().numpy())
                                                    )
                                                    ds_continuous_gt = torch.tensor(
                                                        action_tokenizer.decode_token_ids_to_actions(ds_gt.cpu().numpy())
                                                    )
                                                ds_l1_loss = torch.nn.functional.l1_loss(ds_continuous_pred, ds_continuous_gt)
                                                metrics.commit_for_dataset(
                                                    dataset_name=f"{ds.decode()}/{transform_type_str}",
                                                    l1_loss=ds_l1_loss
                                                )
                        
                        else:
                            # For "all", use qa_segments to precisely identify action token locations
                            qa_segments = find_qa_segments(action_gt)
                            action_mask = torch.zeros_like(action_gt, dtype=torch.bool)
                            
                            # For "all" we only care about the last segment contains actions (*always true)
                            for batch_idx in range(len(qa_segments)):
                                if qa_segments[batch_idx]:  # if there are any segments
                                    start, end = qa_segments[batch_idx][-1]  # take last segment
                                    # Exclude the last 2 tokens which are typically EOS/padding
                                    action_mask[batch_idx, start:end-2] = True
                            
                            correct_preds = (action_preds == action_gt) & action_mask
                            action_accuracy = correct_preds.sum().float() / action_mask.sum().float()
                            
                            if sequence_level_decoding:
                                # Maintain sequence structure for each batch item
                                pred_sequences = []
                                gt_sequences = []
                                for b in range(action_preds.size(0)):
                                    batch_mask = action_mask[b]
                                    pred_sequences.append(action_preds[b][batch_mask].tolist())
                                    gt_sequences.append(action_gt[b][batch_mask].tolist())
                                
                                continuous_actions_pred = torch.tensor(
                                    action_tokenizer.decode_token_ids_to_actions(pred_sequences),
                                    device=action_preds.device
                                ).clone().detach()
                                continuous_actions_gt = torch.tensor(
                                    action_tokenizer.decode_token_ids_to_actions(gt_sequences),
                                    device=action_gt.device
                                ).clone().detach()
                            else:
                                # Original flattened behavior
                                continuous_actions_pred = torch.tensor(
                                    action_tokenizer.decode_token_ids_to_actions(action_preds[action_mask].cpu().numpy()),
                                    device=action_preds.device
                                ).clone().detach()
                                continuous_actions_gt = torch.tensor(
                                    action_tokenizer.decode_token_ids_to_actions(action_gt[action_mask].cpu().numpy()),
                                    device=action_gt.device
                                ).clone().detach()
                            
                            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
                            metrics.commit(action_accuracy=action_accuracy, l1_loss=action_l1_loss, update_step_time=True)

                            # Per-dataset metrics for "all" (only on rank zero)
                            if overwatch.is_rank_zero():
                                datasets = set(batch["dataset_names"])
                                if len(datasets) > 1:
                                    for ds in datasets:
                                        # Create dataset mask and expand to match tensor dimensions
                                        ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]], 
                                                             device=action_preds.device)
                                        ds_mask = ds_mask[transform_type_mask.cpu()]
                                        ds_mask = ds_mask.unsqueeze(1).expand(-1, action_preds.size(1))
                                        
                                        # Apply both dataset mask and action mask
                                        combined_mask = ds_mask & action_mask
                                        
                                        if combined_mask.sum() > 0:
                                            ds_accuracy = correct_preds[combined_mask].sum().float() / combined_mask.sum().float()
                                            
                                            if sequence_level_decoding:
                                                # Maintain sequence structure for "all" dataset-specific predictions
                                                ds_pred_sequences = []
                                                ds_gt_sequences = []
                                                for b in range(action_preds.size(0)):
                                                    batch_combined_mask = combined_mask[b]
                                                    if batch_combined_mask.any():
                                                        ds_pred_sequences.append(action_preds[b][batch_combined_mask].tolist())
                                                        ds_gt_sequences.append(action_gt[b][batch_combined_mask].tolist())
                                                
                                                ds_continuous_pred = torch.tensor(
                                                    action_tokenizer.decode_token_ids_to_actions(ds_pred_sequences),
                                                    device=action_preds.device
                                                ).clone().detach()
                                                ds_continuous_gt = torch.tensor(
                                                    action_tokenizer.decode_token_ids_to_actions(ds_gt_sequences),
                                                    device=action_gt.device
                                                ).clone().detach()
                                            else:
                                                # Original flattened behavior
                                                ds_continuous_pred = torch.tensor(
                                                    action_tokenizer.decode_token_ids_to_actions(action_preds[combined_mask].cpu().numpy()),
                                                    device=action_preds.device
                                                ).clone().detach()
                                                ds_continuous_gt = torch.tensor(
                                                    action_tokenizer.decode_token_ids_to_actions(action_gt[combined_mask].cpu().numpy()),
                                                    device=action_gt.device
                                                ).clone().detach()
                                            
                                            ds_l1_loss = torch.nn.functional.l1_loss(ds_continuous_pred, ds_continuous_gt)
                                            metrics.commit_for_dataset(
                                                dataset_name=f"{ds.decode()}/all",
                                                action_accuracy=ds_accuracy,
                                                l1_loss=ds_l1_loss
                                            )

                # === Gradient Step ===

                # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                self.clip_grad_norm()

                # Optimizer & LR Scheduler Step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Compute epoch value using number of completed gradient steps
                epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                # Push Metrics
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                # Check for Save Interval or Max Steps & Save Checkpoint
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                    (metrics.global_step % save_interval) == 0
                ):
                    self.save_checkpoint(
                        metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                    )
                    dist.barrier()

                    if terminate:
                        return

                # Update Progress Bar
                progress.update()
                progress.set_description(status)

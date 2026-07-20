"""
resnet_backbone.py

Vision backbone that uses ResNet architecture, preserving spatial information for robotics applications.
"""

from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
from timm.models.resnet import ResNet
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Resize

from prismatic.models.backbones.vision.base_vision import (
    ImageTransform,
    LetterboxPad,
    VisionBackbone,
    compute_sequence_patches,
)

# Registry =>> Supported ResNet Backbones (from TIMM)
RESNET_VISION_BACKBONES = {
    "resnet50-224px": "resnet50.a1_in1k",  # ImageNet-1K pretrained ResNet-50
    "resnet101-224px": "resnet101.a1_in1k",  # ImageNet-1K pretrained ResNet-101
}

class ResNetBackbone(VisionBackbone):
    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        image_sequence_len: int = 1,
        grid_size: int = 14,  # Default grid size to match ViT-14 patch size
    ) -> None:
        super().__init__(
            vision_backbone_id,
            image_resize_strategy,
            default_image_size=default_image_size,
            image_sequence_len=image_sequence_len,
        )
        self.grid_size = grid_size
        self.timm_path_or_url = RESNET_VISION_BACKBONES[vision_backbone_id]

        # Initialize ResNet backbone
        self.featurizer: ResNet = timm.create_model(
            self.timm_path_or_url,
            pretrained=True,
            features_only=True,  # Get intermediate feature maps
            out_indices=None,  # Get all feature levels, then we'll select the appropriate one
        )
        self.featurizer.eval()
        
        # Get all feature stages to determine the correct one to use
        dummy_input = torch.zeros(1, 3, default_image_size, default_image_size)
        with torch.no_grad():
            output_features = self.featurizer(dummy_input)
        
        # Select the last feature stage which should have the highest channel count
        last_stage_idx = len(output_features) - 1
        feature_channels = output_features[last_stage_idx].shape[1]
        
        # Re-initialize with the correct out_indices
        self.featurizer: ResNet = timm.create_model(
            self.timm_path_or_url,
            pretrained=True,
            features_only=True,
            out_indices=(last_stage_idx,),
        )
        self.featurizer.eval()
        
        # Add spatial adaptive pooling to get desired grid size
        self.spatial_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        
        # Project features to match desired embedding dimension (e.g., 1024 for compatibility)
        self.feature_projection = nn.Conv2d(feature_channels, 1024, 1)

        # Get Config for image transforms
        self.data_cfg = timm.data.resolve_model_data_config(self.featurizer)
        self.data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        # Initialize Default Image Transform
        default_image_transform = timm.data.create_transform(**self.data_cfg, is_training=False)

        # Switch on `image_resize_strategy`
        if self.image_resize_strategy == "resize-naive":
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert isinstance(default_image_transform.transforms[0], Resize)

            target_size = (self.default_image_size, self.default_image_size)
            self.image_transform = Compose(
                [
                    Resize(target_size, interpolation=default_image_transform.transforms[0].interpolation),
                    *default_image_transform.transforms[1:],
                ]
            )

        elif self.image_resize_strategy == "resize-crop":
            self.image_transform = default_image_transform

        elif self.image_resize_strategy == "letterbox":
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert "mean" in self.data_cfg, "TIMM `data_cfg` missing image normalization mean!"

            # Compute Padding Fill Value (rescaled normalization mean if applicable)
            fill = tuple([int(x * 255) for x in self.data_cfg["mean"]])

            # Build New Transform
            self.image_transform = Compose([LetterboxPad(fill), *default_image_transform.transforms])

        else:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps the ResNet backbone."""
        return partial(_module_wrap_policy, module_classes={ResNet})

    def process_single_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Process a single image (no sequence dimension)."""
        # Get feature map from ResNet
        features_list = self.featurizer(pixel_values)
        
        # Use the last (and only) feature map
        features = features_list[0]  # We configured out_indices with only one value
        
        # Apply spatial pooling first to get desired grid size
        features = self.spatial_pool(features)
        
        # Then project to desired embedding dimension
        features = self.feature_projection(features)
        
        # Reshape to sequence of patches
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1)  # [B, grid_size, grid_size, C]
        features = features.reshape(B, H * W, C)  # [B, grid_size * grid_size, C]
        
        return features

    def forward(self, pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass for pixel features. Handles both single images and image sequences.
        
        For single images (image_sequence_len=1):
        - Takes input of shape [B, C, H, W]
        - Returns output of shape [B, grid_size * grid_size, embed_dim]
        
        For image sequences (image_sequence_len>1):
        - Takes input of shape [B, T, C, H, W] where T is the sequence length
        - Processes each frame separately
        - Concatenates all frame features along the patch dimension
        - Returns output of shape [B, (grid_size * grid_size * T), embed_dim]
        
        Args:
            pixel_values: Either a tensor of shape [B, C, H, W] or [B, T, C, H, W],
                          or a dict with key "img" containing such a tensor
                          
        Returns:
            A tensor of shape [B, grid_size * grid_size, embed_dim] (for single images)
            or [B, (grid_size * grid_size * T), embed_dim] (for sequences)
        """
        # Ensure input is a tensor
        if isinstance(pixel_values, dict):
            pixel_values = pixel_values["img"]
        
        # Check if input includes a sequence dimension
        if self.image_sequence_len == 1:
            return self.process_single_image(pixel_values)
        else:
            # Handle sequence of images
            B, T, C, H, W = pixel_values.shape
            assert T == self.image_sequence_len, f"Expected sequence length {self.image_sequence_len}, got {T}"
            
            # Process each image in the sequence separately and concatenate
            sequence_features = []
            for t in range(T):
                single_image = pixel_values[:, t]  # [B, C, H, W]
                features = self.process_single_image(single_image)  # [B, grid_size * grid_size, C]
                sequence_features.append(features)
            
            # Concatenate along the sequence dimension
            all_features = torch.cat(sequence_features, dim=1)  # [B, (grid_size * grid_size * T), C]
            return all_features

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        return 1024  # Projected feature dimension

    @property
    def num_patches(self) -> int:
        return (self.grid_size * self.grid_size) * self.image_sequence_len

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.float16  # ResNet typically uses float16 for mixed precision 
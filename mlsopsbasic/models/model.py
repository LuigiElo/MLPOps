import torch
from torch import nn
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


class SegmentationModel(nn.Module):
    """Wrapper for a pretrained segmentation model."""

    def __init__(self, pretrained=True, num_classes=11):
        """
        Initialize the segmentation model wrapper.

        Args:
            pretrained (bool): Whether to load a pretrained backbone (default: True).
            num_classes (int): Number of output classes for segmentation.
        """
        super().__init__()
        # Load a pretrained DeepLabV3 model
        self.base_model = deeplabv3_resnet50(pretrained=pretrained)
        
        # Replace the classifier head for the custom number of classes
        self.base_model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor (e.g., [batch_size, channels, height, width]).

        Returns:
            dict: Output dictionary containing "out" key with the predicted segmentation map.
        """
        return self.base_model(x)


if __name__ == "__main__":
    # Example: Initialize the model
    num_classes = 11  # Custom number of segmentation classes
    model = SegmentationModel(pretrained=True, num_classes=num_classes)

    # Test with dummy input
    dummy_input = torch.randn(4, 3, 256, 256)  # Batch size 4, RGB image 256x256
    output = model(dummy_input)

    print(f"Output shape: {output['out'].shape}")

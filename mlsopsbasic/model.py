import torch
from torch import nn
from omegaconf import DictConfig


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self, channels, num_classes, out_channels=1) -> None:
        """
        Initialize the model.

        Args:
            channels (list): Number of filters for each convolutional layer.
                             The first value corresponds to the input channels (e.g., 1 for grayscale images).
            num_classes (int): Number of output classes.
            out_channels (int): Number of output channels in the final layer (default is 1 for regression tasks).
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(channels[4], channels[5], kernel_size=3, stride=1, padding=1)

        # Final fully connected layer
        self.fc1 = nn.Linear(channels[5], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv4(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv5(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, 1)  # Flatten while keeping the batch dimension
        x = self.fc1(x)  # Final output
        return x


def initialize_model_from_config(cfg: DictConfig) -> MyAwesomeModel:
    """
    Initialize the model using parameters from a Hydra configuration.

    Args:
        cfg (DictConfig): The Hydra configuration object.

    Returns:
        MyAwesomeModel: The initialized model.
    """
    model = MyAwesomeModel(
        channels=cfg.model.channels,
        num_classes=cfg.model.num_classes,
        out_channels=cfg.model.out_channels
    )
    return model


if __name__ == "__main__":
    # Test the model with dummy configuration
    from omegaconf import OmegaConf

    # Example config
    config = OmegaConf.create({
        "model": {
            "channels": [1, 64, 128, 256, 512, 1024],
            "num_classes": 1,
            "out_channels": 1
        }
    })

    # Initialize the model
    model = initialize_model_from_config(config)
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Test with dummy input
    dummy_input = torch.randn(1, 1, 256, 256)  # Batch size 1, grayscale image 256x256
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")


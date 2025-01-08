import torch
import pytest

#from my_project.models import MyModel
#from mlsopsbasic import MyAwesomeModel
from mlsopsbasic.models.unet import UNet
from mlsopsbasic.config import NUM_CLASSES

@pytest.fixture
def model():
    """Fixture to initialize and return the model."""
    # return MyAwesomeModel()
    #return UNet(channels=[3, 64, 128, 256, 1024], out_channels=11)
    return UNet(channels=[3, 64, 128, 256, 512], out_channels=NUM_CLASSES)

def test_model_output_shape(model):
    """Test that the model produces and output of the expected shape given an input - COCO-style input"""
    input_shape = (1, 3, 512, 512) # (batch_size, channels, height, width)

    #expected_output_shape = (1, 1000) # (batch_size, number of classes)
    expected_output_shape = torch.Size((1, NUM_CLASSES, 512, 512)) # (batch_size, num_classes, height, width)

    dummy_input = torch.randn(*input_shape)  # Create a dummy input tensor
    model_output = model(dummy_input) # Get the model's output

    # Compare shapes, not the tensor itself
    assert model_output.shape == expected_output_shape, (
            f"Expected output shape {expected_output_shape}, but got {model_output.shape}"
    )

    # Additional check to verify number of classes
    assert model_output.shape[1] == NUM_CLASSES, (
        f"Expected {NUM_CLASSES} output channels but got {model_output.shape[1]}"
    )

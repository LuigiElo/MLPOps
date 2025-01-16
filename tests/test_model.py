import torch
import pytest

from mlsopsbasic.models.model import SegmentationModel

NUM_CLASSES = 11

@pytest.fixture
def model():
    """Fixture to initialize and return the model."""
    return SegmentationModel(pretrained=True, num_classes=NUM_CLASSES)

def test_model_initialization():
    """Test that the model initializes with different configurations."""
    model1 = SegmentationModel(pretrained=True, num_classes=NUM_CLASSES)
    assert isinstance(model1, SegmentationModel)

    model2 = SegmentationModel(pretrained=False, num_classes=NUM_CLASSES)
    assert isinstance(model2, SegmentationModel)


def test_model_output_shape(model):
    batch_size = 4
    channels = 3
    height, width = 512, 512
    input_shape = (batch_size, channels, height, width)
    expected_output_shape = (batch_size, NUM_CLASSES, height, width)

    dummy_input = torch.randn(*input_shape)
    model_output = model(dummy_input)

    assert isinstance(model_output, dict), "Model output should be a dictionary"
    assert "out" in model_output, "Model output dictionary should contain 'out' key"

    result_shape = model_output['out'].shape

    assert result_shape == expected_output_shape, f"Expected output shape {expected_output_shape}, but got {result_shape}"

import os
import torch
from models.model import SegmentationModel

# Define the path to the model file
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model/football_segmentation_model.pth'))

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load your trained PyTorch model
model = SegmentationModel()
model.load_state_dict(torch.load(model_path))
model.eval()

# Dummy input for the model
dummy_input = torch.randn(1, 3, 224, 224)

# Define the path to save the ONNX model
onnx_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model/football_segmentation_model.onnx'))

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, onnx_model_path, 
                  input_names=["input"], output_names=["output"], 
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

print(f"Model successfully converted to ONNX format and saved to: {onnx_model_path}")
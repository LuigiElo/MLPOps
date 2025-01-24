# filepath: /home/anpaisgar/MLPOps/mlsopsbasic/predict_model.py
import logging
import onnxruntime as ort
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms
import uvicorn

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to the console
        logging.FileHandler("prediction.log")  # Log to a file named "prediction.log"
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load the ONNX model
ort_session = ort.InferenceSession("../model/football_segmentation_model.onnx")

def preprocess_image(image: Image.Image) -> np.ndarray:
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size for your model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0)
    return image_tensor.numpy()

@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Predict the class of the uploaded image.
    """
    try:
        # Load and preprocess the uploaded image
        image = Image.open(file.file)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_np = preprocess_image(image)

        # Run inference
        ort_inputs = {ort_session.get_inputs()[0].name: image_np}
        ort_outs = ort_session.run(None, ort_inputs)
        predicted_classes = np.argmax(ort_outs[0], axis=1)

        # Convert the predicted classes to a list of lists for easier interpretation
        predicted_classes_list = predicted_classes.squeeze().tolist()
        logger.info(f"Predicted class list: {predicted_classes_list}")

        return {"predicted_classes": predicted_classes_list}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"error": str(e)}

# If this script is executed directly, FastAPI runs with Uvicorn
if __name__ == "__main__":
    # Start FastAPI app with Uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
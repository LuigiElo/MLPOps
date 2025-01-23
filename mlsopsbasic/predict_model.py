from collections.abc import Generator
import datetime
import json
import logging
import torch
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from omegaconf import DictConfig
from hydra import compose, initialize
from PIL import Image
from torchvision import transforms
from mlsopsbasic.models.model import SegmentationModel
import uvicorn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We set up the logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to the console
        logging.FileHandler("prediction.log")  # Log to a file named "prediction.log"
    ]
)
logger = logging.getLogger(__name__)


model = None  # Global variable for the model

def load_config():
    """
    Load Hydra configuration manually.
    """
    with initialize(version_base=None, config_path="./config"):
        cfg = compose(config_name="config")
    return cfg

app = FastAPI()

@app.on_event("startup")
def load_model():
    """
    Load the model during FastAPI's startup event.
    """
    global model
    try:
        # Load configuration using Hydra
        cfg = load_config()
        model_path = cfg.misc.save_path # Path to model in the Hydra configuration
        logger.info(f"Loading model from {model_path}")

        # Load the model
        model = SegmentationModel().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()  # Set the model to evaluation mode
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
    
    # open file for writing prediction results
    with open("predictions.csv", "w") as f:
        f.write("timestamp,filename,predicted_matrix\n")

def add_prediction_to_file(file_name, prediction : list):
    """
    Add the prediction to the CSV file.
    """
    now = str(datetime.datetime.now())
    with open("predictions.csv", "a") as f:
        # Convert the prediction list to a JSON string
        prediction_json = json.dumps(prediction)
        f.write(f"{now},{file_name},{prediction_json}\n")
    


@app.post("/predict/")
async def predict_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Predict the class of the uploaded image.
    """
    try:
        # Load and preprocess the uploaded image
        image = Image.open(file.file)
        if image.mode != "RGB":
            image = image.convert("RGB")

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor()
        ])
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)["out"]
            print(output.shape)
            predicted_class = torch.argmax(output, dim=1)
            print(predicted_class.shape)
            predicted_classes_list = predicted_class.squeeze().cpu().numpy().tolist()
            print(predicted_classes_list)  # Print the list to debug

        background_tasks.add_task(add_prediction_to_file, file_name=file.filename, prediction=predicted_classes_list)
        return {"predicted_class": predicted_classes_list}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"error": str(e)}

# If this script is executed directly, FastAPI runs with Uvicorn
if __name__ == "__main__":
    # Start FastAPI app with Uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

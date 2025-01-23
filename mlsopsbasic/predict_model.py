import logging
import torch
from fastapi import FastAPI, File, UploadFile
from omegaconf import DictConfig
from hydra import compose, initialize
from PIL import Image
from torchvision import transforms
from mlsopsbasic.models.model import SegmentationModel
from prometheus_client import Counter, Histogram, Summary, CollectorRegistry, make_asgi_app
import uvicorn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MY_REGISTRY = CollectorRegistry()

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

# Define Prometheus metrics
error_counter = Counter("prediction_error", "Number of prediction error", registry=MY_REGISTRY)
request_counter = Counter("prediction_requsts", "Number of prediction requests", registry=MY_REGISTRY)
request_latency = Histogram("prediction_latency_seconds", "Prediction latency in seconds", registry=MY_REGISTRY)
# review_summary = Summary("review_length_summary", "Review length summary", registry=MY_REGISTRY)

app = FastAPI()
app.mount("/metrics", make_asgi_app(registry=MY_REGISTRY))

model = None  # Global variable for the model

def load_config():
    """
    Load Hydra configuration manually.
    """
    with initialize(version_base=None, config_path="./config"):
        cfg = compose(config_name="config")
    return cfg

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

@app.get("/")
async def get_root():
    """Root endpoint"""
    return {"message": "Hello from backend"}

@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Predict the class of the uploaded image.
    """
    request_counter.inc()
    with request_latency.time():
        try:
            # review_summary.observe()
            # Load and preprocess the uploaded image
            image = Image.open(file.file)
            if image.mode != "RGB":
                image = image.convert("RGB")

            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),  # Adjust size for your model
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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


            return {"predicted_class": predicted_classes_list}
        except Exception as e:
            error_counter.inc()
            logger.error(f"Error during prediction: {e}")
            return {"error": str(e)}

# If this script is executed directly, FastAPI runs with Uvicorn
if __name__ == "__main__":
    # Start FastAPI app with Uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

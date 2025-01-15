import logging
import torch
from fastapi import FastAPI, File, UploadFile
import hydra
from omegaconf import DictConfig
from contextlib import asynccontextmanager
#from models.model import MyAwesomeModel
from model import SegmentationModel
from train_model import DEVICE
from PIL import Image
from torchvision import transforms

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


def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> torch.Tensor:
    """Run prediction for a given model and dataloader.

    Args:
        model: Model to use for prediction.
        dataloader: DataLoader with batches.

    Returns:
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model.
    """
    logger.info("Starting predictions...")
    predictions = []
    for i, batch in enumerate(dataloader):
        batch = batch.to(DEVICE)
        preds = model(batch)
        predictions.append(preds)
        logger.info(f"Processed batch {i+1}/{len(dataloader)}")
    logger.info("Predictions complete.")
    return torch.cat(predictions, 0)


# @hydra.main(config_path=".", config_name="config")
# def main(cfg: DictConfig) -> None:
#     """Main function to run predictions."""
#     # Obtain model and dataset paths from the Hydra configuration
#     model_path = cfg.logging.model_path  # Path to the trained model
#     images_path = cfg.directories.data_dir  # Path to the dataset

#     logger.info(f"Model: {model_path}")
#     logger.info(f"Images: {images_path}")

#     # Loading model
#     model = MyAwesomeModel().to(DEVICE)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     logger.info("Model loaded successfully.")

#     # Loading images
#     images = torch.load(images_path)
#     logger.info(f"Loaded {len(images)} images for prediction.")

#     # Creating dataloader
#     dataloader = torch.utils.data.DataLoader(
#         images,
#         batch_size=cfg.hyperparameters.batch_size
#     )
#     logger.info(f"DataLoader created with batch size {cfg.hyperparameters.batch_size}.")

#     # Run prediction
#     predictions = predict(model, dataloader)

#     # For each prediction, we print the class with the highest probability
#     for i, pred in enumerate(predictions):
#         logger.info(f"Image {i+1}: Predicted class: {pred.argmax().item()}")
#     logger.info("Prediction process complete.")

#if __name__ == "__main__":
#    main()
# Define FastAPI app as a global variable
app = FastAPI()

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to run predictions."""
    # Access model path from the Hydra configuration
    model_path = cfg.model.model_path  # The path to the trained model from the configuration
    logger.info(f"Model path: {model_path}")

    # Load model in the global scope
    logger.info("Loading model...")
    model = SegmentationModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))  # Load the model using path from config
    model.eval()
    logger.info("Model loaded successfully.")

    @app.post("/predict/")
    async def predict_endpoint(file: UploadFile = File(...)):
        """
        Predict the class of the uploaded image.
        Args:
            file: Uploaded image file.

        Returns:
            JSON response with the predicted class.
        """
        try:
            # Load image
            image = Image.open(file.file)
            if image.mode != "RGB":
                image = image.convert("RGB")
            logger.info("Image loaded successfully.")

            # Preprocess image
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),  # Adjust to your model's input size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

            # Run prediction
            with torch.no_grad():
                output = model(image_tensor)
                predicted_class = torch.argmax(output, dim=1).item()

            logger.info("Prediction complete.")
            return {"predicted_class": predicted_class}
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {"error": str(e)}

# If this script is executed directly, main will run
if __name__ == "__main__":
    main()
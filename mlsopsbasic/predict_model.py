import logging
import torch
import hydra
from omegaconf import DictConfig
from models.model import MyAwesomeModel
from train_model import DEVICE

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


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to run predictions."""
    # Obtain model and dataset paths from the Hydra configuration
    model_path = cfg.logging.model_path  # Path to the trained model
    images_path = cfg.directories.data_dir  # Path to the dataset

    logger.info(f"Model: {model_path}")
    logger.info(f"Images: {images_path}")

    # Loading model
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logger.info("Model loaded successfully.")

    # Loading images
    images = torch.load(images_path)
    logger.info(f"Loaded {len(images)} images for prediction.")

    # Creating dataloader
    dataloader = torch.utils.data.DataLoader(
        images,
        batch_size=cfg.hyperparameters.batch_size
    )
    logger.info(f"DataLoader created with batch size {cfg.hyperparameters.batch_size}.")

    # Run prediction
    predictions = predict(model, dataloader)

    # For each prediction, we print the class with the highest probability
    for i, pred in enumerate(predictions):
        logger.info(f"Image {i+1}: Predicted class: {pred.argmax().item()}")
    logger.info("Prediction process complete.")


if __name__ == "__main__":
    main()

import os
import logging
import matplotlib.pyplot as plt
import torch
import hydra
from omegaconf import DictConfig
from models.model import MyAwesomeModel

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to the console
        logging.FileHandler("training.log")  # Logs to a file
    ]
)
logger = logging.getLogger(__name__)

# load MNIST dataset
def corrupt_mnist(data_path) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test dataloaders for corrupt MNIST."""
    test_images = torch.load(os.path.join(data_path, "test_images.pt"))
    test_target = torch.load(os.path.join(data_path, "test_target.pt"))
    train_images = torch.load(os.path.join(data_path, "train_images.pt"))
    train_target = torch.load(os.path.join(data_path, "train_target.pt"))

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set

@hydra.main(config_path=".", config_name="config")
def train(cfg: DictConfig) -> None:
    """Train a model on MNIST."""
    logger.info("Starting training process...")
    logger.info(f"Configuration: {cfg.pretty()}")

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist(cfg.directories.data_dir)

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.hyperparameters.batch_size,
        num_workers=cfg.hyperparameters.num_workers
    )

    # Loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.hyperparameters.learning_rate,
        weight_decay=cfg.hyperparameters.weight_decay
    )

    # Training loop
    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(cfg.hyperparameters.epochs):
        model.train()
        logger.info(f"Starting epoch {epoch + 1}/{cfg.hyperparameters.epochs}...")
        epoch_loss = 0
        epoch_accuracy = 0
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            # Log training metrics
            batch_loss = loss.item()
            batch_accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_loss"].append(batch_loss)
            statistics["train_accuracy"].append(batch_accuracy)
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy

            if i % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {i}, Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.4f}")

        logger.info(
            f"Epoch {epoch + 1}/{cfg.hyperparameters.epochs} complete! "
            f"Average Loss: {epoch_loss / len(train_dataloader):.4f}, "
            f"Average Accuracy: {epoch_accuracy / len(train_dataloader):.4f}"
        )

    logger.info("Training complete")

    # Save model and training statistics
    model_path = cfg.logging.model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    # Plot and save training statistics
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train Loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train Accuracy")

    log_path = cfg.logging.log_path
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fig.savefig(log_path)
    logger.info(f"Training statistics saved to {log_path}")

@hydra.main(config_path=".", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained model."""
    logger.info("Starting evaluation process...")

    # Load model
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(cfg.logging.model_path))

    # Load dataset
    _, test_set = corrupt_mnist(cfg.directories.data_dir)
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=cfg.hyperparameters.batch_size
    )

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = model(img)
            correct += (y_pred.argmax(dim=1) == target).float().sum().item()
            total += target.size(0)
    logger.info(f"Test Accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    train()

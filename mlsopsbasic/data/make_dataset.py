import click
import kagglehub
import os
import torch
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
import logging
import wandb

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
import torch
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_data(raw_dir: str, processed_dir: str, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Process raw data (including fuse and save variants), split into train/val/test,
    and save it to the processed directory.

    Args:
        raw_dir (str): Path to the raw data directory.
        processed_dir (str): Path to save processed data.
        train_ratio (float): Proportion of training data.
        val_ratio (float): Proportion of validation data.
        test_ratio (float): Proportion of test data.
    """
    # Initialize WandB
    wandb.init(project="image_processing", name="make_data", config={
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio
    })
    wandb.config.update({"status": "started"})

    logger.info("Starting processing of raw data from directory: %s", raw_dir)
    os.makedirs(processed_dir, exist_ok=True)
    logger.info("Created processed directory: %s", processed_dir)
    wandb.log({"processed_directory_created": 1})

    # Define a transform to convert images to tensors and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize (adjust for your dataset)
    ])

    # Group images into triplets: (original, fuse, save)
    data = []
    total_images = 0
    missing_fuse = 0
    missing_save = 0

    logger.info("Scanning files in directory: %s", raw_dir)
    for file_name in sorted(os.listdir(raw_dir)):
        if file_name.endswith(".jpg") and "___fuse" not in file_name and "___save" not in file_name:
            base_name = file_name
            original_path = os.path.join(raw_dir, base_name)
            fuse_path = os.path.join(raw_dir, f"{base_name}___fuse.png")
            save_path = os.path.join(raw_dir, f"{base_name}___save.png")

            try:
                # Load original image
                logger.info("Processing original image: %s", original_path)
                original_image = Image.open(original_path).convert("RGB")
                original_tensor = transform(original_image)

                # Load fuse variant
                if os.path.exists(fuse_path):
                    logger.info("Processing fuse image: %s", fuse_path)
                    fuse_image = Image.open(fuse_path).convert("RGB")
                    fuse_tensor = transform(fuse_image)
                else:
                    missing_fuse += 1
                    logger.warning("Fuse image not found for: %s", original_path)
                    fuse_tensor = None

                # Load save variant
                if os.path.exists(save_path):
                    logger.info("Processing save image: %s", save_path)
                    save_image = Image.open(save_path).convert("RGB")
                    save_tensor = transform(save_image)
                else:
                    missing_save += 1
                    logger.warning("Save image not found for: %s", original_path)
                    save_tensor = None

                # Append triplet to data
                data.append((original_tensor, fuse_tensor, save_tensor))
                total_images += 1
            except Exception as e:
                logger.error("Error processing file %s: %s", file_name, e)
                wandb.log({"error": str(e)})

    # Log processing stats
    logger.info("Total images processed: %d", total_images)
    logger.info("Missing fuse images: %d", missing_fuse)
    logger.info("Missing save images: %d", missing_save)
    wandb.log({
        "total_images_processed": total_images,
        "missing_fuse_images": missing_fuse,
        "missing_save_images": missing_save
    })

    # Split data into train, val, and test sets
    logger.info("Splitting data into train, validation, and test sets.")
    train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio), random_state=42)
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(temp_data, test_size=(1 - val_ratio_adjusted), random_state=42)

    logger.info("Data split complete:")
    logger.info("  Train set size: %d", len(train_data))
    logger.info("  Validation set size: %d", len(val_data))
    logger.info("  Test set size: %d", len(test_data))
    wandb.log({
        "train_set_size": len(train_data),
        "validation_set_size": len(val_data),
        "test_set_size": len(test_data)
    })

    # Save the datasets
    logger.info("Saving datasets to processed directory.")
    train_path = os.path.join(processed_dir, "train_data.pt")
    val_path = os.path.join(processed_dir, "val_data.pt")
    test_path = os.path.join(processed_dir, "test_data.pt")

    torch.save(train_data, train_path)
    logger.info("Train data saved to: %s", train_path)
    wandb.log({"train_data_saved": train_path})

    torch.save(val_data, val_path)
    logger.info("Validation data saved to: %s", val_path)
    wandb.log({"val_data_saved": val_path})

    torch.save(test_data, test_path)
    logger.info("Test data saved to: %s", test_path)
    wandb.log({"test_data_saved": test_path})

    # Mark the process as completed
    wandb.config.update({"status": "completed"})
    wandb.finish()
    logger.info("Processing complete. Data saved to: %s", processed_dir)

if __name__ == "__main__":
    make_data("C:/Users/luisf/mlsopsbasic/data/raw", "C:/Users/luisf/mlsopsbasic/data/processed")

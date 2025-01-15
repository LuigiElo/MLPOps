import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np

# Hydra imports
import hydra
from omegaconf import DictConfig

# Progress bar

# Weights & Biases
import wandb

# cProfile for profiling
import cProfile
import pstats

from mlsopsbasic.model import SegmentationModel

# Suppress warnings for beta transforms
torchvision.disable_beta_transforms_warning()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################
# Datasets
##############################################
class FootballSegmentationDataset(torch.utils.data.Dataset):
    """
    Dataset for football segmentation task using triplets (original, fuse, save).

    Args:
        root_dir (str): Path to the dataset directory (train/val/test folder).
        use_mask_type (str): Use either 'fuse' or 'save' as the segmentation mask.
        transform (callable, optional): Transformations to apply to the input images.
        target_transform (callable, optional): Transformations to apply to the segmentation masks.
    """
    def __init__(self, root_dir, use_mask_type="fuse", transform=None, target_transform=None):
        self.root_dir = root_dir
        self.use_mask_type = use_mask_type
        self.transform = transform
        self.target_transform = target_transform

        # Build the dataset by finding all base image files
        self.base_files = []
        for file_name in sorted(os.listdir(root_dir)):
            if file_name.endswith(".jpg") and "___fuse" not in file_name and "___save" not in file_name:
                self.base_files.append(file_name)

    def __len__(self):
        return len(self.base_files)

    def __getitem__(self, idx):
        base_file = self.base_files[idx]
        base_name = os.path.splitext(base_file)[0]

        # Paths for original image and mask
        original_path = os.path.join(self.root_dir, base_file)
        mask_path = os.path.join(self.root_dir, f"{base_name}.jpg___{self.use_mask_type}.png")

        # Load original image
        original_image = Image.open(original_path).convert("RGB")

        # Load segmentation mask
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        # Apply transformations
        if self.transform:
            original_image = self.transform(original_image)
            mask = self.transform(mask)
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            # Default conversion to tensor with class indices
            mask = torch.tensor(np.array(mask, dtype=np.int64))

        return original_image, mask


##############################################
# Helper Functions
##############################################
def mask_transform(mask):
    """Convert PIL mask to torch tensor of class indices."""
    mask = np.asarray(mask, dtype=np.int64)
    return torch.tensor(mask)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """One epoch of training."""
    model.train()
    total_loss = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        masks = masks.squeeze(1)  # remove channel dimension if needed

        optimizer.zero_grad()
        outputs = model(images)["out"]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def compute_metrics(predictions, targets, num_classes):
    """Computes mIoU, Pixel Accuracy, and Dice Coefficient."""
    predictions = torch.argmax(predictions, dim=1)  # Get predicted class per pixel
    intersection = torch.zeros(num_classes, device=predictions.device)
    union = torch.zeros(num_classes, device=predictions.device)
    dice = torch.zeros(num_classes, device=predictions.device)
    correct_pixels = (predictions == targets).sum().item()
    total_pixels = targets.numel()

    for cls in range(num_classes):
        pred_cls = (predictions == cls)
        true_cls = (targets == cls)

        intersection[cls] = (pred_cls & true_cls).sum().item()
        union[cls] = (pred_cls | true_cls).sum().item()
        # dice coefficient for each class
        dice[cls] = (2 * intersection[cls]) / (2 * intersection[cls] + union[cls] - intersection[cls] + 1e-6)

    miou = (intersection / (union + 1e-6)).mean().item()  # Avoid division by zero
    pixel_accuracy = correct_pixels / total_pixels
    dice_score = dice.mean().item()

    return miou, pixel_accuracy, dice_score


def evaluate(model, loader, criterion, device, num_classes):
    """Evaluate model and compute loss and metrics."""
    model.eval()
    total_loss = 0
    total_miou, total_pa, total_dice = 0, 0, 0
    count = 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.squeeze(1)  # remove channel dimension
            outputs = model(images)["out"]

            loss = criterion(outputs, masks)
            total_loss += loss.item()

            # Compute Metrics
            miou, pa, dice = compute_metrics(outputs, masks, num_classes)
            total_miou += miou
            total_pa += pa
            total_dice += dice
            count += 1

    avg_loss = total_loss / len(loader)
    avg_miou = total_miou / count
    avg_pa = total_pa / count
    avg_dice = total_dice / count

    return avg_loss, avg_miou, avg_pa, avg_dice


##############################################
# Main with Hydra and Profiling
##############################################
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    Main training function, parameterized by Hydra config.
    We optionally enable cProfile if cfg.profiling.enable is True.
    """
    # Check if profiling is enabled
    if cfg.profiling.enable:
        profiler = cProfile.Profile()
        profiler.enable()

    # 1) Define transformations
    image_transform = transforms.Compose([
        transforms.Resize((cfg.training.image_size, cfg.training.image_size)),
        transforms.ToTensor(),
    ])

    # 2) Create datasets and data loaders
    train_dataset = FootballSegmentationDataset(
        root_dir=cfg.data.train_dir,
        use_mask_type=cfg.data.use_mask_type,
        transform=image_transform,
        target_transform=mask_transform
    )
    val_dataset = FootballSegmentationDataset(
        root_dir=cfg.data.val_dir,
        use_mask_type=cfg.data.use_mask_type,
        transform=image_transform,
        target_transform=mask_transform
    )
    test_dataset = FootballSegmentationDataset(
        root_dir=cfg.data.test_dir,
        use_mask_type=cfg.data.use_mask_type,
        transform=image_transform,
        target_transform=mask_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )

    # 3) Initialize Weights & Biases
    wandb.init(project=cfg.wandb.project_name)

    # 4) Load or create your model
    model = SegmentationModel(num_classes=cfg.model.num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 5) Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )

    # 6) Main training loop
    for epoch in range(cfg.training.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_miou, val_pa, val_dice = evaluate(model, val_loader, criterion, device, cfg.model.num_classes)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mIoU": val_miou,
            "val_Pixel_Accuracy": val_pa,
            "val_Dice_Score": val_dice
        })

        print(
            f"Epoch {epoch+1}/{cfg.training.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val mIoU: {val_miou:.4f} | Val Pixel Acc: {val_pa:.4f} | Val Dice: {val_dice:.4f}"
        )

    # 7) Save the model
    save_path = cfg.misc.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    wandb.save(save_path)

    # 8) Evaluate on test set
    test_loss, test_miou, test_pa, test_dice = evaluate(
        model, test_loader, criterion, device, cfg.model.num_classes
    )
    print(
        f"Test Metrics | Loss: {test_loss:.4f} | mIoU: {test_miou:.4f} | "
        f"Pixel Accuracy: {test_pa:.4f} | Dice Score: {test_dice:.4f}"
    )

    # If profiling was enabled, print or save results
    if cfg.profiling.enable:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats(cfg.profiling.sort_key)
        # Print top 30 lines by default
        stats.print_stats(30)
        # Optionally, you can dump stats to a file for further analysis:
        # stats.dump_stats("profile_results.pstat")


if __name__ == "__main__":
    main()

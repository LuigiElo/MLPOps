import glob
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
import torchvision
from tqdm.auto import tqdm
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet50
import wandb

from config import (BATCH_SIZE, CHANNELS, DEVICE, LEARNING_RATE, LOG_PATH, MODEL_PATH, NUM_CLASSES, OUT_CHANNELS, WEIGHT_DECAY)
from models.unet import UNet
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import torch

# Suppress warnings
torchvision.disable_beta_transforms_warning()

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class FootballSegmentationDataset(Dataset):
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
        print(mask_path)

        # Load original image
        original_image = Image.open(original_path).convert("RGB")

        # Load segmentation mask
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")  # Convert to grayscale (1 channel)
        else:
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        # Apply transformations
        if self.transform:
            original_image = self.transform(original_image)
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = torch.tensor(np.array(mask, dtype=np.int64))  # Convert to tensor with class indices

        return original_image, mask

# Mask transformations (optional)
def mask_transform(mask):
    mask = np.array(mask, dtype=np.int64)  # Convert to NumPy array
    return torch.tensor(mask)  # Convert to tensor with class indices

# Training Loop
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)["out"]  # Get predictions
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(loader)

def adapt_model(model, num_classes):
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model

def main():
    # Input image transformations
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = FootballSegmentationDataset(
        root_dir="data/processed/train",
        use_mask_type="fuse",  # Use 'fuse' masks (or 'save' as needed)
        transform=image_transform,
        target_transform=mask_transform
    )

    val_dataset = FootballSegmentationDataset(
        root_dir="data/processed/val",
        use_mask_type="fuse",
        transform=image_transform,
        target_transform=mask_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # Initialize WandB
    wandb.init(project="football_segmentation")

    # Load Pretrained Model
    model = deeplabv3_resnet50(pretrained=True)
    model = adapt_model(model, num_classes=11)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Main Training Loop
    for epoch in range(10):  # Change epochs as needed
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the Model
    torch.save(model.state_dict(), "football_segmentation_model.pth")
    wandb.save("football_segmentation_model.pth")

if __name__ == "__main__":
    main()
import glob
from typing import Dict, List, Tuple
import cv2
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
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

        original_image = cv2.cvtColor(cv2.imread(original_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Apply transformations
        if self.transform:
            data = self.transform(image=original_image,mask=mask)
            original_image = data["image"]
            mask = data["mask"]
        
        original_image=torch.Tensor(np.transpose(original_image,(2,0,1))) / 255.0 
        mask=torch.Tensor(np.transpose(mask,(2,0,1))) / 255.0

        return original_image, mask

# Mask transformations (optional)
def mask_transform(mask):
    mask = np.asarray(mask, dtype=np.int64) # Convert to numpy array
    return torch.tensor(mask)  # Convert to tensor with class indices

# Training Loop
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        masks = masks.squeeze(1)  # Remove channel dimension

        optimizer.zero_grad()
        outputs = model(images)["out"]  # Get predictions
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

def adapt_model(model, num_classes):
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model

def compute_metrics(predictions, targets, num_classes):
    """Computes per-class and overall mIoU, Pixel Accuracy, and Dice Coefficient."""
    # predictions shape: (N, H, W)
    # targets shape: (N, H, W)
    predictions = torch.argmax(predictions, dim=1)

    intersection = torch.zeros(num_classes, device=predictions.device, dtype=torch.float)
    union = torch.zeros(num_classes, device=predictions.device, dtype=torch.float)
    dice_per_class = torch.zeros(num_classes, device=predictions.device, dtype=torch.float)

    correct_pixels = (predictions == targets).sum().item()
    total_pixels = targets.numel()

    for cls in range(num_classes):
        pred_cls = (predictions == cls)
        true_cls = (targets == cls)

        inter = (pred_cls & true_cls).sum().float()
        union_ = (pred_cls | true_cls).sum().float()
        intersection[cls] = inter
        union[cls] = union_

        # Dice for class 'cls'
        # dice_per_class[cls] = 2 * inter / (pred_cls.sum() + true_cls.sum() + 1e-6)
        # Alternatively, using the formula from your code:
        dice_per_class[cls] = (2.0 * inter) / (2.0 * inter + union_ - inter + 1e-6)

    # Per-class IoU
    iou_per_class = intersection / (union + 1e-6)

    # Overall means
    miou = iou_per_class.mean().item()
    pixel_accuracy = correct_pixels / total_pixels
    dice_score = dice_per_class.mean().item()

    return iou_per_class, dice_per_class, miou, pixel_accuracy, dice_score

class DiceLoss(nn.Module):
    """
    Implementation of Dice Loss for multi-class segmentation.
    
    inputs:  (N, C, H, W) -> raw logits
    targets: (N, H, W)    -> class indices [0..C-1]
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Step 1: Apply softmax to get class probabilities
        inputs = F.softmax(inputs, dim=1)  # shape: (N, C, H, W)
        
        # Step 2: Convert target from (N, H, W) to one-hot: (N, C, H, W)
        # NOTE: F.one_hot returns (N, H, W, C), so we permute to (N, C, H, W).
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # (N, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (N, C, H, W)

        # Step 3: Calculate intersection and union for each class
        # intersection = sum of elementwise multiplication over spatial dims
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        # sums for each
        inputs_sum = inputs.sum(dim=(2, 3))
        targets_sum = targets_one_hot.sum(dim=(2, 3))
        
        # Dice Coefficient for each sample & each class
        dice_per_class = (2.0 * intersection + self.smooth) / (
            inputs_sum + targets_sum + self.smooth
        )  # shape: (N, C)

        # Average over classes, then average over batch
        dice_score = dice_per_class.mean()

        # Dice loss is 1 - mean dice coefficient
        dice_loss = 1.0 - dice_score
        return dice_loss



def evaluate(model, loader, criterion, device, num_classes):
    """Evaluate model and compute loss and metrics (including per-class)."""
    model.eval()
    total_loss = 0
    total_miou, total_pa, total_dice = 0.0, 0.0, 0.0
    count = 0

    # For accumulating per-class IoU/Dice across the dataset
    iou_sums = torch.zeros(num_classes, device=device, dtype=torch.float)
    dice_sums = torch.zeros(num_classes, device=device, dtype=torch.float)

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.squeeze(1)  # Remove channel dimension
            outputs = model(images)["out"]

            loss = criterion(outputs, masks)
            total_loss += loss.item()

            # Compute Metrics
            iou_per_class, dice_per_class, miou, pa, dice = compute_metrics(outputs, masks, num_classes)
            iou_sums += iou_per_class
            dice_sums += dice_per_class

            total_miou += miou
            total_pa += pa
            total_dice += dice
            count += 1

    avg_loss = total_loss / len(loader)
    avg_miou = total_miou / count
    avg_pa = total_pa / count
    avg_dice = total_dice / count

    # Average per-class IoU/Dice over the dataset
    iou_per_class_avg = iou_sums / count
    dice_per_class_avg = dice_sums / count

    return (
        avg_loss,
        avg_miou,
        avg_pa,
        avg_dice,
        iou_per_class_avg,
        dice_per_class_avg
    )


def main():
    mean, std, im_h, im_w = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 256, 256
    # Input image transformations
    image_transform = A.Compose([
        A.Resize(im_h, im_w),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    train_dataset = FootballSegmentationDataset(
        root_dir="data/processed/train",
        use_mask_type="fuse",
        transform=image_transform,
        target_transform=mask_transform
    )

    val_dataset = FootballSegmentationDataset(
        root_dir="data/processed/val",
        use_mask_type="fuse",
        transform=image_transform,
        target_transform=mask_transform
    )

    num_workers = os.cpu_count() - 1
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=num_workers)

    # Initialize WandB
    wandb.init(project="football_segmentation")

    # Suppose we have 11 classes and want to set custom weights for each class.
    # Example: background has weight=0.2, a rare class might have weight=2.0, etc.
    class_weights = torch.tensor([
        0.1,  # class 0
        3.0,  # class 1
        3.5,  # class 2
        3.0,  # class 3
        3.0,  # class 4
        3.0,  # class 5
        3.0,  # class 6
        3.0,  # class 7
        3.0,  # class 8
        3.0,  # class 9
        3.0
    ])

    # Move weights to the same device as your model/tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = class_weights.to(device)

    # CWE
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Initialize model
    model = deeplabv3_resnet50(pretrained=True)
    model = adapt_model(model, num_classes=11).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # Training loop
    for epoch in range(60):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        (val_loss, val_miou, val_pa, val_dice, val_iou_per_class, val_dice_per_class
        ) = evaluate(model, val_loader, criterion, device, num_classes=11)

        # Prepare per-class logging
        val_iou_dict = {f"val_IoU_class_{i}": val_iou_per_class[i].item() for i in range(11)}
        val_dice_dict = {f"val_Dice_class_{i}": val_dice_per_class[i].item() for i in range(11)}

        # Log to WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mIoU": val_miou,
            "val_Pixel_Accuracy": val_pa,
            "val_Dice_Score": val_dice,
            **val_iou_dict,
            **val_dice_dict
        })

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val Metrics - mIoU: {val_miou:.4f}, Pixel Accuracy: {val_pa:.4f}, Dice Score: {val_dice:.4f}")

    # Save the Model
    torch.save(model.state_dict(), "models/football_segmentation_model.pth")
    wandb.save("football_segmentation_model.pth")

    # Test the model
    test_dataset = FootballSegmentationDataset(
        root_dir="data/processed/test",
        use_mask_type="fuse",
        transform=image_transform,
        target_transform=mask_transform
    )

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=num_workers)
    (
        test_loss,
        test_miou,
        test_pa,
        test_dice,
        test_iou_per_class,
        test_dice_per_class
    ) = evaluate(model, test_loader, criterion, device, num_classes=11)

    print(f"Test Metrics - mIoU: {test_miou:.4f}, Pixel Accuracy: {test_pa:.4f}, Dice Score: {test_dice:.4f}")
    print("Per-class IoU:", test_iou_per_class)
    print("Per-class Dice:", test_dice_per_class)

if __name__ == "__main__":
    main()

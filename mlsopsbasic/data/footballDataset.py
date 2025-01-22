import torch
import os
import numpy as np
from PIL import Image

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

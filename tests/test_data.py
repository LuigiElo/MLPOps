import pytest
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from mlsopsbasic.config import DATA_DIR

class FootballDataset:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        print(f"\nInitializing dataset with root_dir: {self.root_dir}")
        print(f"Does directory exist? {self.root_dir.exists()}")
        print(f"Is it a directory? {self.root_dir.is_dir()}")


        self.image_files = [f for f in self.root_dir.glob("Frame*1*(*).jpg")
                            if not str(f).endswith(("___fuse.png", "___save.png"))]
        # image_files = glob.glob(os.path.join(self.root_dir, "Frame*(*).jpg"))
        
        # Load class mappings / these are the 11 classes from the dataset
        self.classes = {
            0: "Goal Bar",
            1: "Referee",
            2: "Advertisement",
            3: "Ground",
            4: "Ball",
            5: "Coaches & Officials",
            6: "Audience",
            7: "Goalkeeper A",
            8: "Goalkeeper B",
            9: "Team A",
            10: "Team B",
        }

        print(f"Found {len(self.image_files)} images in {self.root_dir}")
        print(f"Data dir: {DATA_DIR}")

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        img_path =   self.image_files[index]
        fuse_mask_path = str(img_path) + "___fuse.png"

        # Load image and mask
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(fuse_mask_path), cv2.IMREAD_GRAYSCALE)

        return image, mask
    

@pytest.fixture
def dataset():
    return FootballDataset(DATA_DIR)

def test_dataset_not_empty(dataset):
    expected_length = len(dataset)
    assert expected_length > 0, "Dataset should not be empty"

def test_dataset_length(dataset):
    """Test if dataset contains the expected number of images"""
    expected_length = 100 # I think this is the right length of our dataset, but double-check
    assert len(dataset) == expected_length, f"Dataset should contain {expected_length} images"

def test_image_dimensions(dataset):
    """Test if images have the correct dimensions and channels"""
    image, _ = dataset[0]

    # Test image dimensions
    assert len(image.shape) == 3, "Image should have 3 dimensions (H, W, C)"
    assert image.shape[2] == 3, "Image should have 3 channels (RGB)"

    # Maybe add specific size assertions if images are consistently sized
    # assert image.shape[:2] == (HEIGHT, WIDTH), f"Image should be {HEIGHT}x{WIDTH}"

def test_mask_dimensions(dataset):
    """Test if masks have the correct dimensions"""
    _, mask = dataset[0]

    # Test mask dimensions
    assert len(mask.shape) == 2, "Mask should have 2 dimensions (H, W)"
    assert mask.dtype == np.uint8, "Mask should be of type uint8"

"""def test_mask_values(dataset):
    #Test if mask contains valid class labels
    #_, mask = dataset[0]

    # Get unique values in mask
    #unique_values = np.unique(mask)

    for i in range(len(dataset)):
        _, mask = dataset[i]
    unique_values = np.unique(mask)
    print(f"Mask {i} unique values: {unique_values}")


    # Check if all values are within 0 and 10
    assert all(val >= 0 and val <= 10 for val in unique_values), "Mask values should be between 0 and 10"
"""

"""def test_all_classes_represented(dataset):
    # Test if all classes appear in the dataset
    all_classes = set(range(11)) # 11 classes
    found_classes = set()

    for i in range(len(dataset)):
        _, mask = dataset[i]
        found_classes.update(np.unique(mask))

    missing_classes = all_classes - found_classes
    assert len(missing_classes) == 0, f"Missing classes: {missing_classes}"
"""
def test_image_mask_pairs(dataset):
    """Test if all images have corresponding mask files"""
    for i in range(len(dataset)):
        img_path = dataset.image_files[i]
        fuse_mask_path = Path(str(img_path) + "___fuse.png")

        assert fuse_mask_path.exists(), f"Missing mask file for file: {img_path}"

def test_image_loading(dataset):
    """Test if images can be loaded and are not corrupted"""
    for i in range(len(dataset)):
        try:
            image, mask = dataset[i]
            assert image is not None, f"Failed to load image at index {i}"
            assert mask is not None, f"Failed to load mask at index {i}"
            assert not np.isnan(image).any(), f"Image contains NaN values at index {i}"
            assert not np.isnan(mask).any(), f"Mask contains NaN values at index {i}"
        except Exception as e:
            pytest.fail(f"Failed to load/mask pair at index {i}: {str(e)}")

def test_image_channels_range(dataset):
    """Test if image values are in the expected range(0-255 for uint8)"""
    image, _ = dataset[0]
    assert image.dtype == np.uint8, "Image should be uint8"
    assert np.min(image) >= 0 and np.max(image) <= 255, "Image values should be between 0 and 255"

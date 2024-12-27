import os
import torch
import datetime

# Directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data/archive/images/")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
LOG_DIR = os.path.join(ROOT_DIR, "logs") # Maybe we should add a logs folder

# Device
DEVICE = (
    "cuda"if torch.cuda.is_available() else "mps"
    if torch.backends.mps.is_available() else "cpu"
)

# Hyperparameters
BATCH_SIZE = 8

LEARNING_RATE = 1e-3                    
EPOCHS = 10
NUM_WORKERS = os.cpu_count() - 1
WEIGHT_DECAY = 1e-4 # L2 regularization - This small value allows the model to fit the training data more closely, a higher value would increase regularization

# Image size
IMAGE_SIZE = 256

# Number of classes
NUM_CLASSES = 1 # 11?

# Model parameters
CHANNELS = [1, 64, 128, 256, 512, 1024]
OUT_CHANNELS = 1

# Model name
now = datetime.datetime.now()
MODEL_NAME = f"unet_{now.strftime('%Y-%m-%d-%H-%M-%S')}"

# Model path
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pt")

# Log path
LOG_PATH = os.path.join(LOG_DIR, f"{MODEL_NAME}.log")
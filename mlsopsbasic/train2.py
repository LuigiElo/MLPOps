from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
import torchvision
from tqdm.auto import tqdm

from config import (BATCH_SIZE, CHANNELS, DEVICE, EPOCHS, LEARNING_RATE, LOG_PATH, MODEL_PATH, NUM_CLASSES, NUM_WORKERS, OUT_CHANNELS, WEIGHT_DECAY)
from models.unet import UNet

torchvision.disable_beta_transforms_warning()

model = UNet(in_channels=CHANNELS, out_channels=OUT_CHANNELS)
model.to(DEVICE)

def unet_loss(outputs, targets, alpha=0.5, beta=1.5):
    """"
        U-Net loss function with per-pixel weights to balance the classes and an extra to penalize joining two bits of the segmentation
    """

    weights = alpha * targets = beta * (1 - targets)
    loss = F.binary_cross_entropy_with_logits(outputs, targets, weights, reduction="none")
    intersection = torch.sum(outputs * targets * weights)
    union = torch.sum(outputs * weights) = torch.sum(targets * weights)
    loss += 1 - 2 * (intersection + 1) / (union + 1)
    return torch.mean(loss)


def accuracy(outputs, targets):
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > 0.5).float()
    return torch.mean((outputs == targets).float())


"""
    - Args:
        - model: torch.nn.Module: The model to train.
        - train_loader: torch.utils.data.DataLoader: The training data loader.
        - optimizer: torch.optim.Optimizer: The optimizer to use for training.
        - loss_fn: torch.nn.Module: The loss function to use for training.

    - Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example: (0.1112, 0.8743)
"""
def train_step(model, dataloader, loss_fn, optimizer):
    # Trains the model for one epoch.
    model.train()

    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric accross all batches
        train_acc += accuracy(y_pred, y).item()
    
    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc

""""
    Turns a target Pytorch model to "eval" mode and then performs a forward pass on a testing dataset
    - Args:
        - model: torch.nn.Module: The model to evaluate.
        - dataloader: torch.utils.data.DataLoader: The evaluation data loader.
        - loss_fn: torch.nn.Module: The loss function to use for evaluation.
    
    - Returns:
        - Tuple[float, float]
            A tuple of testing loss and testing accuracy metrics.
            In the form (val_loss, val_accuracy). For example: (0.0223, 0.8985)
"""
def val_step(model, dataloader, loss_fn):
    model.eval()

    val_loss, val_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_model():
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(DEVICE), y.to(DEVICE)

            # 1. Forward pass
            val_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()

            # Calculate and accumulate accuracy
            val_acc += accuracy(val_pred_logits, y).item()
        
        # Adjust metrics to get average loss and accuracy per batch
        val_loss = val_loss / len(dataloader)
        val_acc = val_acc / len(dataloader)

        return val_loss, val_acc
    

"""
    Trains and tests a Pytorch model.
    Passes a target Pytorch model through train_step() and val_step() functions for a number of epochs, training and testing the model in the same epoch loop.
    Calculates, prints and stores evaluation metrics throughtout.
    - Args:
        - model: torch.nn.Module: The model to train.
        - train_dataloader: torch.utils.data.DataLoader: The training data loader.
        - val_dataloader: torch.utils.data.DataLoader: The testing data loader.
        - optimizer: torch.optim.Optimizer: The optimizer to use for training.
        - loss_fn: torch.nn.Module: The loss function to use for training.
        - epochs: int: The number of epochs to train the model for.

    - Returns:
        - Dict[str, List]
            A dictionary of training and validation metrics.
            In the form {"train_loss": [0.1112, 0.1093, ...], "train_accuracy": [0.8743, 0.8874, ...], "val_loss": [0.1112, 0.1093, ...], "val_accuracy": [0.8743, 0.8874, ...]}
            For example: {"train_loss": [0.1112, 0.1093, ...], "train_accuracy": [0.8743, 0.8874, ...], "val_loss": [0.1112, 0.1093, ...], "val_accuracy": [0.8743, 0.8874, ...]}
"""
def train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs):
    results = {
        "train_loss": [], 
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # Loop through training and testing steps for a number of epochs
    for epoch in tdqm(range(epochs), desc="Epochs"):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer
        )
        val_loss, val_acc = val_step(
            model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn
        )

        # Prints
        print(
            f"Epochs: {epoch+1} |" 
            f"train_loss: {train_loss:.4f} |" 
            f"train_acc: {train_acc:.4f} |"
            f"val_loss: {val_loss: .4f} |" 
            f"val_acc: {val_acc: .4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    # Return the filled results at the end of the epochs
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")
    
    return results


if __name__ == "__main__":
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS
    )

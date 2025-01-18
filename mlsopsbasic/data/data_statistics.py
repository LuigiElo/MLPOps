import matplotlib.pyplot as plt
import torch
import typer
from footballDataset import FootballSegmentationDataset

def dataset_statistics(datadir: str = "data") -> None:
    """Compute dataset statistics."""
    train_dataset = FootballSegmentationDataset(
        root_dir="data/processed/train"
    )
    val_dataset = FootballSegmentationDataset(
        root_dir="data/processed/val"
    )
    test_dataset = FootballSegmentationDataset(
        root_dir="data/processed/test"
    )
    print(f"Train dataset: FootballSegmentation")
    print(f"Number of images: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].size}")
    print("\n")
    print(f"Test dataset: FootballSegmentation")
    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0][0].size}")
    print("\n")
    print(f"Validation dataset: FootballSegmentation")
    print(f"Number of images: {len(val_dataset)}")
    print(f"Image shape: {val_dataset[0][0].size}")


if __name__ == "__main__":
    typer.run(dataset_statistics)
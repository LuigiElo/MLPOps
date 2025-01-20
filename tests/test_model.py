import torch
import wandb
import os
import time

from mlsopsbasic.models.model import SegmentationModel


def load_model(artifact):
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )

    logdir = "./models"
    artifact = api.artifact(artifact)
    artifact.download(root=logdir)
    file_name = artifact.files()[0].name

    model = model = SegmentationModel().to(torch.device("cpu"))
    state_dict = torch.load(f"{logdir}/{file_name}")
    model.load_state_dict(state_dict)
    model.eval()

    return model

def test_model_speed():
    model = load_model(os.getenv("MODEL_NAME"))
    start = time.time()
    for _ in range(1):
        model(torch.randn(4, 3, 256, 256))
    end = time.time()
    assert end - start < 1 # 1 second

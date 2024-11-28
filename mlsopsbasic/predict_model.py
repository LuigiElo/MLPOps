import sys
from matplotlib import pyplot as plt
import torch

from models.model import MyAwesomeModel
from train_model import DEVICE

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.
    . Recommended interface is that users can give this file either a folder with raw images that gets loaded in or a numpy or pickle file with already loaded images e.g. something like this
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """   
    return torch.cat([model(batch) for batch in dataloader], 0)

if __name__ == "__main__":
    # retrieve model and dataloader from the arguments, the first one being model and the second one being dataloader, example:
    # python <project_name>/models/predict_model.py \
    # models/my_trained_model.pt \  # file containing a pretrained model
    # data/example_images.npy  # file containing just 10 images for prediction
    model_arg = sys.argv[1]
    images_arg = sys.argv[2]

    print(f"Model: {model_arg}")
    print(f"Dataloader: {images_arg}")

    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_arg))

    # load images
    images = torch.load(images_arg)

    # create dataloader
    dataloader = torch.utils.data.DataLoader(images, batch_size=32)

    # run prediction
    predictions = predict(model, dataloader)

    #for each prediction, print the class with the highest probability and show image
    for pred in predictions:
        print(f"Predicted class: {pred.argmax()}")
    print("Prediction complete")
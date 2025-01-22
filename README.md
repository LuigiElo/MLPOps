# Segmentation of Football match images.

### A. Overall goal of the project.
The goal of the project is to create an automated system to segment pre-selected images from a real football match.

### B. What framework are you going to use and do you intend to include the framework in the project?
We are using the Pytorch framework, as it was strongly suggested in the course.

### C. What data are you going to run on (initially, may change)
We are getting the data from the following Kagle dataset: https://www.kaggle.com/datasets/sadhliroomyprime/football-semantic-segmentation/data

From the dataset description -> The dataset was collected from the UEFA Super Cup match between Real Madrid and Manchester United in 2017 (Highlights). It has 11 standard classes and it uses SuperAnnotate’s pixel editor to label and classify the images following instance segmentation principles. Export was made in COCO with fused labels to optimise interoperability and visual understanding.

### D. What models do you expect to use
For the model, we are using a DeepLabV3 model with a ResNet-50 backbone (https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.deeplabv3_resnet50.html)

We are using this model since members of the group have experience working with it before. Furthermore, this is a pre-trained model with optimized feature-extraction capabilities (great for segmentation) and pre-trained weights out of the box.

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── mlsopsbasic  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

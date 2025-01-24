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
.
├── LICENSE                         <- Open-source license if one is chosen
├── Makefile                        <- Makefile with convenience commands like `make data` or `make train`
├── README.md                       <- The top-level README for developers using this project.
├── cloudbuild.yaml                 <- Configuration for Google Cloud Build
├── cloudbuild_api.yaml             <- API-specific Cloud Build configuration
├── data                            <- Directory for storing datasets
│   ├── processed                   <- The final, canonical data sets for modeling.
│   └── raw                         <- The original, immutable data dump.
├── data.dvc                        <- DVC file for data version control
├── docker-compose.yaml             <- Docker Compose configuration
├── dockerfiles                     <- Directory for Dockerfiles
│   ├── api.dockerfile              <- Dockerfile for the API
│   ├── predict_model.dockerfile    <- Dockerfile for prediction model
│   └── train_model.dockerfile      <- Dockerfile for training the model
├── docs                            <- Documentation resources
│   ├── README.md                   <- Documentation overview
│   ├── mkdocs.yaml                 <- MkDocs configuration file
│   └── source                      <- Source files for documentation
├── frontend.py                     <- Frontend code for the application
├── metrics_testing.py              <- Script for testing model metrics
├── mlsopsbasic                     <- Sorce code for use in this project
│   ├── __init__.py                
│   ├── __pycache__                
│   ├── config                      <- Configuration files
│   ├── convert_to_onx.py           <- Script to convert models to ONNX format
│   ├── data                        <- Data workflow
│   ├── models                      <- Vertex AI config and model
│   ├── predict_model.py            <- Script for model prediction
│   ├── predict_model_onnx.py       <- Script for ONNX model prediction
│   ├── train_model.py              <- Script for training the model
│   └── visualizations              <- Scripts for data visualization
├── model                           <- DVC weights
│   ├── football_segmentation_model.onnx  <- ONNX model for football segmentation
│   ├── football_segmentation_model.pth   <- PyTorch model for football segmentation
│   └── model.pth                   <- General PyTorch model
├── models                          <- Pretrained weights
│   └── model.pth                 
├── outputs                         <- Output results from Hydra 
│   ├── 2025-01-08                  <- Outputs from January 8, 2025
│   └── 2025-01-14                  <- Outputs from January 14, 2025
├── prediction.log                  <- Log file for predictions
├── pyproject.toml                  <- Python project configuration
├── reports                         <- Reports and figures
│   ├── README.md                   <- Markdowns and project questions
│   ├── figures                     <- Generated figures for reports
│   └── report.py                   <- Script to generate reports
├── requirements.txt                <- General project dependencies
├── requirements_backend.txt        <- Backend-specific dependencies
├── requirements_dev.txt            <- Development dependencies
├── requirements_tests.txt          <- Testing dependencies
├── tests                           <- Test suite for the project
│   ├── __init__.py                
│   ├── __pycache__                
│   ├── integrationtests            <- API testing
│   ├── performancetests            <- Load testing
│   ├── test_data.py                <- Tests for data coverage
│   ├── test_model.py               <- Tests for model functionality
│   ├── test_model_performance.py   <- Tests for model performance
│   ├── test_training.py            <- Tests for training functionality
├── training_statistics.png         <- Visualization of training statistics
└── utils                           <- Utility scripts
    └── link_model.py               <- Script to link models
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

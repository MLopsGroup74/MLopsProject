# assignment

Assignment repository MLops

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```

## Project Description

The overall goal of the project is to use a CNN to solve a classification task of classifying pokemons into types. Doing this, we aim to utilize and familiarize the tools learned in this course with a focus on reproducibility and ability to collaborate as a group, using git, docker and other relevant tools.  We hope to develop a clean and structured “code” setup, for instance using templates, such that we in the future can refer back to this project for guidance on how to approach larger team based machine learning projects where there is a need for easy collaboration on different machines. 

We are going to be using the dataset from kaggle: Pokemon dataset (https://www.kaggle.com/datasets/lantian773030/pokemonclassification/data)  which consists of 7000 images of pokemon from 150 different types with the number of images per class ranging from 26 to 49. The train data will have information about the specific type and the test data will not have said information. The dataset was created by Lance Zhang six years ago (2019). 

We intend on performing image classification using a Convolutional Neural Network, CNN, model. We use PyTorch as the core deep learning framework. We will be using the frameworks PyTorch Vision used for computer vision modelling, PyTorch Lightning for training and Scikit-Learn for datasplitting and evaluation metrics.
The information about what the different frameworks are used for comes from https://landscape.pytorch.org/.

Our model will be based on a model from kaggle, called Pokemon Classify Pytorch Lightning 
CNN (https://www.kaggle.com/code/stpeteishii/pokemon-classify-pytorch-lightning-cnn) by STPETE_ISHII. This model will serve as a baseline and might be adapted if needed. The original model is released under the Apache 2.0 open source license permitting modification and redistribution.
 This ensures that the project follows established open-source practices and aligns with academic standards for transparency and reproducibility.


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

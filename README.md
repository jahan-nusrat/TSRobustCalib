# Project Title: Improving Robustness and Calibration of Neural Networks for Time Series Data

This project focuses on improving the robustness and calibration of neural networks for time-series data, specifically for ECG signal classification. The goal is to develop and evaluate methods such as ensemble learning and Manifold Mixup to improve the reliability of predictions in real-world healthcare applications.

This repository follows a structured template to ensure reproducibility and organization, suitable for thesis projects at KIS\*MED.

## Repository Structure

The structure of the project is organized as follows:

```
├── README.md          <- The top-level README for using and installing this project.
├── data               <- The content of this folder is not tracked by git
│   ├── interim        <- Intermediate data that has been cleaned up, transformed, ...
│   ├── processed      <- The final data for modeling and visualizations.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Documentation of the project, notes, datasheets. Could use mkdocs
│
├── models             <- Trained and serialized models
│
├── notebooks          <- Jupyter notebooks (only python-projects)
│
├── report             <- Latex code of your thesis
│   └── figures        <- Generated graphics and figures to be used in the report
│
├── presentation       <- Contains the final presentation (e.g. .ppx) and all media used
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment,
│                         (only python-projects)
│
└── code_project_name   <- Source code for use in this project. Rename accordingly!
    │                    The following are example files that could be part of a python project
    │
    ├── __init__.py             <- Makes code_project_name a Python package
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    └── plots.py                <- Code to create visualizations
```

## Installation

### Python Environment

### 1. Clone the repository:

The code is tested with Python Version 3.9. We recommend using Miniconda: [Installing Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/)

```
gh repo clone jahan-nusrat/TSRobustCalib

cd jh_students
```

### 2. Create and activate a new Conda environment:

```
conda create -n ecg_project python=3.9

conda activate ecg_project
```

### 3. Install required packages: `pip install -r requirements.txt`

Or using setuptools install the project as package:
`pip install -e .`

## Usage

### 1. Place raw ECG datasets in the data/raw directory

### 2. Use the preprocessing script to clean, transform, and segment the data:

`python code_project_name/dataset.py`

> The cleaned and processed data will be saved in `data/processed`

## Training the Model

Train the model using the following command:
`python code_project_name/train.py --config config.yaml`

> Modify the parameters in config.yaml to adjust training settings (e.g., learning rate, batch size, etc.).

## Evaluation

Evaluate the trained model on the test dataset:
`python code_project_name/evaluate.py --model models/best_model.pth`

## Visualization

Generate plots for analysis and visualization: `python code_project_name/plots.py`

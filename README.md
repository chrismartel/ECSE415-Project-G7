# ECSE415-Project-G7

Pipeline to detect, localize and track vehicles in image sequences

## Requirements

The list of project requirements and dependencies are located in `./requirements.txt`.

## Dataset and Classification

The dataset and classification code is located in `./classification/classification.ipynb` and can be run with Google Colab.

## Detection and Localization

The detection and classification code is located in `./detection/detection.ipynb` and can be run locally or with Google Colab by downloading the image sequences dataset and importing the tools directory (`./detection/tools`) to the Colab runtime.

### Paths

- Set the `model_path` and `config_path` to the locations of the model and model config files respectively.
- Set the `dataset_path` to location of the image sequence dataset.

## Tracking

The tracking code is located in `./tracking/tracking.ipynb` and can both be run with Google Colab.

#### Train model

To train a model, we can run the python script: train.py in the following format

python3 train.py -p models/\<new model name\>.sav -y models/configs/\<hyperparameters config file name\>.yaml

The argument p is the path where the model is to be saved in and the argument y is a yaml file with the desired parameter values, a default file "models/configs/svm_default.yaml" is provided.

If no arguments are provided, the default path of the saved model will be models/configs/svm_default.sav with the parameters in the yaml file: models/svm_default.yml.

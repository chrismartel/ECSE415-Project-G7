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

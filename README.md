# ECSE415-Project-G7

Pipeline to detect, localize and track vehicles in image sequences

## Requirements

The list of project requirements and dependencies are located in `./requirements.txt`.

## Dataset and Classification

The dataset and classification code is located in `./classification/classification.ipynb` and can be run with Google Colab. 

The notebook contains all the methods required to download data, build datasets, perform preprocessing on images, perform cross-validation on SVM and random forest classifiers, and train and save SVM classifiers and associated hyper-parameters files. It also contains a bunch of hyper-parameter tuning experiments.

## Detection and Localization

The detection and classification code is located in `./detection/detection.ipynb` and can be run locally or with Google Colab by uploading the image sequences dataset to the Colab runtime.

#### Paths

- Set the `model_path` and `config_path` to the locations of the model and model config files respectively.
- Set the `dataset_path` to location of the image sequence dataset.

## Tracking

The tracking code is located in `./tracking/tracking.ipynb` and can both be run with Google Colab.

The annotated tracking videos are located at `./tracking/lucas-kanade-tracker.mp4` and `./tracking/alternate_tracker.mp4`.

#### Paths

- Set the `project_path` to the locations of a Google Drive folder containing your model and dataset files.
- Set the `database_path` to the location of the 0003 image sequence dataset.
- Set the `model_path` to the location of the classification model used.

## Deep Learning Implementation 

The code for the deep learning implementation of detection is located in `./cnn/cnn.ipynb` and can both be run with Google Colab.

#### Paths

- Set the `drive_path` to the locations of a Google Drive folder containing your model and dataset files.
- Set the `dataset_path` to the location of the image sequence dataset within your project folder.

#!/usr/bin/python
from os import remove
import sys, getopt
import numpy as np
import pickle
import yaml
from sklearn import svm
from sklearn.ensemble import BaggingClassifier

# Default parameters
model_path_default = 'models/svm.sav'
hyperparams_path_default = 'models/svm.yaml'


# Dataset parameters
positive_negative_ratio_default=1
min_intersection_ratio_default = 0.8
use_external_vehicle_samples_default = True
number_of_external_vehicle_samples_default = 800

# Preprocessing parameters
input_shape_default = (80,128)
orientations_default = 9 # number of orientation bins
pixels_per_cell_default = (12,16) # number of pixels per cell
cells_per_block_default = (4,4) # number of cells per block

# SVM parameters
C_default = 10
bagging_default = False
n_estimators_default = 1

# args:
# -p --model_path: path of the model to train and save
# -j --json_path: if provided, get hyperparameters from json file. Else use default parameters
def main(argv):


    model_path = model_path_default
    hyperparams_path = None

    try:
        opts, args = getopt.getopt(argv,"p:y:",["model_path=","hyperparams_path="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("train.py --model_path <path of model to train> --json_path <path of json file to use for hyperparameters>")
            sys.exit()
        elif opt in ("-p", "--model_path"):
            model_path = arg
        elif opt in ("-y", "--hyperparams_path"):
            json_path = arg

    if hyperparams_path is None:
        hyperparams_path = hyperparams_path_default
    else:
        with open(hyperparams_path, "r") as yaml_file:
            hyperparameters = yaml.load(yaml_file)

        # Get model hyperparameters
        positive_negative_ratio = hyperparameters['dataset']['positive_negative_ratio']
        min_intersection_ratio = hyperparameters['dataset']['min_intersection_ratio']
        use_external_vehicle_samples = hyperparameters['dataset']['use_external_vehicle_samples']
        number_of_external_vehicle_samples = hyperparameters['dataset']['number_of_external_vehicle_samples']
        
        input_shape = hyperparameters['preprocessing']['input_shape']['y'], hyperparameters['preprocessing']['input_shape']['x']
        orientations = hyperparameters['preprocessing']['orientations']
        pixels_per_cell = hyperparameters['preprocessing']['pixels_per_cell']['y'], hyperparameters['preprocessing']['pixels_per_cell']['x']
        cells_per_block = hyperparameters['preprocessing']['cells_per_block']['y'], hyperparameters['preprocessing']['cells_per_block']['x']
        
        C = hyperparameters['svm']['C']
        bagging = hyperparameters['svm']['bagging']
        n_estimators = hyperparameters['svm']['n_estimators']


    # import python file tools from repo at runtime
    sys.path.insert(1, '/tools')
    from data import build_dataset, download_datasets, remove_datasets
    from preprocessing import preprocess_sequences

    # download dataset

    # download & build dataset
    download_datasets()
    sequences = build_dataset(positive_negative_ratio=positive_negative_ratio,min_intersection_ratio=min_intersection_ratio, use_external_vehicle_samples=use_external_vehicle_samples, number_of_external_vehicle_samples=number_of_external_vehicle_samples)
    remove_datasets()

    # Preprocess
    features, labels = preprocess_sequences(sequences, input_shape, orientations, pixels_per_cell, cells_per_block)

    # Train

    # Build Train set
    x_train, y_train = None, None
    for i in range(4):
        if x_train is None:
            x_train = features[i]
        else:
            x_train = np.concatenate((x_train,features[i]),axis=0)

        if y_train is None:
            y_train = labels[i]
        else:
            y_train = np.concatenate((y_train,labels[i]),axis=0)

    # Bagging classifier
    if bagging:
        model = BaggingClassifier(base_estimator=svm.SVC(gamma='scale', C=C), n_estimators=n_estimators, random_state=None)

    # SVM Binary Classifier
    else:
        model = svm.SVC(gamma='scale', C=C)

    # Train
    model.fit(x_train, y_train)

    # Save model
    pickle.dump(model,open(model_path, 'wb'))

if __name__ == "__main__":
   main(sys.argv[1:])
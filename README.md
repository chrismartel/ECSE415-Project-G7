# ECSE415-Project-G7
Pipeline to detect, localize and track vehicles in image sequences

#### Train model
To train a model, we can run the python script: train.py in the following format

!python train.py -p models/svm_test.sav -y models/svm_default.yaml

The argument p is the path where the model is saved in and the argument y is a yaml file with the desired parameter values, a default file is provided.
If no arguments are provided, the default path of the saved model will be ECSE415-Project-G7/classification/models/svm_default.sav with the parameters in the yaml file: ECSE415-Project-G7/classification/models/svm_default.yml.

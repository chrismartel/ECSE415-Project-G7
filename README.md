# ECSE415-Project-G7
Pipeline to detect, localize and track vehicles in image sequences

#### Train model
To train a model, we can run the python script: train.py in the following format

python3 train.py -p models/<new model name>.sav -y models/configs/<hyperparameters config file name>.yaml

The argument p is the path where the model is to be saved in and the argument y is a yaml file with the desired parameter values, a default file "models/configs/svm_default.yaml" is provided.

If no arguments are provided, the default path of the saved model will be models/configs/svm_default.sav with the parameters in the yaml file: models/svm_default.yml.

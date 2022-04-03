import pickle
import json
import yaml

def train_and_save_model(path, version='v2', save_model=True, save_config=True, positive_negative_ratio=1, min_intersection_ratio=0.8, use_external_vehicle_samples=True, number_of_positive_samples=800, input_shape=(64,64),orientations=9, pixels_per_cell=(12,16),cells_per_block=(4,4),C=1, bagging=False, n_estimators=10):
  '''
    Train a svm classifier with HoG features and save it.

    path: model path and name ex: classification/models/svm1

    Return the trained classifier.
  '''

  # ----------------------------------------- #
  # ------------- Version 1 ----------------- #
  # ----------------------------------------- #

  if version == 'v1':
    # Build Dataset
    sequences = build_dataset(positive_negative_ratio=positive_negative_ratio,min_intersection_ratio=min_intersection_ratio, use_external_vehicle_samples=use_external_vehicle_samples, number_of_positive_samples=number_of_positive_samples)
    # Preprocess
    features, labels = preprocess_sequences(sequences, input_shape, orientations, pixels_per_cell, cells_per_block)

    # Build Train set: sequences 0, 1 and 2
    x_train, y_train = None, None
    for i in range(3):
      if x_train is None:
        x_train = features[i]
      else:
        x_train = np.concatenate((x_train,features[i]),axis=0)

      if y_train is None:
        y_train = labels[i]
      else:
        y_train = np.concatenate((y_train,labels[i]),axis=0) 

  # ----------------------------------------- #
  # ------------- Version 2 ----------------- #
  # ----------------------------------------- #

  elif version == 'v2':
    # Build Dataset
    imgs, labels = build_dataset_v2(positive_negative_ratio=positive_negative_ratio, number_of_positive_samples=number_of_positive_samples)

    # Preprocess
    features = preprocess_images_v2(imgs, orientations, pixels_per_cell, cells_per_block)

    x_train, y_train = features, labels

  # Bagging classifier
  if bagging:
    model = BaggingClassifier(base_estimator=svm.SVC(gamma='scale', C=C), n_estimators=n_estimators, random_state=None)

  # SVM Binary Classifier
  else:
    model = svm.SVC(gamma='scale', C=C)

  # Train
  model.fit(x_train, y_train)

  # Save model
  if save_model:
    pickle.dump(model,open(path+'.sav', 'wb'))

  # Save model hyperparameters
  if save_config:
    data = dict()
    data['dataset'] = dict()
    data['preprocessing'] = dict()
    data['svm'] = dict()

    data['dataset']['positive_negative_ratio'] = positive_negative_ratio
    data['dataset']['min_intersection_ratio'] = min_intersection_ratio
    data['dataset']['use_external_vehicle_samples'] = use_external_vehicle_samples
    data['dataset']['number_of_positive_samples'] = number_of_positive_samples
    
    data['preprocessing']['input_shape'] = dict()
    data['preprocessing']['input_shape']['y'], data['preprocessing']['input_shape']['x'] = input_shape[0], input_shape[1]
    data['preprocessing']['orientations'] = orientations

    data['preprocessing']['pixels_per_cell'] = dict()
    data['preprocessing']['pixels_per_cell']['y'], data['preprocessing']['pixels_per_cell']['x'] = pixels_per_cell[0], pixels_per_cell[1]
    data['preprocessing']['cells_per_block'] = dict()
    data['preprocessing']['cells_per_block']['y'],data['preprocessing']['cells_per_block']['x'] = cells_per_block[0], cells_per_block[1]
    
    data['svm']['C'] = C
    data['svm']['bagging'] = bagging
    data['svm']['n_estimators'] = n_estimators

    with open(path+'.yaml', "w") as outfile:
      yaml.dump(data, outfile)

  return model
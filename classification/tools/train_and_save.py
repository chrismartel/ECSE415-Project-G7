import pickle
import json
import yaml
import gc

import cross_validation
import data
import preprocessing

def train_and_save_model(path, save_model=True, save_config=True, train_sequences=sequence_ids_default,classifier_type=classifier_type_default, number_of_positive_samples=number_of_positive_samples_default, number_of_negative_samples=number_of_negative_samples_default, number_of_positive_samples_per_sequence=number_of_positive_samples_per_sequence_default, number_of_negative_samples_per_sequence=number_of_negative_samples_per_sequence_default, min_intersection_ratio=min_intersection_ratio_default, resize_shape=resize_shape_default, compute_spatial_features=compute_spatial_features_default, spatial_bins=spatial_bins_default, orientations=orientations_default, pixels_per_cell=pixels_per_cell_default,cells_per_block=cells_per_block_default,C=C_default, gamma=gamma_default, bagging=bagging_default, n_estimators=n_estimators_default, n_estimators_random_forest=n_estimators_random_forest_default, criterion=criterion_default, max_depth=max_depth_default, min_samples_split=min_samples_split_default):
  '''
    Train a svm classifier with HoG features and save it.


    path: model path and name ex: classification/models/svm1

    Return the trained classifier.
  '''

  # Build Dataset
  imgs, labels = build_dataset_from_udacity(number_of_positive_samples=number_of_positive_samples, number_of_negative_samples=number_of_negative_samples, visualize=False)
  seq_imgs, seq_labels = build_dataset_from_sequences(train_sequences,min_intersection_ratio=min_intersection_ratio, number_of_positive_samples_per_sequence=number_of_positive_samples_per_sequence, number_of_negative_samples_per_sequence=number_of_negative_samples_per_sequence, visualize=False)
  imgs, labels = imgs + seq_imgs, np.concatenate((labels,seq_labels), axis=0)

  # Preprocess
  features = preprocess_images(imgs, orientations=orientations, resize_shape=resize_shape, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, scaling=True, preprocessing_time=False, compute_spatial_features=False, spatial_bins=spatial_bins, visualize=False, grayscale=False)

  # Shuffle
  number_of_samples = features.shape[0]
  inds = list(range(number_of_samples))
  shuffle(inds)

  x_train, y_train = features[inds], labels[inds]

  # LINEAR SVM
  if classifier_type == 'rbf':
    # Bagging classifier
    if bagging:
      clf = BaggingClassifier(base_estimator=svm.SVC(gamma=gamma, C=C), n_estimators=n_estimators, random_state=None)
    # Binary Classifier
    else:
      clf = svm.SVC(gamma=gamma, C=C)
  
  # RBF SVM
  elif classifier_type == 'linear':
    # Bagging classifier
    if bagging:
      clf = BaggingClassifier(base_estimator=svm.LinearSVC(C=C), n_estimators=n_estimators, random_state=None)
    # Binary Classifier
    else:
      clf = svm.LinearSVC(C=C)

  # Random Forest
  elif classifier_type == 'rf':
    # Bagging classifier
    if bagging:
      clf = BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=n_estimators_random_forest, criterion=criterion), n_estimators=n_estimators, random_state=None)
    # Random Forest Classifier
    else:
      clf = RandomForestClassifier(n_estimators=n_estimators_random_forest, criterion=criterion)

  # Train
  clf.fit(x_train, y_train)

  # Save model
  if save_model:
    pickle.dump(clf,open(path+'.sav', 'wb'))

  # Save model hyperparameters
  if save_config:
    data = dict()
    data['dataset'] = dict()
    data['preprocessing'] = dict()
    data['clf'] = dict()

    data['dataset']['min_intersection_ratio'] = min_intersection_ratio
    data['dataset']['number_of_positive_samples'] = number_of_positive_samples
    data['dataset']['number_of_negative_samples'] = number_of_negative_samples
    data['dataset']['number_of_positive_samples_per_sequence'] = number_of_positive_samples_per_sequence
    data['dataset']['number_of_negative_samples_per_sequence'] = number_of_negative_samples_per_sequence
    
    data['preprocessing']['resize_shape'] = dict()
    data['preprocessing']['resize_shape']['y'], data['preprocessing']['resize_shape']['x'] = input_shape[0], input_shape[1]
    data['preprocessing']['orientations'] = orientations
    data['preprocessing']['compute_spatial_features'] = compute_spatial_features
    data['preprocessing']['spatial_bins'] = dict()
    data['preprocessing']['spatial_bins']['y'], data['preprocessing']['spatial_bins']['x'] = spatial_bins[0], spatial_bins[1]

    data['preprocessing']['pixels_per_cell'] = dict()
    data['preprocessing']['pixels_per_cell']['y'], data['preprocessing']['pixels_per_cell']['x'] = pixels_per_cell[0], pixels_per_cell[1]
    data['preprocessing']['cells_per_block'] = dict()
    data['preprocessing']['cells_per_block']['y'],data['preprocessing']['cells_per_block']['x'] = cells_per_block[0], cells_per_block[1]
    
    data['clf']['type'] = classifier_type
    data['clf']['C'] = C
    data['clf']['gamma'] = gamma
    data['clf']['bagging'] = bagging
    data['clf']['n_estimators'] = n_estimators

    with open(path+'.yaml', "w") as outfile:
      yaml.dump(data, outfile)

    del imgs
    del features
    del seq_imgs
    del seq_labels
    del labels
    gc.collect()

  return clf
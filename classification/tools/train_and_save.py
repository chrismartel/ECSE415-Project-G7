import pickle
import json
import yaml
import gc

# Dataset parameters
min_intersection_ratio = 0.8
number_of_positive_samples = 4000
number_of_negative_samples = 6000
number_of_positive_samples_per_sequence = 1000
number_of_negative_samples_per_sequence = 3000

# Preprocessing parameters
resize_shape = (64,64)
orientations = 9 # number of orientation bins
pixels_per_cell = (8,8) # number of pixels per cell
cells_per_block = (2,2) # number of cells per block
compute_spatial_features=False
spatial_bins =(8,8)

classifier_type = 'linear'

# SVM parameters
C = 1000
bagging = False
n_estimators = 1

# Random Forest params
n_estimators_random_forest = 500
criterion = 'entropy'
max_depth = None
min_samples_split = 2

sequence_ids = ['0','1','2']


def train_and_save_model(path, save_model=True, save_config=True, train_sequences=sequence_ids,classifier_type=classifier_type, number_of_positive_samples=number_of_positive_samples, number_of_negative_samples=number_of_negative_samples, number_of_positive_samples_per_sequence=number_of_positive_samples_per_sequence, number_of_negative_samples_per_sequence=number_of_negative_samples_per_sequence, min_intersection_ratio=min_intersection_ratio, resize_shape=resize_shape, compute_spatial_features=compute_spatial_features, spatial_bins=spatial_bins, orientations=orientations, pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block,C=C, bagging=bagging, n_estimators=n_estimators, n_estimators_random_forest=n_estimators_random_forest, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split):
  '''
    Train a svm classifier with HoG features and save it.

    path: model path and name ex: classification/models/svm1

    Return the trained classifier.
  '''

  # Build Dataset
  imgs, labels = build_dataset_from_udacity(number_of_positive_samples=number_of_positive_samples, number_of_negative_samples=number_of_negative_samples, visualize=False)
  seq_imgs, seq_labels = build_dataset_from_sequences(sequence_ids,min_intersection_ratio=min_intersection_ratio, number_of_positive_samples_per_sequence=number_of_positive_samples_per_sequence, number_of_negative_samples_per_sequence=number_of_negative_samples_per_sequence, visualize=False)
  imgs, labels = imgs + seq_imgs, np.concatenate((labels,seq_labels), axis=0)

  # Preprocess
  features = preprocess_images(imgs, orientations=orientations, resize_shape=resize_shape, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, scaling=True, preprocessing_time=False, compute_spatial_features=False, spatial_bins=spatial_bins, visualize=False, grayscale=True)

  # Shuffle
  number_of_samples = features.shape[0]
  inds = list(range(number_of_samples))
  shuffle(inds)

  x_train, y_train = features[inds], labels[inds]

  # LINEAR SVM
  if classifier_type == 'linear':
    # Bagging classifier
    if bagging:
      clf = BaggingClassifier(base_estimator=svm.SVC(gamma='scale', C=C), n_estimators=n_estimators, random_state=None)
    # Binary Classifier
    else:
      clf = svm.SVC(gamma='scale', C=C)
  
  # RBF SVM
  elif classifier_type == 'rbf':
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
import pickle
import json
import yaml

# Train Dataset parameters
positive_negative_ratio=1
number_of_positive_samples = 2000
sequences_positive_negative_ratio = 0.1
number_of_positive_samples_per_sequence = 50
min_intersection_ratio = 0.8
train_sequences = ['0','1','2']

# Preprocessing parameters
input_shape = (64,64)
orientations = 20 # number of orientation bins
pixels_per_cell = (2,2) # number of pixels per cell
cells_per_block = (4,4) # number of cells per block
compute_spatial_features = True
spatial_bins = (16,16)

# SVM parameters
classifier_type = 'linear'
C = 10
bagging = False
n_estimators = 1


visualize = False

def train_and_save_model_v2(path, save_model=True, save_config=True, train_sequences=train_sequences,classifier_type=classifier_type, number_of_positive_samples_per_sequence=number_of_positive_samples_per_sequence, sequences_positive_negative_ratio=sequences_positive_negative_ratio, positive_negative_ratio=positive_negative_ratio, min_intersection_ratio=min_intersection_ratio, number_of_positive_samples=number_of_positive_samples, input_shape=input_shape, compute_spatial_features=compute_spatial_features, orientations=orientations, pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block,C=C, bagging=bagging, n_estimators=n_estimators):
  '''
    Train a svm classifier with HoG features and save it.

    path: model path and name ex: classification/models/svm1

    Return the trained classifier.
  '''

  # Build Dataset
  imgs, labels = build_dataset_v2(positive_negative_ratio=positive_negative_ratio, number_of_positive_samples=number_of_positive_samples)
  sequences_imgs, sequences_labels = build_dataset_from_sequences(train_sequences,positive_negative_ratio=sequences_positive_negative_ratio,min_intersection_ratio=min_intersection_ratio, number_of_positive_samples=number_of_positive_samples_per_sequence, resize_shape=input_shape, visualize=visualize)
  imgs, labels = np.concatenate((imgs,sequences_imgs), axis=0), np.concatenate((labels,sequences_labels), axis=0)

  # Preprocess
  features = preprocess_images_v2(imgs, orientations, pixels_per_cell, cells_per_block, compute_spatial_features=compute_spatial_features, spatial_bins=spatial_bins)

  # Shuffle
  number_of_samples = features.shape[0]
  inds = list(range(number_of_samples))
  shuffle(inds)

  x_train, y_train = features[inds], labels[inds]

  if classifier_type == 'linear':
    # Bagging classifier
    if bagging:
      model = BaggingClassifier(base_estimator=svm.LinearSVC(C=C), n_estimators=n_estimators, random_state=None)

    # SVM Binary Classifier
    else:
      model = svm.LinearSVC(C=C)
  elif classifier_type == 'rbf':
    # Bagging classifier
    if bagging:
      model = BaggingClassifier(base_estimator=svm.SVC(gamma='scale',C=C), n_estimators=n_estimators, random_state=None)

    # SVM Binary Classifier
    else:
      model = svm.SVC(gamma='scale',C=C)

  # Train
  model.fit(x_train, y_train)

  # Save model
  if save_model:
    pickle.dump(model,open(path+'.sav', 'wb'))

  # Save model hyperparameters

# Sequences -> image sequences to use to train
train_sequences = ['0','1','2']
  if save_config:
    data = dict()
    data['dataset'] = dict()
    data['preprocessing'] = dict()
    data['svm'] = dict()

    data['dataset']['positive_negative_ratio'] = positive_negative_ratio
    data['dataset']['min_intersection_ratio'] = min_intersection_ratio
    data['dataset']['number_of_positive_samples'] = number_of_positive_samples
    data['dataset']['number_of_positive_samples_per_sequence'] = number_of_positive_samples_per_sequence
    data['dataset']['sequences_positive_negative_ratio'] = sequences_positive_negative_ratio
    
    data['preprocessing']['input_shape'] = dict()
    data['preprocessing']['input_shape']['y'], data['preprocessing']['input_shape']['x'] = input_shape[0], input_shape[1]
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

  return model
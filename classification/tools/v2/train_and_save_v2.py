import pickle
import json
import yaml

# Train Dataset parameters
positive_negative_ratio_default=1
number_of_train_positive_samples_default = 2000
# number_of_train_negative_samples_default = 4000
min_intersection_ratio_default = 0.8

# Preprocessing parameters
input_shape_default = (64,64)
orientations_default = 9 # number of orientation bins
pixels_per_cell_default = (16,16) # number of pixels per cell
cells_per_block_default = (4,4) # number of cells per block
compute_spatial_features_default = True

# SVM parameters
C_default = 10
bagging_default = False
n_estimators_default = 1

# Sequences -> image sequences to use to train
train_sequences_default = ['0','1','2']

visualize = False

def train_and_save_model_v2(path, save_model=True, save_config=True, train_sequences=train_sequences_default,positive_negative_ratio=positive_negative_ratio_default, min_intersection_ratio=min_intersection_ratio_default, number_of_train_positive_samples=number_of_train_positive_samples_default, input_shape=input_shape_default, compute_spatial_features=compute_spatial_features_default, orientations=orientations_default, pixels_per_cell=pixels_per_cell_default,cells_per_block=cells_per_block_default,C=C_default, bagging=bagging_default, n_estimators=n_estimators_default):
  '''
    Train a svm classifier with HoG features and save it.

    path: model path and name ex: classification/models/svm1

    Return the trained classifier.
  '''

  # Build Dataset
  imgs, labels = build_dataset_v2(positive_negative_ratio=positive_negative_ratio, number_of_positive_samples=int(number_of_train_positive_samples/2))
  sequences_imgs, sequences_labels = build_dataset_from_sequences(train_sequences,positive_negative_ratio=positive_negative_ratio,min_intersection_ratio=min_intersection_ratio, number_of_positive_samples=int(number_of_train_positive_samples/2), resize_shape=input_shape, visualize=visualize)
  imgs, labels = np.concatenate((imgs,sequences_imgs), axis=0), np.concatenate((labels,sequences_labels), axis=0)

  # Preprocess
  features = preprocess_images_v2(imgs, orientations, pixels_per_cell, cells_per_block)

  # Shuffle
  number_of_samples = features.shape[0]
  inds = list(range(number_of_samples))
  shuffle(inds)

  x_train, y_train = features[inds], labels[inds]

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
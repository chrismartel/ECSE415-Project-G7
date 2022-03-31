from skimage.transform import resize
from skimage.feature import hog
import numpy as np
from random import shuffle
import time

def preprocess_image(image, resize_shape, orientations, pixels_per_cell, cells_per_block, preprocessing_time=False, visualize=False):
  '''
  Preprocess an image.

  sequences: dict - Each key contains a list of tuples. Each tuple has the following format (np.array(h,w,3), integer label)
  resize_shape: tuple (new_height, new_width) - resizing shape
  orientations: int - number of orientation bins in hog feature computations
  pixels_per_cells: tuple (h,w) - number of pixels per cell in hog feature computation
  cells_per_block: tuple (h,w) - number of cells per block in hog feature computation
  preprocessing_time: indicates if preprocessing time must be computed and returned
  visualize: indicates if HoG image must be returned along with the feature vector.

  return (features, (optional)hog_img, (optional)time) - HoG features extracted from the image, the HoG image, and the preprocessing time
  '''
  start = time.time()

  # Resize image
  resized = resize(image, resize_shape)

  # Min-max normalization
  min, max = np.min(resized), np.max(resized)
  normalized = (resized - min) / (max - min)

  # Extract HoG features
  if visualize:
    feature, hog_img = hog(normalized, orientations=orientations, pixels_per_cell = pixels_per_cell, cells_per_block=cells_per_block, feature_vector=True, multichannel=True, visualize=visualize)
  else:
    feature = hog(normalized, orientations=orientations, pixels_per_cell = pixels_per_cell, cells_per_block=cells_per_block, feature_vector=True, multichannel=True, visualize=visualize)
  
  end = time.time()

  # return package
  ret = list()
  ret.append(feature)

  if visualize:
    ret.append(hog_img)

  if preprocessing_time:
     ret.append((end-start))

  if len(ret) == 1:
    return ret[0]
  else:
    return tuple(ret)

def preprocess_sequences(sequences, resize_shape, orientations, pixels_per_cell, cells_per_block, preprocessing_time=False):
  '''
  Preprocess a series of extracted image from image sequences sequence. 

  sequences: dict - Each key contains a list of tuples. Each tuple has the following format (np.array(h,w,3), integer label)
  resize_shape: tuple (new_height, new_width) - resizing shape
  orientations: int - number of orientation bins in hog feature computations
  pixels_per_cells: tuple (h,w) - number of pixels per cell in hog feature computation
  cells_per_block: tuple (h,w) - number of cells per block in hog feature computation

  return (features, labels) - features and labels are dict. Each key represents the sequence id.
  '''
  number_of_sequences = len(sequences.keys())

  if preprocessing_time:
    count = 0
    time_sum = 0

  # compute number of hog features
  number_hog_features = len(hog(np.zeros(resize_shape), orientations=orientations, pixels_per_cell = pixels_per_cell, cells_per_block=cells_per_block, feature_vector=True))
  
  features = dict()
  labels = dict()

  for i, seq in enumerate(sequences.values()):
    number_of_samples = len(seq) # number of image samples in sequence

    seq_features = np.zeros((number_of_samples, number_hog_features)) # x
    seq_labels = np.zeros(number_of_samples) # y

    for j, (sample, label) in enumerate(seq):

      # Extract HoG features
      seq_labels[j] = label

      if preprocessing_time:
        seq_features[j], time = preprocess_image(sample, resize_shape, orientations, pixels_per_cell, cells_per_block, preprocessing_time=preprocessing_time)
        count += 1
        time_sum += time
      else:
        seq_features[j] = preprocess_image(sample, resize_shape, orientations, pixels_per_cell, cells_per_block, preprocessing_time=preprocessing_time)

    features[i] = seq_features
    labels[i] = seq_labels

  if preprocessing_time:
    return features, labels, (time_sum/count)
  else:
    return features, labels


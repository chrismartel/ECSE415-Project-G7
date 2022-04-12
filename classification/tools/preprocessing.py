from skimage.transform import resize
from skimage.feature import hog
import numpy as np
from random import shuffle
import time
from sklearn.preprocessing import StandardScaler
import cv2 as cv

def preprocess_images(imgs, orientations=9, resize_shape=(64,64), pixels_per_cell=(2,2), cells_per_block=(4,4), scaling=True, preprocessing_time=False, compute_spatial_features=False, spatial_bins=(8,8), visualize=False, grayscale=Falsee):
  '''
  Preprocess an image.

  Params
  ------
  img: numpy array of shape (N, H,W,C)
  resize_shape: tuple - shape to which image are resized before hof computations
  scaling: boolean - indicates if feature vector must be scaled
  grayscale: boolean - indicate if image must be converted to grayscale
  orientations: int - number of orientation bins in hog feature computations
  pixels_per_cells: tuple (h,w) - number of pixels per cell in hog feature computation
  cells_per_block: tuple (h,w) - number of cells per block in hog feature computation
  preprocessing_time: indicates if preprocessing time must be computed and returned

  Return
  ------
  return (features, (optional)hog_img, (optional)time) - HoG features extracted from the images,
  the HoG images, and the preprocessing times in ms
  '''
  N = len(imgs)
  H, W = resize_shape
  exec_times_ms = np.zeros(N)

  features = None

  if grayscale:
    multichannel = False
  else:
    C = 3
    multichannel = True

  if visualize:
    hog_imgs = None

  for i in range(N):
    start = time.time() # preprocessing start
    img = imgs[i]

    # 1. Convert to gray scale
    if grayscale:
      img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # 2. Resize
    img = resize(img,(H,W))

    # 4. Features Extraction

    # 4.1 HoG
    if visualize:
      hog_feature, hog_img = hog(img, orientations=orientations, pixels_per_cell = pixels_per_cell, cells_per_block=cells_per_block, feature_vector=True, multichannel=multichannel, visualize=True)
    else:
      hog_feature = hog(img, orientations=orientations, pixels_per_cell = pixels_per_cell, cells_per_block=cells_per_block, feature_vector=True, multichannel=multichannel, visualize=False)

    # 4.2 Spatial
    if compute_spatial_features:
      spatial_img = resize(img,spatial_bins)
      spatial_feature = spatial_img.flatten()
      feature = np.concatenate((hog_feature,spatial_feature), axis=-1)
    else:
      feature = hog_feature

    end = time.time() # preprocessing done

    # inti features array
    if features is None:
      features = np.zeros((N, len(feature)))
    features[i] = feature

    if visualize:
      if hog_imgs is None:
        hog_imgs = np.zeros((N,hog_img.shape[0],hog_img.shape[1]))
      hog_imgs[i] = hog_img

    exec_times_ms[i] = (end - start) * 1000 # in ms

  # 5. Scaling
  if scaling:
    # Scaler
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)

  # return package
  ret = list()
  ret.append(features)

  if preprocessing_time:
    ret.append(np.mean(exec_times_ms))

  if visualize:
    ret.append(hog_imgs)

  if len(ret) == 1:
    return ret[0]
  else:
    return tuple(ret)

seq_features, features = None, None
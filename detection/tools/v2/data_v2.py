import os
import cv2 as cv
import numpy as np
import os
from random import randint, shuffle, choice
from skimage.transform import resize

# -------------------------------------------------------- #
# ------------- Download and Delete Datasets ------------- #
# -------------------------------------------------------- #

def download_datasets(datasets):

  cmds = list()
  # McGill ECSE415 Provided Image Sequences
  if 'ecse415' in datasets and not os.path.exists("dataset"):
    cmds.append('wget -nc -O dataset.zip https://mcgill-my.sharepoint.com/:u:/g/personal/raghav_mehta_mail_mcgill_ca/EVEvhY9_jyVEk2uSZ8wZhFYBQ58C57I7ZB55jBocKwB5Jg?download=1')
    cmds.append('unzip dataset.zip')
    cmds.append('rm dataset.zip')

  # Stanford University Cars Dataset
  if 'stanford' in datasets and not os.path.exists("cars_test"):
    cmds.append('wget -nc http://ai.stanford.edu/~jkrause/car196/cars_test.tgz')
    cmds.append('tar -xf cars_test.tgz')
    cmds.append('rm cars_test.tgz')

  # Udacity Vehicles and Non-Vehicles Dataset
  if 'udacity' in datasets:
    if not os.path.exists("vehicles"):
      cmds.append('wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip')
      cmds.append('unzip vehicles.zip')
      cmds.append('rm vehicles.zip')

    if not os.path.exists("non-vehicles"):
      cmds.append('wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip')
      cmds.append('unzip non-vehicles.zip')
      cmds.append('rm non-vehicles.zip')

    if os.path.exists("__MACOSX"):
      cmds.append('rm -r __MACOSX')

  for cmd in cmds:
    os.system(cmd)

def remove_datasets(datasets):
  cmds = list()
  
  if 'ecse415' in datasets and os.path.exists("dataset"):
    cmds.append('rm -r dataset')

  if 'stanford' in datasets and os.path.exists("cars_test"):
    cmds.append('rm -r cars_test')

  if 'udacity' in datasets:
    if os.path.exists("vehicles"):
      cmds.append('rm -r vehicles')

    if os.path.exists("non-vehicles"):
      cmds.append('rm -r non-vehicles')

  for cmd in cmds:
    os.system(cmd)

# -------------------------------------------------------- #
# ------------- Build Dataset ---------------------------- #
# -------------------------------------------------------- #

def build_dataset_v2(positive_negative_ratio=1, number_of_positive_samples=2000, resize_shape=(64,64)):
  '''
      Build a dataset from udacity vehicle/non-vehicle dataset.

      positive_negative_ratio: The ratio of number of positive samples versus negative to generate in the dataset.
      number_of_positive_samples: The number of positive samples to use. The number of negative samples to use
                                  is computed from the positive-negative ratio and this number.

      return a tuple (imgs, labels). imgs is a numpy array with shape N x H x W x C where N is the number of samples, H is the images height,
      W is the images width and C is the number of channels. labels correspond to the labels associated with each image
  '''

  number_of_negative_samples = int((1.0/positive_negative_ratio)*number_of_positive_samples)

  # N x H x W x C
  N = number_of_positive_samples+number_of_negative_samples
  H, W = resize_shape
  C = 3

  imgs = np.zeros((N,H,W,C))
  labels = np.zeros((N))

  # ------------------------------------------------------ #
  # -------------------- Positive Set -------------------- #
  # ------------------------------------------------------ #

  vehicle_db_main_path = 'vehicles/'
  vehicle_db_subpaths= ['GTI_Far','GTI_Left','GTI_MiddleClose','GTI_Right','KITTI_extracted']

  vehicle_db_dict = dict()
  for db in vehicle_db_subpaths:
    vehicle_db_dict[db] = dict()
    vehicle_db_dict[db]['visited'] = list() # keep track of visited ids
    vehicle_db_dict[db]['length'] = len(os.listdir(vehicle_db_main_path+db))

  for i in range(number_of_positive_samples):

    while (True):
      random_db = choice(list(vehicle_db_dict.keys()))
      random_id = randint(0,vehicle_db_dict[random_db]['length']-1)

      filepath = vehicle_db_main_path + random_db + '/' + os.listdir(vehicle_db_main_path+random_db)[random_id]
      if not filepath.endswith('.png'):
        continue
      if random_id in vehicle_db_dict[random_db]['visited']:
        continue
      break
    
    img = cv.imread(vehicle_db_main_path + random_db + '/' + os.listdir(vehicle_db_main_path+random_db)[random_id])
    if img is None:
      print(vehicle_db_main_path + random_db + '/' + os.listdir(vehicle_db_main_path+random_db)[random_id])
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    img = resize(img, resize_shape)
    imgs[i] = img
    labels[i] = 1

  # ------------------------------------------------------ #
  # -------------------- Negative Set -------------------- #
  # ------------------------------------------------------ #

  non_vehicle_db_main_path = 'non-vehicles/'
  non_vehicle_db_subpaths= ['Extras','GTI']

  non_vehicle_db_dict = dict()
  for db in non_vehicle_db_subpaths:
    non_vehicle_db_dict[db] = dict()
    non_vehicle_db_dict[db]['visited'] = list() # keep track of visited ids
    non_vehicle_db_dict[db]['length'] = len(os.listdir(non_vehicle_db_main_path+db))

  for i in range(number_of_negative_samples):

    while (True):
      random_db = choice(list(non_vehicle_db_dict.keys()))
      random_id = randint(0,non_vehicle_db_dict[random_db]['length']-1)

      filepath = non_vehicle_db_main_path + random_db + '/' + os.listdir(non_vehicle_db_main_path+random_db)[random_id]
      if not filepath.endswith('.png'):
        continue
      if random_id in non_vehicle_db_dict[random_db]['visited']:
        continue
      break
    
    img = cv.imread(non_vehicle_db_main_path + random_db + '/' + os.listdir(non_vehicle_db_main_path+random_db)[random_id])
    if img is None:
      print(non_vehicle_db_main_path + random_db + '/' + os.listdir(non_vehicle_db_main_path+random_db)[random_id])
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    img = resize(img, resize_shape)
    imgs[number_of_positive_samples+i] = img
    labels[number_of_positive_samples+i] = 0


  return imgs, labels


import numpy as np

def dataset_statistics_v2(imgs, labels, statistic_types, query_labels=[0,1]):
  '''
      Collect width, height, and aspect ratios statistics from image dataset.

      imgs: a numpy array of the shape N x H x W x C
      labels: a numpy array of the shape N
      
      statistics: list containing the statistics to compute. Can contain the following values: 'width', 'height',
                  'aspect_ratio', 'class_distribution'.
      
      labels: list of labels indicating from which image we want to collect statistics. 
      
      return a dictionary of statistics. For height, width, and aspect ratio, the value is a min-max tuple.
                                         For class_distribution, the value is an array of counts per class.
  '''
  # Number of samples
  number_of_samples = imgs.shape[0]

  # Class distribution
  if 'class_distribution' in statistic_types:
    class_count = np.zeros(2)

  # compute statistics on full image dataset
  if 'aspect_ratio' in statistic_types:
    aspect_ratios = np.zeros(number_of_samples)

  if 'width' in statistic_types:
    widths = np.zeros(number_of_samples)

  if 'height' in statistic_types:
    heights = np.zeros(number_of_samples)

  i = 0
  for i in range(number_of_samples):
    # only collect statistics for specific labels
    if labels[i] not in query_labels:
      continue

    if 'width' in statistic_types:
      widths[i] = imgs[i].shape[2]

    if 'height' in statistic_types:
      heights[i] = imgs[i].shape[1]

    if 'aspect_ratio' in statistic_types:
      aspect_ratios[i] = heights[i] / float(widths[i])
    
    if 'class_distribution' in statistic_types:
      class_count[labels[i]] += 1

  stats = dict()
  if 'width' in statistic_types:
    stats['widths'] = widths    

  if 'height' in statistic_types:
    stats['heights'] = heights

  if 'aspect_ratio' in statistic_types:
    stats['aspect_ratios'] = aspect_ratios
          
  if 'class_distribution' in statistic_types:
    stats['class_distribution'] = class_count
  
  return stats


def build_dataset_from_sliding_window(min_intersection_ratio=0.8, number_of_samples=1000, resize_shape=(64,64)):
  '''
      Build a dataset from provided image sequences and other external datasets. The built dataset consists of a 
      dictionary. Each key corresponds to a sequence of images. The image sequences can be split and used for training.

      positive_negative_ratio: The ratio of number of positive samples versus negative to generate in the dataset.
      min_intersection_ratio: The minimum ratio of a random generated patch vs. a vehicle bbox to be considered a vehicle

      return sequences, a dictionary containing list of images in each sequence.
  '''

  # Build sequence dictionary
  number_of_sequences = 4
  number_of_samples_per_sequence = int(number_of_samples/number_of_sequences)

  N = number_of_samples
  H, W = resize_shape
  C = 3
  imgs = np.zeros((N,H,W,C))
  labels = np.zeros(number_of_samples_per_sequence*number_of_sequences)
  count = 0

  for seq_id in range(number_of_sequences):
    
    bboxes = parse('dataset/000{seq_id}.txt'.format(seq_id=seq_id))

    img = cv.imread('dataset/000{seq_id}/000001.png'.format(seq_id=seq_id))
    windows = slidingWindow(img.shape[0:2], init_size=(64,64), x_overlap=0.5, y_step=0.05,x_range=(0, 1), y_range=(0, 1), scale=1.5, dims=False)

    for i in range(number_of_samples_per_sequence):
      frame_id = choice(list(bboxes.keys()))
      vehicle_bboxes = bboxes[frame_id]
      img = cv.imread('dataset/000{seq_id}/{frame_id:06d}.png'.format(seq_id=seq_id, frame_id=frame_id))
      img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
      random_window = choice(windows)

      random_window_x1, random_window_y1, random_window_x2, random_window_y2 = random_window
      random_window_area = area(random_window)

      # check intersection with vehicles
      is_vehicle = 0
      for (id, x1,y1,x2,y2) in vehicle_bboxes:
        vehicle_bbox = (x1,y1,x2,y2)
        vehicle_area = area(vehicle_bbox)

        intersection_bbox = intersection(vehicle_bbox, random_window)
        
        # not a vehicle
        if intersection_bbox is None:
          continue
        else:
          intersection_area = area(intersection_bbox)
          # check for minimal intersection
          if intersection_area/vehicle_area > min_intersection_ratio and intersection_area/random_window_area > min_intersection_ratio:       
            is_vehicle = 1
            break

    random_bbox_img = img[random_window_y1:random_window_y2,random_window_x1:random_window_x2]
    random_bbox_img = resize(random_bbox_img,(input_shape))
    imgs[count] = random_bbox_img
    labels[count] = is_vehicle
    count += 1

  return imgs, labels
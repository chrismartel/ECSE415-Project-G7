import os
import cv2 as cv
import numpy as np
from random import randint, shuffle, choice
from skimage.transform import resize
import matplotlib.pyplot as plt

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



def build_dataset_v2(positive_negative_ratio=1, number_of_positive_samples=2000, resize_shape=(64,64), visualize=False):
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

  samples_available = 0

  vehicle_db_dict = dict()
  for db in vehicle_db_subpaths:
    vehicle_db_dict[db] = dict()
    vehicle_db_dict[db]['length'] = len(os.listdir(vehicle_db_main_path+db))
    samples_available += vehicle_db_dict[db]['length']

    inds = list(range(vehicle_db_dict[db]['length']))
    shuffle(inds)
    vehicle_db_dict[db]['inds'] = inds
    vehicle_db_dict[db]['i'] = 0

  if samples_available < number_of_negative_samples:
      print("Error: {av} positive samples available, but {req} required.".format(av=samples_available, req=number_of_negative_samples))
      sys.exit(-1)

  for i in range(number_of_positive_samples):
    while(True):
      random_db = choice(list(vehicle_db_dict.keys()))

      if vehicle_db_dict[random_db]['i'] >= vehicle_db_dict[random_db]['length']:
        continue

      random_id = vehicle_db_dict[random_db]['inds'][vehicle_db_dict[random_db]['i']]
      vehicle_db_dict[random_db]['i'] += 1

      filepath = vehicle_db_main_path + random_db + '/' + os.listdir(vehicle_db_main_path+random_db)[random_id]
      if not filepath.endswith('.png'):
        continue
      break

    img = cv.imread(vehicle_db_main_path + random_db + '/' + os.listdir(vehicle_db_main_path+random_db)[random_id])

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    img = resize(img, resize_shape)
    imgs[i] = img
    labels[i] = 1

  # ------------------------------------------------------ #
  # -------------------- Negative Set -------------------- #
  # ------------------------------------------------------ #

  non_vehicle_db_main_path = 'non-vehicles/'
  non_vehicle_db_subpaths= ['Extras','GTI']

  samples_available = 0


  non_vehicle_db_dict = dict()
  for db in non_vehicle_db_subpaths:
    non_vehicle_db_dict[db] = dict()
    non_vehicle_db_dict[db]['length'] = len(os.listdir(non_vehicle_db_main_path+db))

    inds = list(range(non_vehicle_db_dict[db]['length']))
    samples_available += non_vehicle_db_dict[db]['length']

    shuffle(inds)
    non_vehicle_db_dict[db]['inds'] = inds
    non_vehicle_db_dict[db]['i'] = 0

  if samples_available < number_of_negative_samples:
      print("Error: {av} negative samples available, but {req} required.".format(av=samples_available, req=number_of_negative_samples))
      sys.exit(-1)

  for i in range(number_of_negative_samples):
    while(True):
      random_db = choice(list(non_vehicle_db_dict.keys()))

      if non_vehicle_db_dict[random_db]['i'] >= non_vehicle_db_dict[random_db]['length']:
        continue

      random_id = non_vehicle_db_dict[random_db]['inds'][non_vehicle_db_dict[random_db]['i']]
      non_vehicle_db_dict[random_db]['i'] += 1

      filepath = non_vehicle_db_main_path + random_db + '/' + os.listdir(non_vehicle_db_main_path+random_db)[random_id]
      if not filepath.endswith('.png'):
        continue
      break

    img = cv.imread(non_vehicle_db_main_path + random_db + '/' + os.listdir(non_vehicle_db_main_path+random_db)[random_id])

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    img = resize(img, resize_shape)
    imgs[number_of_positive_samples+i] = img
    labels[number_of_positive_samples+i] = 0


  return imgs, labels


def build_dataset_from_sequences(sequences,min_intersection_ratio=0.8, positive_negative_ratio=1, number_of_positive_samples=200, resize_shape=(64,64), visualize=False):
  '''
      Build a dataset from provided image sequences and other external datasets. The built dataset consists of a 
      dictionary. Each key corresponds to a sequence of images. The image sequences can be split and used for training.

      positive_negative_ratio: The ratio of number of positive samples versus negative to generate in the dataset.
      min_intersection_ratio: The minimum ratio of a random generated patch vs. a vehicle bbox to be considered a vehicle

      return sequences, a dictionary containing list of images in each sequence.
  '''

  # Build sequence dictionary
  number_of_sequences = len(sequences)
  number_of_negative_samples = int((1.0/positive_negative_ratio)*number_of_positive_samples)
  number_of_positive_samples_per_sequence = int(number_of_positive_samples/number_of_sequences)
  number_of_negative_samples_per_sequence = int(number_of_negative_samples/number_of_sequences)

  N = number_of_negative_samples_per_sequence*number_of_sequences + number_of_positive_samples_per_sequence*number_of_sequences
  H, W = resize_shape
  C = 3
  imgs = np.zeros((N,H,W,C))
  labels = np.zeros(N)
  global_count = 0

  for seq_id in sequences:
    positive_count = 0

    bboxes = parse('dataset/000{seq_id}.txt'.format(seq_id=seq_id))

    # init sliding window
    img = cv.imread('dataset/000{seq_id}/{frame_id:06d}.png'.format(seq_id=seq_id, frame_id=1))
    windows = slidingWindow(img.shape[0:2], init_size=(64,64), x_overlap=0.05, y_step=0.05,x_range=(0, 1), y_range=(0, 1), scale=1.2, dims=False)


    ################
    # NEGATIVE SET #
    ################

    # Apply sliding window and add samples with minimal interesection with vehicle boxes
    for i in range(number_of_negative_samples_per_sequence):

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

      random_bbox_img = resize(random_bbox_img,(resize_shape))

      imgs[global_count] = random_bbox_img
      labels[global_count] = is_vehicle

      if visualize:
        plt.imshow(imgs[global_count])
        plt.xticks([]), plt.yticks([])
        plt.show()  
        print("label: ",labels[global_count])

      if is_vehicle:
        positive_count += 1

      global_count += 1


    ################
    # POSITIVE SET #
    ################
    sequence_complete = False
    while(not sequence_complete):

      frame_id = choice(list(bboxes.keys()))
      vehicle_bboxes = bboxes[frame_id]
      img = cv.imread('dataset/000{seq_id}/{frame_id:06d}.png'.format(seq_id=seq_id, frame_id=frame_id))
      img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

      for (id, x1,y1,x2,y2) in vehicle_bboxes:

        if sequence_complete:
          break

        vehicle_bbox = (x1,y1,x2,y2)
        new_bboxes = more_bboxes(vehicle_bbox,img.shape[0:2], number_of_bboxes=1, x_scale=(0.98,1.02), y_scale=(0.98,1.02), x_translate=(-0.03,0.03), y_translate=(-0.03,0.03))

        for new_bbox in new_bboxes:

          if sequence_complete:
            break

          new_bbox_img = img[new_bbox[1]:new_bbox[3],new_bbox[0]:new_bbox[2]]
          new_bbox_img = resize(new_bbox_img,(resize_shape))

          imgs[global_count] = new_bbox_img
          labels[global_count] = 1
          positive_count += 1

          if visualize:
            plt.imshow(imgs[global_count])
            plt.xticks([]), plt.yticks([])
            plt.show()
            print("label: ",labels[global_count])

          if positive_count >= number_of_positive_samples_per_sequence:
            sequence_complete = True

          global_count += 1

  return imgs, labels
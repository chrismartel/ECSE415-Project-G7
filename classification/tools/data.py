
import os
import cv2 as cv
import numpy as np
from random import randint, shuffle, choice, uniform
from skimage.transform import resize
import sys
import matplotlib.pyplot as plt

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

def intersection(r1, r2):
  '''
      Compute intersection rectangle of two rectangles

      r1: tuple (x1,y1,x2,y2) coordinates of rectangle 1
      r2: tuple (x1,y1,x2,y2) coordinates of rectangle 2

      return area interesction rectangle
  '''
  # x axis

  # r2 in r1
  if r1[0] <= r2[0] and r1[2] >= r2[2]:
    x1, x2 = r2[0], r2[2]

  # r1 in r2
  elif r2[0] <= r1[0] and r2[2] >= r1[2]:
    x1, x2 = r1[0], r1[2]

  # r1 left of r2 and overlap
  elif r1[0] <= r2[0] and r1[2] >= r2[0]:
    x1, x2 = r2[0], r1[2]

  # r2 left of r1 and overlap
  elif r2[0] <= r1[0] and r2[2] >= r1[0]:
    x1, x2 = r1[0], r2[2]

  # no overlap
  else:
    return None

  # y axis

  # r2 in r1
  if r1[1] <= r2[1] and r1[3] >= r2[3]:
    y1, y2 = r2[1], r2[3]

  # r1 in r2
  if r2[1] <= r1[1] and r2[3] >= r1[3]:
    y1, y2 = r1[1], r1[3]

  # r1 top of r2 and overlap
  elif r1[1] <= r2[1] and r1[3] >= r2[1]:
    y1, y2 = r2[1], r1[3]

  # r2 top of r1 and overlap
  elif r2[1] <= r1[1] and r2[3] >= r1[1]:
    y1, y2 = r1[1], r2[3]

  # no overlap
  else:
    return None

  return int(x1), int(y1), int(x2), int(y2)


def area(r):
  '''
      Rectangle area

      r: tuple (x1,y1,x2,y2) coordinates of rectangle
      
      return area
  '''
  return (r[2]-r[0])*(r[3]-r[1])

def slidingWindow(image_size, init_size=(64,64), x_overlap=0.5, y_step=0.05,
        x_range=(0, 1), y_range=(0, 1), scale=1.5, dims=False):

    """
    Run a sliding window across an input image and return a list of the
    coordinates of each window.
    Window travels the width of the image (in the +x direction) at a range of
    heights (toward the bottom of the image in the +y direction). At each
    successive y, the size of the window is increased by a factor equal to
    @param scale. The horizontal search area is limited by @param x_range
    and the vertical search area by @param y_range.
    @param image_size (int, int): Size of the image (width, height) in pixels.
    @param init_size (int, int): Initial size of of the window (width, height)
        in pixels at the initial y, given by @param y_range[0].
    @param x_overlap (float): Overlap between adjacent windows at a given y
        as a float in the interval [0, 1), where 0 represents no overlap
        and 1 represents 100% overlap.
    @param y_step (float): Distance between successive heights y as a
        fraction between (0, 1) of the total height of the image.
    @param x_range (float, float): (min, max) bounds of the horizontal search
        area as a fraction of the total width of the image.
    @param y_range (float, float) (min, max) bounds of the vertical search
        area as a fraction of the total height of the image.
    @param scale (float): Factor by which to scale up window size at each y.
    @return windows: List of tuples, where each tuple represents the
        coordinates of a window in the following order: (upper left corner
        x coord, upper left corner y coord, lower right corner x coord,
        lower right corner y coord).
    """

    windows = []
    y_count = 0
    h, w = image_size[1], image_size[0]
    for y in range(int(y_range[0] * h), int(y_range[1] * h), int(y_step * h)):
        y_count += 1
        win_width = int(init_size[0] + (scale * (y - (y_range[0] * h))))
        win_height = int(init_size[1] + (scale * (y - (y_range[0] * h))))
        if y + win_height > int(y_range[1] * h) or win_width > w:
            break
        x_step = int((1 - x_overlap) * win_width)
        for x in range(int(x_range[0] * w), int(x_range[1] * w), x_step):
            windows.append((x, y, x + win_width, y + win_height))

    if (dims):
        return windows, y_count
    return windows


def more_bboxes(bbox,img_dim,  number_of_bboxes=10, x_scale=(0.98,1.02), y_scale=(0.98,1.02), x_translate=(-0.02,0.02), y_translate=(-0.02,0.02)):
  '''
      Generate list of bounding boxes around box of interest by applying random transations and scaling.

      @param bbox: the bounding box of interest
      @ img_dim: dimensions of the full image
      @ number_of_bboxes: the number of new bboxes to generate
      @ x_scale: how much to scale the new bboxes around the bbox of interest on the x axis
      @ y_scale: how much to scale the new bboxes around the bbox of interest on the y axis
      @ x_translate: how much to translate the new bboxes around the bbox of interest on the x axis
      @ y_translate: how much to translate the new bboxes around the bbox of interest on the y axis
  '''
  # bounds
  x_left_bound, x_right_bound = 0, img_dim[1]
  y_top_bound, y_bottom_bound = 0, img_dim[1]
  x1, y1, x2, y2 = bbox
  w = float(x2 - x1)
  h = float(y2 - y1)

  bboxes = list()

  # print("init:",bbox)
  for i in range(number_of_bboxes):
    while(True):
      random_x_scale =  uniform(x_scale[0],x_scale[1])
      random_y_scale = uniform(y_scale[0],y_scale[1])
      random_x_translate = uniform(w*x_translate[0],w*x_translate[1])
      random_y_translate = uniform(h*y_translate[0],h*y_translate[1])

      # print("scales: ",random_x_scale,random_y_scale,random_x_translate,random_y_translate)

      # Scale
      new_x1, new_x2 = int(x1 * (1/random_x_scale)), int(x2 * random_x_scale)
      new_y1, new_y2 = int(y1 * (1/random_y_scale)), int(y2 * random_y_scale)

      # Translate
      new_x1, new_x2 = int(new_x1 + random_x_translate), int(new_x2 + random_x_translate)
      new_y1, new_y2 = int(new_y1 + random_y_translate), int(new_y2 + random_y_translate)

      # print("post transform: ", new_x1, new_y1, new_x2, new_y2)
      #validate x1
      if new_x1 < x_left_bound:
        new_x1 = x_left_bound

      if new_x1 > x_right_bound:
        continue

      #validate x2
      if new_x2 < x_left_bound:
        continue

      if new_x2 > x_right_bound:
        new_x2 = x_right_bound

      #validate y1
      if new_y1 < y_top_bound:
        new_y1 = y_top_bound

      if new_y1 > y_bottom_bound:
        continue

      #validate y2
      if new_y2 < y_top_bound:
        continue

      if new_y2 > y_bottom_bound:
        new_y2 = y_bottom_bound

      # Validate
      if new_x2 <= new_x1 or new_y2 <= new_y1:
        continue

      # OK
      bboxes.append((new_x1,new_y1,new_x2,new_y2))
      break
  return bboxes

def build_dataset_from_sequences(sequences,min_intersection_ratio=0.9, number_of_positive_samples_per_sequence=1000, number_of_negative_samples_per_sequence=4000, visualize=False):
  '''
      Build a dataset from provided image sequences and other external datasets. The built dataset consists of a 
      dictionary. Each key corresponds to a sequence of images. The image sequences can be split and used for training.

      positive_negative_ratio: The ratio of number of positive samples versus negative to generate in the dataset.
      min_intersection_ratio: The minimum ratio of a random generated patch vs. a vehicle bbox to be considered a vehicle

  '''

  # Build sequence dictionary
  number_of_sequences = len(sequences)


  N = number_of_negative_samples_per_sequence*number_of_sequences + number_of_positive_samples_per_sequence*number_of_sequences

  imgs = list()
  
  labels = list()

  for seq_id in sequences:
    bboxes = parse('dataset/000{seq_id}.txt'.format(seq_id=seq_id))

    # init sliding window
    img = cv.imread('dataset/000{seq_id}/{frame_id:06d}.png'.format(seq_id=seq_id, frame_id=1))
    windows = slidingWindow(img.shape[0:2], init_size=(64,64), x_overlap=0.05, y_step=0.05,x_range=(0, 1), y_range=(0, 1), scale=1.1, dims=False)

    ################
    # NEGATIVE SET #
    ################

    # Apply sliding window and add samples with minimal interesection with vehicle boxes
    for i in range(number_of_negative_samples_per_sequence):

      frame_id = choice(list(bboxes.keys()))
      vehicle_bboxes = bboxes[frame_id]
      img = cv.imread('dataset/000{seq_id}/{frame_id:06d}.png'.format(seq_id=seq_id, frame_id=frame_id))


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

      imgs.append(random_bbox_img)
      labels.append(is_vehicle)

      if visualize:
        plt.imshow(random_bbox_img)
        plt.xticks([]), plt.yticks([])
        plt.show()  
        print("label: ",is_vehicle)

    ################
    # POSITIVE SET #
    ################
    sequence_complete = False
    positive_count = 0
    while(not sequence_complete):

      frame_id = choice(list(bboxes.keys()))
      vehicle_bboxes = bboxes[frame_id]
      img = cv.imread('dataset/000{seq_id}/{frame_id:06d}.png'.format(seq_id=seq_id, frame_id=frame_id))

      for (id, x1,y1,x2,y2) in vehicle_bboxes:

        if sequence_complete:
          break

        vehicle_bbox = (x1,y1,x2,y2)
        new_bboxes = more_bboxes(vehicle_bbox,img.shape[0:2], number_of_bboxes=1)

        for new_bbox in new_bboxes:

          if sequence_complete:
            break

          new_bbox_img = img[new_bbox[1]:new_bbox[3],new_bbox[0]:new_bbox[2]]

          imgs.append(new_bbox_img)
          labels.append(1)
          positive_count += 1

          if visualize:
            plt.imshow(new_bbox_img)
            plt.xticks([]), plt.yticks([])
            plt.show()
            print("label: ",1)

          if positive_count >= number_of_positive_samples_per_sequence:
            sequence_complete = True
  return imgs, np.array(labels)

def build_dataset_from_udacity(number_of_positive_samples=2000, number_of_negative_samples=4000, visualize=False):
  '''
      Build a dataset from udacity vehicle/non-vehicle dataset.

      number_of_negative_samples: The number of negative samples to use. 
      number_of_positive_samples: The number of positive samples to use. 

      return a tuple (imgs, labels). imgs is a numpy array with shape N x H x W x C where N is the number of samples, H is the images height,
      W is the images width and C is the number of channels. labels correspond to the labels associated with each image
  '''
  imgs = list()
  labels = list()

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
    imgs.append(img)
    labels.append(1)

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
    imgs.append(img)
    labels.append(0)


  return imgs, np.array(labels)


def dataset_statistics(imgs, labels, statistic_types, query_labels=[0,1]):
  '''
      Collect width, height, and aspect ratios statistics from image dataset.

      imgs: a list of images
      labels: a numpy array of the shape N
      
      statistics: list containing the statistics to compute. Can contain the following values: 'width', 'height',
                  'aspect_ratio', 'class_distribution'.
      
      labels: list of labels indicating from which image we want to collect statistics. 
      
      return a dictionary of statistics. For height, width, and aspect ratio, the value is a min-max tuple.
                                         For class_distribution, the value is an array of counts per class.
  '''
  # Number of samples
  number_of_samples = len(imgs)

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
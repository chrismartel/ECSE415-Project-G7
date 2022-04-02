import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2 as cv
from random import randint, shuffle, choice


# -------------------------------------------------------- #
# ------------- Download and Delete Datasets ------------- #
# -------------------------------------------------------- #

import os

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
# ------------- Parse ECSE415 Provided Image Sequences --- #
# -------------------------------------------------------- #

def parse(filepath):
  '''
      Parse gt.txt with format
        <frame>, <id>, <type>, <truncated>, <occluded>, <alpha>, <bb_left>, <bb_top>, <bb_right>, <bb_bottom>, <3D_H>, <3D_W>, <3D_L>, <x>, <y>, <z>, <ry>
      Return dict as:
        <type> = "Car", "Van", "Truck", "Pedastrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare"
        key: frame
        value: list - <id>, <bb_left>, <bb_top>, <bb_right>, <bb_bottom>, <is_vehicle>
      Feel free to edit your structure as needed!
  '''

  used_type = ["Car", "Van", "Truck", "Tram"]

  lines = open(filepath, "r").readlines()                                 
  bbox = {}

       #  <frame>, <id>, <truncated>, <occluded>, <alpha>, <bb_left>, <bb_top>, <bb_right>, <bb_bottom>, <3D_H>, <3D_W>, <3D_L>, <x>,   <y>,   <z>,   <ry>
  mask = [False,   True,  False,       False,      False,   True,      True,     True,       True,        False,  False,  False,  False, False, False, False]
  
  for line in lines:
    l = line.strip().split(' ') #convert line to list
    typ = l.pop(2)  # get type of bbox 
    line = np.asarray(l).astype(np.float32) # convert into array 
    frame, line = int(line[0]), line[mask] # get frame number and mask the line   
    if frame not in bbox.keys():
      bbox[frame] = []   
    if typ in used_type:
        bbox[frame].append(line)
  return bbox

def add_bbox(img, bbox, color=(255, 0, 0), thickness=2):
  ''' 
    annotate an image with bounding boxes:
    supports single bbox or list of bboxs
  '''

  annotated = np.copy(img)
  if bbox: 
    if isinstance(bbox[0], np.ndarray) or isinstance(bbox[0], list):
        for (_,x1,y1,x2,y2) in bbox:
            cv.rectangle(annotated, (x1, y1), (x2, y2), color , thickness)
    else:
        _,x1,y1,x2,y2 = bbox
        cv.rectangle(annotated, (x1, y1), (x2, y2), color , thickness)
  
  return annotated



# -------------------------------------------------------- #
# ------------- Build Dataset ---------------------------- #
# -------------------------------------------------------- #

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


# OLD VERSION
def build_dataset(positive_negative_ratio=1, min_intersection_ratio=0.8, use_external_vehicle_samples=False, number_of_external_vehicle_samples=500):
  '''
      Build a dataset from provided image sequences and other external datasets. The built dataset consists of a 
      dictionary. Each key corresponds to a sequence of images. The image sequences can be split and used for training.

      positive_negative_ratio: The ratio of number of positive samples versus negative to generate in the dataset.
      min_intersection_ratio: The minimum ratio of a random generated patch vs. a vehicle bbox to be considered a vehicle
      use_external_vehicle_samples: indicates if external data must be used.

      return sequences, a dictionary containing list of images in each sequence.
  '''

  # Build sequence dictionary
  number_of_sequences = 4
  sequences = dict()
  for i in range(number_of_sequences):
    sequences[i] = list()

  number_of_positive_samples = 0


  # -------------------- Positive Train Set -------------------- #

  for seq_id in range(number_of_sequences):
    visited_ids = list() # keep track of vehicle ids to avoid adding the same vehicle multiple times

    bboxes = parse('dataset/000{seq_id}.txt'.format(seq_id=seq_id))

    # Get positive samples from provided bounding boxes
    for frame_id, frame_bboxes in bboxes.items():
      img = cv.imread('dataset/000{seq_id}/{frame_id:06d}.png'.format(seq_id=seq_id, frame_id=frame_id))
      img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

      for bbox in frame_bboxes:
        id, x1, y1, x2, y2 = bbox.astype('int32')

        if id not in visited_ids:
          visited_ids.append(id)
          bbox_img = img[y1:y2,x1:x2]
          sequences[seq_id].append((bbox_img,1))
          number_of_positive_samples += 1

  # Add Car Images from Stanford University dataset
  if use_external_vehicle_samples:
    external_dataset_length = len(os.listdir('cars_test'))

    inds = list(range(external_dataset_length))
    shuffle(inds)

    for i in range(number_of_external_vehicle_samples):
      img_id = inds[i]
      img = cv.imread("cars_test/{img_id:05d}.jpg".format(img_id=img_id+1))

      random_sequence_id = choice(list(range(number_of_sequences)))
      sequences[random_sequence_id].append((img,1))
      number_of_positive_samples += 1

  # -------------------- Negative Train Set -------------------- #

  # Random patches constraints
  minimum_bbox_h, maximum_bbox_h = 0.1, 0.5
  minimum_bbox_w, maximum_bbox_w = 0.1, 0.5
  max_diff_width_height = 30

  negative_samples_set_length = int((1 / positive_negative_ratio) * number_of_positive_samples)
  number_of_negative_samples_per_sequence = int(negative_samples_set_length/number_of_sequences)

  for seq_id in range(number_of_sequences):

    bboxes = parse('dataset/000{seq_id}.txt'.format(seq_id=seq_id))
    # Generate negative samples
    for sample in range(number_of_negative_samples_per_sequence):

      frame_id = choice(list(bboxes.keys()))
      frame_bboxes = bboxes[frame_id]
      
      img = cv.imread('dataset/000{seq_id}/{frame_id:06d}.png'.format(seq_id=seq_id, frame_id=frame_id))
      img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

      # Generate valid dimensions
      while(True):
        random_y1 = randint(0,img.shape[0])
        random_y2 = randint(random_y1,img.shape[0])

        random_x1 = randint(0,img.shape[1])
        random_x2 = randint(random_x1,img.shape[1])

        # validate dimensions
        if random_y2 - random_y1 < minimum_bbox_h*img.shape[0] or random_y2 - random_y1 > maximum_bbox_h*img.shape[0] :
          continue
        if random_x2 - random_x1 < minimum_bbox_w*img.shape[1] or random_x2 - random_x1 > maximum_bbox_w*img.shape[1] :
          continue

        if abs((random_x2 - random_x1) - (random_y2 - random_y1)) > max_diff_width_height:
          continue
        break

      random_bbox = (random_x1, random_y1, random_x2, random_y2)
      random_area = area(random_bbox)

      # check intersection with vehicles
      is_vehicle = 0
      for (id, x1,y1,x2,y2) in frame_bboxes:
        vehicle_bbox = (x1,y1,x2,y2)
        vehicle_area = area(vehicle_bbox)

        intersection_bbox = intersection(vehicle_bbox, random_bbox)
        
        # no intersection
        if intersection_bbox is None:
          continue
        else:
          intersection_area = area(intersection_bbox)
          # check for minimal intersection
          if intersection_area/vehicle_area > min_intersection_ratio and intersection_area/random_area > min_intersection_ratio:       
            is_vehicle = 1
            break

      random_bbox_img = img[random_y1:random_y2,random_x1:random_x2]
      if is_vehicle == 0:
        sequences[seq_id].append((random_bbox_img,is_vehicle))

  return sequences


import numpy as np


# NEW VERSION
def build_dataset_from_sliding_window(min_intersection_ratio=0.8, number_of_positive_samples=500):
  '''
      Build a dataset from provided image sequences and other external datasets. The built dataset consists of a 
      dictionary. Each key corresponds to a sequence of images. The image sequences can be split and used for training.

      min_intersection_ratio: The minimum ratio of a random generated patch vs. a vehicle bbox to be considered a vehicle
      number_of_positive: indicates number of vehicle samples to add to dataset

      return sequences, a dictionary containing list of images in each sequence.
  '''

  # Build sequence dictionary
  number_of_sequences = 4
  sequences = dict()
  for i in range(number_of_sequences):
    sequences[i] = list()

  positive_samples_count = 0


  # -------------------- Positive Train Set -------------------- #

  for seq_id in range(number_of_sequences):
    visited_ids = list() # keep track of vehicle ids to avoid adding the same vehicle multiple times

    bboxes = parse('dataset/000{seq_id}.txt'.format(seq_id=seq_id))

    # Get positive samples from provided bounding boxes
    for frame_id, frame_bboxes in bboxes.items():
      img = cv.imread('dataset/000{seq_id}/{frame_id:06d}.png'.format(seq_id=seq_id, frame_id=frame_id))
      img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

      for bbox in frame_bboxes:
        id, x1, y1, x2, y2 = bbox.astype('int32')

        if id not in visited_ids:
          visited_ids.append(id)
          bbox_img = img[y1:y2,x1:x2]
          sequences[seq_id].append((bbox_img,1))
          positive_samples_count += 1

  # Add Car Images from Stanford University dataset
  external_dataset_length = len(os.listdir('cars_test'))

  inds = list(range(external_dataset_length))
  shuffle(inds)

  for i in len(range(positive_samples_count, number_of_positive_samples)):
    img_id = inds[i]
    img = cv.imread("cars_test/{img_id:05d}.jpg".format(img_id=img_id+1))

    random_sequence_id = choice(list(range(number_of_sequences)))
    sequences[random_sequence_id].append((img,1))

  # -------------------- Negative Train Set -------------------- #

  # Random patches constraints
  # minimum_bbox_h, maximum_bbox_h = 0.1, 0.5
  # minimum_bbox_w, maximum_bbox_w = 0.1, 0.5
  # max_diff_width_height = 30

  # negative_samples_set_length = int((1 / positive_negative_ratio) * number_of_positive_samples)
  # number_of_negative_samples_per_sequence = int(negative_samples_set_length/number_of_sequences)

  for seq_id in range(number_of_sequences):

    bboxes = parse('dataset/000{seq_id}.txt'.format(seq_id=seq_id))
    # Generate negative samples
    # for sample in range(number_of_negative_samples_per_sequence):

    frame_id = choice(list(bboxes.keys()))
    frame_bboxes = bboxes[frame_id]
    
    img = cv.imread('dataset/000{seq_id}/{frame_id:06d}.png'.format(seq_id=seq_id, frame_id=frame_id))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Generate valid dimensions
    # while(True):
    #   random_y1 = randint(0,img.shape[0])
    #   random_y2 = randint(random_y1,img.shape[0])

    #   random_x1 = randint(0,img.shape[1])
    #   random_x2 = randint(random_x1,img.shape[1])

    #   # validate dimensions
    #   if random_y2 - random_y1 < minimum_bbox_h*img.shape[0] or random_y2 - random_y1 > maximum_bbox_h*img.shape[0] :
    #     continue
    #   if random_x2 - random_x1 < minimum_bbox_w*img.shape[1] or random_x2 - random_x1 > maximum_bbox_w*img.shape[1] :
    #     continue

    #   if abs((random_x2 - random_x1) - (random_y2 - random_y1)) > max_diff_width_height:
    #     continue
    #   break

    # TODO apply sliding window
    sliding_y1, sliding_y2, sliding_x1, sliding_x2 = 0,0,0,0
    sliding_bbox = ()
    sliding_area = area(sliding_bbox)

    # check intersection with vehicles
    is_vehicle = 0
    for (id, x1,y1,x2,y2) in frame_bboxes:
      vehicle_bbox = (x1,y1,x2,y2)
      vehicle_area = area(vehicle_bbox)

      intersection_bbox = intersection(vehicle_bbox, sliding_bbox)
      
      # no intersection
      if intersection_bbox is None:
        continue
      else:
        intersection_area = area(intersection_bbox)
        # check for minimal intersection
        if intersection_area/vehicle_area > min_intersection_ratio and intersection_area/sliding_area > min_intersection_ratio:       
          is_vehicle = 1
          break

    random_bbox_img = img[sliding_y1:sliding_y2,sliding_x1:sliding_x2]
    if is_vehicle == 0:
      sequences[seq_id].append((random_bbox_img,is_vehicle))

  return sequences

def dataset_statistics(sequences, statistic_types, number_of_banks=5):
  '''
      Collect width, height, and aspect ratios statistics from image dataset.

      sequences: dictionary storing images per sequence. Keys are sequence ids and values are
                lists of images.
      
      statistics: list containing the statistics to compute. Can contain the following values: 'width', 'height',
                  'aspect_ratio', 'class_distribution'.
      
      return a dictionary of statistics. For height, width, and aspect ratio, the value is a min-max tuple.
                                         For class_distribution, the value is an array of counts per class.
  '''
  # Number of samples
  number_of_samples = 0
  for seq in sequences.values():
    number_of_samples += len(seq)

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

  for seq in sequences.values():
    for i, sample in enumerate(seq):
      if 'width' in statistic_types:
        widths[i] = sample[0].shape[1]

      if 'height' in statistic_types:
        heights[i] = sample[0].shape[0]

      if 'aspect_ratio' in statistic_types:
        aspect_ratios[i] = heights[i] / float(widths[i])
      
      if 'class_distribution' in statistic_types:
        class_count[sample[1]] += 1

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


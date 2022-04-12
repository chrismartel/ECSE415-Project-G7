from skimage.transform import resize
from skimage.feature import hog
import numpy as np
from random import shuffle
import time


def preprocess_images_v2(
    imgs,
    orientations,
    pixels_per_cell,
    cells_per_block,
    preprocessing_time=False,
    compute_spatial_features=False,
    hist_features=False,
):
    """
    Preprocess an image.

    img: numpy array of shape (N, H,W,C)
    orientations: int - number of orientation bins in hog feature computations
    pixels_per_cells: tuple (h,w) - number of pixels per cell in hog feature computation
    cells_per_block: tuple (h,w) - number of cells per block in hog feature computation
    preprocessing_time: indicates if preprocessing time must be computed and returned

    return (features, (optional)hog_img, (optional)time) - HoG features extracted from the images,
    the HoG images, and the preprocessing times in ms
    """
    N = imgs.shape[0]
    exec_times_ms = np.zeros(imgs.shape[0])

    features = None

    for i in range(imgs.shape[0]):
        img = imgs[i]
        start = time.time()

        # Min-max normalization
        min, max = 0, 255
        img = (img - min) / (max - min)

        # Extract HoG features
        feature = hog(
            img,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            feature_vector=True,
            multichannel=True,
            visualize=False,
        )

        if hist_features:
            if len(img.shape) < 3:
                img = img[:, :, np.newaxis]

            hist_vector = np.array([])
            for channel in range(img.shape[2]):
                channel_hist = np.histogram(
                    img[:, :, channel], bins=16, range=(0, 255)
                )[0]
                hist_vector = np.hstack((hist_vector, channel_hist))
            feature = np.hstack((feature, hist_vector))

        if compute_spatial_features:
            small_img = resize(img, (20, 20))
            spatial_features = small_img.flatten()
            feature = np.concatenate((feature, spatial_features), axis=-1)

        # inti features array
        if features is None:
            features = np.zeros((N, len(feature)))

        features[i] = feature
        end = time.time()
        exec_times_ms[i] = (end - start) * 1000

    # return package
    ret = list()
    ret.append(features)

    if preprocessing_time:
        ret.append(np.mean(exec_times_ms))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
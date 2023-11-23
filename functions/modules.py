import random
import string

import zipfile
import os
import cv2
import time
import shutil
import random
import inspect
import keras
import numpy as np
from PIL import Image
from tabulate import tabulate
from tqdm import tqdm
from scipy import ndimage
import mahotas as mh

from skimage import measure, filters, exposure
from matplotlib.ticker import MultipleLocator
from matplotlib import image as mpimg
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import keras.optimizers
import skimage.filters as filters
import skimage.measure as measure
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Conv2DTranspose, Dense, MaxPooling2D
from keras.metrics import MeanIoU
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras_unet.losses import jaccard_distance
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.models import custom_unet
from keras_unet.utils import plot_imgs, plot_segm_history
from sklearn.metrics import classification_report

import tifffile




import os
import numpy as np

from sklearn.metrics import jaccard_score, f1_score, confusion_matrix
from keras.models import load_model
import keras.optimizers
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance








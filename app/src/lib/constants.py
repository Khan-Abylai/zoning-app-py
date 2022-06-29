import os

import numpy as np

DETECTION_IMAGE_W = 512
DETECTION_IMAGE_H = 512
PIXEL_MAX_VALUE = 255

RECOGNIZER_IMAGE_W = 128
RECOGNIZER_IMAGE_H = 32

IMG_C = 3

DETECTION_IMG_CONFIGURATION = (2, 0, 1)
RECOGNIZER_IMG_CONFIGURATION = (0, 3, 1, 2)
SQUARE_LP_RATIO = 2.6
RECOGNIZER_THRESHOLD = 0.8

RECT_LP_H_CM = 0.12
SQUARE_LP_H_CM = 0.2
AVERAGE_LP_H_FROM_GROUND_CM = 0.35
PLATE_RECT = np.array([[0, 0], [0, 32], [127, 0], [127, 31]], dtype='float32')

PLATE_SQUARE = np.array([[0, 0], [0, 63], [63, 0], [63, 63]], dtype='float32')

ABS_BASE_FOLDER = '/home/user/parking_zoning/app/src'
DEBUG_FOLDER = os.path.join(ABS_BASE_FOLDER, 'debug')
MODELS_FOLDER = os.path.join(ABS_BASE_FOLDER, 'models')
STORAGE_FOLDER = os.path.join(ABS_BASE_FOLDER, 'storage')

import os
import sys
import cv2
import csv
import tqdm

from yolov3_core import *
"""
This program takes path to a directory with images as an argument
then returns a single csv with all the bounding boxes for those images

Modify directory with images in IMG_DIR
Then modify output in OUT_PATH
"""

# declare source directory and out path
if sys.argv[0] is not None:
    IMG_DIR = sys.argv[1]
else:
    IMG_DIR = "./data/test_data"

OUT_PATH = "./data/output"


## Declare settings for nn
settings = {'model_def': "cfg/yolov3-spp-1cls.cfg",
            'weights_path': "weights/last.pt",
            'class_path': "416_1_4_full/classes.names",
            'img_size': 608,
            'iou_thres': 0.6,
            'no_gpu': True,
            'conf_thres': 0.3,
            'batch_size': 6,
            'augment': None,
            'classes': None}


## used for padding images
def add_padding_to_square_img(img, cut_size):
    """
    Takes image as input and then returns padded square image
    based on the cut_size. Default should be 416
    """
    y_size, x_size = img.shape[:2]
    y_pad_amount = cut_size - y_size
    x_pad_amount = cut_size - x_size

    pad_img = np.pad(img, [(0,y_pad_amount), (0,x_pad_amount), (0,0)])

    return pad_img


class YoloToCSV(YoloModelLatest):
    def __init__():
        

import os
import sys

import cv2
import csv
import tqdm
import pandas as pd
import random
import numpy as np
import time
import fnmatch


def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
    im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
    return vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)


if __name__ == "__main__":
    # declare source directory and out path
    """
    IMG_DIR is path to the folder with the images
    OUT_PATH is the path to the csv file you would like the output to go to
    i.e './output/sample.csv'
    """
    VID_DIR = sys.argv[1]
    OUT_PATH = sys.argv[2]
    
    
    
    vid_list = fnmatch.filter(os.listdir(VID_DIR),"*.avi")
    vid_list = vid_list[:12]
    total_frame_count = 100000
    for vid_name in vid_list:
        videoPath = VID_DIR+vid_name
        print(videoPath)
        vid = cv2.VideoCapture(videoPath)

        print(vid_name)
        vid_frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        if vid_frame_count < total_frame_count:
            total_frame_count = vid_frame_count
    print(total_frame_count)
    
    for frameN in range(1,10):
        imgNum = 0
        images = np.empty(12), dtype=object)
        for vid_name in vid_list:
            videoPath = VID_DIR+vid_name
            vid = cv2.VideoCapture(videoPath)
            vid.set(1, frameN)
            ret, frame = vid.read()
            height, width, channels = frame.shape
            widthnu = int(width * (1/12))
            heightnu = int(height * (1/12))
            dsize = (widthnu, heightnu)
            framenu = cv2.resize(frame, dsize)
            images[imgNum] = frameNu
            imgNum +=1
       
       im_tile_resize = concat_tile_resize([ [images[0],images[1],images[2],images[3]],
                                             [images[4],images[5],images[6],images[7]],
                                             [images[8],images[9],images[10],images[11]] ])
                                      
     

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



if __name__ == "__main__":
    # declare source directory and out path
    """
    IMG_DIR is path to the folder with the images
    OUT_PATH is the path to the csv file you would like the output to go to
    i.e './output/sample.csv'
    """
    IMG_DIR = "/media/mlcomp/DrugAge/lifespan_machine_images"
    out_video_path= "/media/mlcomp/DrugAge/lifespan_machine_filter/ALM.avi"
    img_list = fnmatch.filter(os.listdir(IMG_DIR),"*.avi")
    frames = 1
    for im_name in img_list:
         print(frames)
         img_path = os.path.join(IMG_DIR, im_name)  
         if frames == 1:
             frame = cv2.imread(img_path)
             height, width, channels = frame.shape
             print(height, width)
             fourcc = cv2.VideoWriter_fourcc(*"MJPG")
             writer = cv2.VideoWriter(out_video_path, fourcc, 10, (width, height), True)
         frame = cv2.imread(img_path) 
         writer.write(img)

         frames +=1
    writer.release()     

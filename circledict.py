import numpy as np
import cv2
import os
import sys
import cv2
import csv
import tqdm
import pandas as pd
import random
import time
from matplotlib import pyplot as plt
from re import sub
import fnmatch




if __name__ == "__main__":
    # declare source directory and out path
    """
    IMG_DIR is path to the folder with the images
    OUT_PATH is the path to the csv file you would like the output to go to
    i.e './output/sample.csv'
    """

    VID_DIR = sys.argv[1]
    OUT_PATH = sys.argv[2]
    
    minDist = 500
    param1 = 30 #500
    param2 = 150 #200 #smaller value-> more false circles
    minRadius = 400
    maxRadius = 700 #10



    
    
    vid_list = fnmatch.filter(os.listdir(VID_DIR),"*.avi")
    print(vid_list)
    circlesheet = [] 
    for vid_name in vid_list:
        videoPath = VID_DIR+vid_name
        print(videoPath)
        vid = cv2.VideoCapture(videoPath)

        total_frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)

        while (1):
            ret, frame = vid.read()
            frame_count = vid.get(cv2.CAP_PROP_POS_FRAMES)
            if total_frame_count < 200:
                break
            if frame_count ==25:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
                print(circles)
                print(vid_name)

                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    #print(circles)
                    #p#rint(vid_name)
                    for i in circles[0,:]:
                        circlesheet.append([vid_name,frame_count, i[0], i[1], i[2]])
                        
                        
                break
            if frame_count >25:    
                break     
        pd.DataFrame(circlesheet).to_csv(OUT_PATH, mode='a', header=True, index=None)











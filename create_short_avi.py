import os
import sys
import cv2
import csv
import tqdm
import pandas as pd
import random


if __name__ == "__main__":
    # declare source directory and out path
    """
    IMG_DIR is path to the folder with the images
    OUT_PATH is the path to the csv file you would like the output to go to
    i.e './output/sample.csv'
    """
    VID_PATH = sys.argv[1]

    OUT_PATH = sys.argv[2]
    
    
    vid = cv2.VideoCapture(VID_PATH)
    total_frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    video_name = os.path.basename(VID_PATH).strip('.avi')
    out_video_path = f"{OUT_PATH}/{os.path.basename(VID_PATH).strip('.avi')}_short.avi"

    
      
    while (1):
        ret, frame = vid.read()
        frame_count = vid.get(cv2.CAP_PROP_POS_FRAMES)
        if frame_count == 1:
            height, width, channels = frame.shape
            print(height, width)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(out_video_path, fourcc, 10, (width, height), True)
            
        check = random.randint(1,50)
        if check == 1:
            print(frame_count)
            writer.write(frame)
        if frame_count == total_frame_count:
            break
    writer.release()        

import os
import sys
import cv2
import csv
import tqdm
import pandas as pd
import random
import numpy as np
import time
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt
import fnmatch




def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou





def saveworm(df,frame,frame_count,OUT_PATH,vid_name): 
    img_base = frame
    frame_count = int(frame_count)
    for output in df:
        frameN, x1, y1, x2, y2, *_ = output
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        buffer = 15
        croppedLarge = frame[(y1-buffer):(y2+buffer),(x1-buffer):(x2+buffer)]
        out_image_path = f"{OUT_PATH}/{vid_name}_{frame_count}__x1y1x2y2_{x1}_{y1}_{x2}_{y2}.png"
        cv2.imwrite(out_image_path, croppedLarge)  
    #return(img_base)

     
    
    
       






if __name__ == "__main__":
    # declare source directory and out path
    """
    IMG_DIR is path to the folder with the images
    OUT_PATH is the path to the csv file you would like the output to go to
    i.e './output/sample.csv'
    """
    CSV_DIR = sys.argv[1]
    VID_DIR = sys.argv[2]
    OUT_PATH = sys.argv[3]
    
    
    
    vid_list = fnmatch.filter(os.listdir(VID_DIR),"*.avi")
 
    for vid_name in vid_list:
        videoPath = VID_DIR+vid_name
        print(videoPath)
        vid = cv2.VideoCapture(videoPath)
        csv_path = f"{CSV_DIR}/{os.path.basename(vid_name).strip('.avi')}.csv"
        print(csv_path)
        print(vid_name)
        total_frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        #out_video_path = f"{OUT_PATH}/{os.path.basename(vid_name).strip('.avi')}_death.avi"
        #csv_path = os.path.join(SORT_DIR, csv_name)
        df = pd.read_csv(csv_path,names=('frame', 'x1', 'y1', 'w','h','label'))
        df['x2']=df[['x1','w']].sum(axis=1)
        df['y2']=df[['y1','h']].sum(axis=1)
        df = df[['frame','x1', 'y1', 'x2', 'y2']]
        outputs = df

    
    
        while (1):
            ret, frame = vid.read()
            frame_count = vid.get(cv2.CAP_PROP_POS_FRAMES)
            #print(frame_count)
            check = random.randint(1,50)

            if check == 1: 
                filtval = outputs['frame'] == frame_count
                csv_outputs = np.asarray(outputs[filtval])[:,:]
                print(frame_count)
                if csv_outputs is not None:
                    print(csv_outputs)
                    saveworm(csv_outputs,frame,frame_count,OUT_PATH,vid_name)

            if frame_count == total_frame_count:
                break
    
    
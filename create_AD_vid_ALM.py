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
#from skimage.morphology import skeletonize, medial_axis
#from matplotlib import pyplot as plt
#from re import sub


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



def analyzeSORT(df,threshold):
    #threshold = 45
    vc = df.label.value_counts()
    test = vc[vc > threshold].index.tolist()
    csv_outputs = []
    deadboxes = []
    for ID in test:        
        filtval = df['label'] ==ID
        interim = df[filtval]
        interimD = []
        interim2 = np.asarray(interim)
        fill = 0
        deadcount = 0
        deathspots = []
    for ID in test:        
        filtval = df['label'] ==ID
        interim = df[filtval]
        interimD = []
        interim2 = np.asarray(interim)
        fill = 0
        deadcount = 0
        for row in interim2:
            frameNA, x1A, y1A, x2A, y2A, labelA, deltaA, catagoryA, *_ = row
            if fill > 1:
                boxA = [x1A, y1A, x2A, y2A]
                boxB = [x1B, y1B, x2B, y2B]
                deltaA = bb_intersection_over_union(boxA, boxB) 
                #print(deltaA)
                if deltaA > 0.95:
                    deadcount += 1
                    #print(deadcount)
                #print(fill)
                if deadcount > 7:
                    catagoryA = 'dead'
                   # print(deadcount)
                if deadcount == 8:
                    if deadboxes == []:
                        deathspots.append([frameNA, x1A, y1A, x2A, y2A])     
                        deadboxes.append([x1A, y1A, x2A, y2A])  
                        csv_outputs.append((frameNA/24)+7)
                    else:
                        notunique = 0
                        for box in deadboxes:
                            #print(box)
                            x1D, y1D, x2D, y2D, *_ = box
                            boxD = [x1D, y1D, x2D, y2D]
                            deltaD = bb_intersection_over_union(boxA, boxD)  
                            if deltaD > 0.3:
                                notunique = 1
                        if notunique == 0:
                            deathspots.append([frameNA, x1A, y1A, x2A, y2A])  
                            deadboxes.append([x1A, y1A, x2A, y2A])               
                            csv_outputs.append((frameNA/24)+7)

                    #print(deathtime)
                    #print(frameNA)
                #newRow = [frameNA, x1A, y1A, x2A, y2A, labelA, deltaA, catagoryA]
                #interimD.append(newRow)
            frameNB, x1B, y1B, x2B, y2B,labelB, deltaB, catagoryB, *_ = row
            fill +=1
            if deadcount == 8:
                break

    csv_outputs = pd.DataFrame(csv_outputs, columns = ['#desc'])
    csv_outputs['neural'] = '1'
    return(deathspots)
















if __name__ == "__main__":
    # declare source directory and out path
    """
    IMG_DIR is path to the folder with the images
    OUT_PATH is the path to the csv file you would like the output to go to
    i.e './output/sample.csv'
    """
    SORT_DIR = sys.argv[1]
    VID_DIR = sys.argv[2]
    OUT_PATH = sys.argv[3]
    
    
    
    vid_list = fnmatch.filter(os.listdir(VID_DIR),"*.avi")
 
    for vid_name in vid_list:
        videoPath = VID_DIR+vid_name
        print(videoPath)
        vid = cv2.VideoCapture(videoPath)
        csv_path = f"{SORT_DIR}/{os.path.basename(vid_name).strip('_yolo.avi')}.csv"
        print(csv_path)
        print(vid_name)
        total_frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        out_video_path = f"{OUT_PATH}/{os.path.basename(vid_name).strip('.avi')}_death.avi"
        #csv_path = os.path.join(SORT_DIR, csv_name)
        df = pd.read_csv(csv_path,names=('frame', 'x1', 'y1', 'x2', 'y2','label','delta'))
        df['catagory'] = 'alive'
        
        while (1):
            ret, frame = vid.read()
            frame_count = vid.get(cv2.CAP_PROP_POS_FRAMES)
            #print(frame_count)
            if frame_count == 1:
                
                height, width, channels = frame.shape
                #print(height, width)
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(out_video_path, fourcc, 10, (width, height), True)
            deathspots = analyzeSORT(df,threshold = 24)

            for death in deathspots:
                #print(death)
                frameNA, x1, y1, x2, y2, *_ = death  
                frameNA = int(frameNA)
                if frame_count > frameNA:
                    x1 = int(x1)
                    x2 = int(x2)
                    y1 = int(y1)
                    y2 = int(y2)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)

            writer.write(frame)
            if frame_count == total_frame_count:
                break
        writer.release() 
        print(out_video_path)

        
    
        
import os
import sys
import cv2
import csv
import tqdm
import pandas as pd
from sort.sort import *
import random
import numpy as np
import time
from skimage.morphology import skeletonize, medial_axis
from matplotlib import pyplot as plt
from re import sub


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



def analyzeSORT(df,threshold,slow_move,delta_overlap,start_age,framerate):
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
        for row in interim2:
            frameNA, x1A, y1A, x2A, y2A, labelA, deltaA, catagoryA, *_ = row
            if fill > 1:
                boxA = [x1A, y1A, x2A, y2A]
                boxB = [x1B, y1B, x2B, y2B]
                deltaA = bb_intersection_over_union(boxA, boxB) 
                #print(deltaA)
                if deltaA > delta_overlap:
                    deadcount += 1
                    #print(deadcount)
                #print(fill)
                if deadcount > slow_move:
                    catagoryA = 'dead'
                   # print(deadcount)
                if deadcount == slow_move+1:
                    if deadboxes == []:
                        deathspots.append([frameNA, x1A, y1A, x2A, y2A])     
                        deadboxes.append([x1A, y1A, x2A, y2A])  
                        csv_outputs.append((frameNA/framerate+start_age))
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
                            csv_outputs.append((frameNA/framerate+start_age))

                    #print(deathtime)
                    #print(frameNA)
                #newRow = [frameNA, x1A, y1A, x2A, y2A, labelA, deltaA, catagoryA]
                #interimD.append(newRow)
            frameNB, x1B, y1B, x2B, y2B,labelB, deltaB, catagoryB, *_ = row
            fill +=1
            if deadcount == slow_move+1:
                break

    csv_outputs = pd.DataFrame(csv_outputs, columns = ['#desc'])
    csv_outputs['neural'] = '1'
    return(csv_outputs)



if __name__ == "__main__":
    # declare source directory and out path
    """
    IMG_DIR is path to the folder with the images
    OUT_PATH is the path to the csv file you would like the output to go to
    i.e './output/sample.csv'
    """
    
    
    CSV_FOLD_PATH = sys.argv[1]
    OUT_FOLD_PATH = sys.argv[2]
    
    #ARGS
    threshold = 30
    max_age_SORT=10, 
    min_hits_SORT=3,
    iou_threshold_SORT=0.3
    start_age = 0
    framerate = 144
    slow_move = 5
    delta_overlap = 0.8
    #END ARGS
    
    csv_list = os.listdir(CSV_FOLD_PATH)
    csv_list = list(filter(lambda f: f.endswith('.csv'), csv_list))

    print(csv_list)
    for csv_name in csv_list:
        print("sorting "+csv_name)


        csv_PATH = os.path.join(CSV_FOLD_PATH, csv_name)
        OUT_PATH = os.path.join(OUT_FOLD_PATH, csv_name)


        video_name = os.path.basename(csv_PATH).strip('.csv')

        #load in csv and convert to xyxy
        df = pd.read_csv(csv_PATH,names=('frame', 'x1', 'y1', 'w','h','label'))
        df['x2']=df[['x1','w']].sum(axis=1)
        df['y2']=df[['y1','h']].sum(axis=1)
        df = df[['frame','x1', 'y1', 'x2', 'y2']]
        unique = df["frame"].unique()

        #initialize tracker
        mot_tracker1 = Sort(max_age=10, min_hits=3, iou_threshold=0.3) 
        csv_outputs = []
        for x in unique:
            frame = int(x)
            #if frame > (144*4):
            #    break
            #print(frame)
            filtval = df['frame'] == x
            boxes_xyxy = np.asarray(df[filtval])[:,1:5]

            track_bbs_ids = mot_tracker1.update(boxes_xyxy)
            for output in track_bbs_ids:
                x1, y1, x2, y2, label, *_ = output
                csv_outputs.append([x.tolist(), x1.tolist(), y1.tolist(), x2.tolist(),y2.tolist(),label.tolist()])
        print("finished sorting")
        #csv_outputs = csv_outputs[['frame', 'x1', 'y1', 'x2', 'y2','label']]
        csv_outputs = pd.DataFrame(csv_outputs)
       
        csv_outputs.columns = ['frame', 'x1', 'y1', 'x2', 'y2','label']
        csv_outputs['delta'] = 0
        csv_outputs['catagory'] = 'alive'
        
        outputs = analyzeSORT(csv_outputs,threshold,slow_move,delta_overlap,start_age,framerate)
        outputs.loc[0] = ['#expID',csv_name]

        pd.DataFrame(outputs).to_csv(OUT_PATH, mode='w', header=True, index=None)
        
        
        
        

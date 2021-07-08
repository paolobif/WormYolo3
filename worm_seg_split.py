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
    deadIDs = []
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
                if deadcount > 15:
                    catagoryA = 'dead'
                   # print(deadcount)
                if deadcount == 16:
                    if deadboxes == []:
                        deathspots.append([frameNA, x1A, y1A, x2A, y2A])     
                        deadboxes.append([x1A, y1A, x2A, y2A])  
                        deadIDs.append(labelA)
                        csv_outputs.append((frameNA/72)+2)
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
                            deadIDs.append(labelA)

                            csv_outputs.append((frameNA/72)+2)

                    #print(deathtime)
                    #print(frameNA)
                #newRow = [frameNA, x1A, y1A, x2A, y2A, labelA, deltaA, catagoryA]
                #interimD.append(newRow)
            frameNB, x1B, y1B, x2B, y2B,labelB, deltaB, catagoryB, *_ = row
            fill +=1
            if deadcount == 16:
                break

    csv_outputs = pd.DataFrame(csv_outputs, columns = ['#desc'])
    csv_outputs['neural'] = '1'
    return(deadIDs)




def saveworm(df,frame,frame_count,OUT_PATH,vid_name): 
    img_base = frame
    frame_count = int(frame_count)
    for output in df:
        frameN, x1, y1, x2, y2, label, delta ,alive, *_ = output
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        buffer = 15
        croppedLarge = frame[(y1-buffer):(y2+buffer),(x1-buffer):(x2+buffer)]
        out_image_path = f"{OUT_PATH}/{label}/{vid_name}_{frame_count}_{label}_x1y1x2y2_{x1}_{y1}_{x2}_{y2}.png"
        cv2.imwrite(out_image_path, croppedLarge)  
    #return(img_base)

     
    
    
       






if __name__ == "__main__":
    # declare source directory and out path
    """
    IMG_DIR is path to the folder with the images
    OUT_PATH is the path to the csv file you would like the output to go to
    i.e './output/sample.csv'
    """
    VID_PATH = sys.argv[1]
    CSV_SORT_PATH = sys.argv[2]
    OUT_PATH = sys.argv[3]
    
    vid = cv2.VideoCapture(VID_PATH)
    total_frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    video_name = os.path.basename(VID_PATH).strip('.avi')
    
    
          
    ####SORT####
    df = pd.read_csv(CSV_SORT_PATH,names=('frame', 'x1', 'y1', 'x2', 'y2','label','delta'))
    df['catagory'] = 'alive'
    outputs = df
    print(outputs)
    
    #df = pd.read_csv(CSV_SORT_PATH,names=('frame', 'x1', 'y1', 'w','h','label'))
    #df['x2']=df[['x1','w']].sum(axis=1)
    #df['y2']=df[['y1','h']].sum(axis=1)
    #df = df[['frame','x1', 'y1', 'x2', 'y2']]
    #df['label'] = 'alive'
    deadIDs = analyzeSORT(df,threshold = 75)
    print(deadIDs)
    for death in deadIDs:
        out_image_dir= f"{OUT_PATH}/{death}"
        os.mkdir(out_image_dir)
        
    #filtval = outputs['label'] in deadIDs
    #csv_outputs = np.asarray(outputs[filtval])[:,:]
    
    boolean_series = df.label.isin(deadIDs)
    outputs = df[boolean_series]
    print(outputs)
    
    while (1):
        ret, frame = vid.read()
        frame_count = vid.get(cv2.CAP_PROP_POS_FRAMES)
        #print(frame_count)




        filtval = outputs['frame'] == frame_count
        #print(filtval)
        csv_outputs = np.asarray(outputs[filtval])[:,:]
        #print(csv_outputs)
        if csv_outputs is not None: 
            saveworm(csv_outputs,frame,frame_count,OUT_PATH,video_name)

        if frame_count == total_frame_count:
            break
    
    
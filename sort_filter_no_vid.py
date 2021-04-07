import os
import sys
import cv2
import csv
import tqdm
import pandas as pd
import random
import numpy as np


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

def reformat_csv(df, threshold):
    vc = df.label.value_counts()
    test = vc[vc > threshold].index.tolist()
    csv_outputs = []
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
                if deltaA > 0.95:
                    deadcount += 1
            if deadcount > 10:
                catagoryA = 'dead'
                if deadcount == 11:
                    deathtime = frameNA
                #print(frameNA)
            newRow = [frameNA, x1A, y1A, x2A, y2A, labelA, deltaA, catagoryA]
            interimD.append(newRow)
            frameNB, x1B, y1B, x2B, y2B,labelB, deltaB, catagoryB, *_ = row
            fill +=1
        interimD = pd.DataFrame(frameNA, columns = ['deathframe'])
        csv_outputs.append(interimD)
    large_df = pd.concat(csv_outputs, ignore_index=True)        
    #df = pd.DataFrame(csv_outputs, columns = ['frame', 'x1', 'y1', 'x2', 'y2','label','delta','catagory'])
    return(large_df)


def draw_on_im(frame, outputs, out_path, writer, text=None):
        """Takes img, then coordinates for bounding box, and optional text as arg"""
        img = frame
        for output in outputs:
            #output = [int(n) for n in output]
            #frameN, x, y, text, *_ = output
            frameN, x1, y1, x2, y2, label, delta, catagory, *_ = output
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            # Draw rectangles
            if catagory == 'alive':
                #print(output)
                cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,0), 2)
            else:
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,255), 2)

            
            # Draw rectangles
            #if text is not None: 
             #   text = str(text)
             #   cv2.putText(img, text, (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,102,102), 2)
        ## write image to path
        #cv2.imwrite(out_path, img)
        writer.write(img)



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
    out_csv_path = f"{OUT_PATH}/{os.path.basename(VID_PATH).strip('.avi')}_SORT_deathtime.csv"

    df = pd.read_csv(CSV_SORT_PATH,names=('frame', 'x1', 'y1', 'x2', 'y2','label','delta'))
    df['catagory'] = 'alive'    #df['X']=df[['x1','x2']].mean(axis=1)
    #df['Y']=df[['y1','y2']].mean(axis=1)
    #df = df[['frame', 'X', 'Y','label']]
    unique = df["frame"].unique()
    threshold = 25
    df = reformat_csv(df, threshold)
   
    pd.DataFrame(df).to_csv(out_csv_path, mode='a', header=False, index=None)
    print(video_name)
    print("done")        
    #writer.release()        
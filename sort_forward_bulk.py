import os
import sys
import cv2
import csv
import tqdm
import pandas as pd
from yolov3_core import *
from sort.sort import *







if __name__ == "__main__":
    # declare source directory and out path
    """
    IMG_DIR is path to the folder with the images
    OUT_PATH is the path to the csv file you would like the output to go to
    i.e './output/sample.csv'
    """
    
    
    CSV_FOLD_PATH = sys.argv[1]
    OUT_FOLD_PATH = sys.argv[2]

    
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
        mot_tracker1 = Sort() 
        csv_outputs = []
        for x in unique:
            frame = int(x)
            #print(frame)
            filtval = df['frame'] == x
            boxes_xyxy = np.asarray(df[filtval])[:,1:5]

            track_bbs_ids = mot_tracker1.update(boxes_xyxy)
            for output in track_bbs_ids:
                x1, y1, x2, y2, wrmid, *_ = output
                csv_outputs.append([x.tolist(), x1.tolist(), y1.tolist(), x2.tolist(),y2.tolist(),wrmid.tolist()])
        print(video_name)
        print("done")
        pd.DataFrame(csv_outputs).to_csv(OUT_PATH, mode='a', header=False, index=None)
        
        

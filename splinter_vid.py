import os
import sys
import cv2
import csv
import tqdm
import pandas as pd
import random
import numpy as np


if __name__ == "__main__":
    # declare source directory and out path
    """
    IMG_DIR is path to the folder with the images
    OUT_PATH is the path to the csv file you would like the output to go to
    i.e './output/sample.csv'
    """
    VID_PATH = sys.argv[1]
    CSV_PATH = sys.argv[2]
    OUT_PATH = sys.argv[3]
    OUT_PATHcsv = sys.argv[4]
    
    vid = cv2.VideoCapture(VID_PATH)
    total_frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    video_name = os.path.basename(VID_PATH).strip('.avi')
    
        #load in csv and convert to xyxy
    df = pd.read_csv(CSV_PATH,names=('frame', 'x1', 'y1', 'w','h','label'))
    df['x2']=df[['x1','w']].sum(axis=1)
    df['y2']=df[['y1','h']].sum(axis=1)
    df = df[['frame','label', 'x1', 'y1', 'w', 'h']]
    unique = df["frame"].unique()
    
      
    while (1):
        ret, frame = vid.read()
        frame_count = vid.get(cv2.CAP_PROP_POS_FRAMES)
        img_out_path =  f"{os.path.join(OUT_PATH, video_name)}_{int(frame_count)}.png"
        check = random.randint(1,100)
        if check == 1:
            
            cv2.imwrite(img_out_path, frame)
            frame_cur = int(frame_count)
            print(frame_cur)
            filtval = df['frame'] == str(frame_cur)
            csv_outputs = np.asarray(df[filtval])[:,:]
            out_df = pd.DataFrame(csv_outputs)
            # change header to datacells for R-shiny processing
            out_df = out_df.set_axis( ['dataCells1','dataCells2','dataCells3','dataCells4','dataCells5','class'] , axis=1)
            csv_out_path = f"{os.path.join(OUT_PATHcsv, video_name)}_{int(frame_count)}_NN.csv"
            out_df.to_csv(csv_out_path, mode='w', header=True, index=None)

            
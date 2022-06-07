import os
import numpy as np
import pandas as pd
from tod import CSV_Reader, WormViewer
import cv2
import numpy as np
import time
from tqdm import tqdm

from utils import *

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (30,6)

%load_ext autoreload
%autoreload 2


if __name__ == "__main__":

    CSV_FOLD_PATH = sys.argv[1]  # folder of YOLO outputs
    MAN_FOLD_PATH = sys.argv[2]  # folder of manual calls
    OUT_FOLD_PATH = sys.argv[3]  # output folder

    
    
    csv_list = os.listdir(MAN_FOLD_PATH)
    csv_list = list(filter(lambda f: f.endswith('.csv'), csv_list))

    print(csv_list)
    csvindex = 0
    # loop through list of CSVs
    for csv_name in tqdm(csv_list):
        
        csv_path = f"{CSV_FOLD_PATH}/{os.path.basename(csv_name).strip('_wormlist.csv')}.csv"
        man_PATH = os.path.join(MAN_FOLD_PATH, csv_name)
        OUT_PATH = f"{OUT_FOLD_PATH}/{os.path.basename(csv_name).strip('_wormlist.csv')}.csv"
        

        exp = CSV_Reader(csv_path, vid_path)
        interval = 25  # Frame interval for getting frame values


        counts = []

        for i in tqdm(range(0, exp.frame_count, interval)):
            interval_bbs = []
            for n in range(i, i + interval):
                _, bbs = exp.get_worms_from_frame(n)
                interval_bbs.append(bbs)
            ls = np.concatenate(interval_bbs, axis=0)
            counts.append(len(ls) / interval)


        plt.plot(counts)

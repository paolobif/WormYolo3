import deadscriminate
import cv2
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np

DEFAULT_SORT = lambda coord: np.sqrt(coord[0]**2+coord[1]**2)
DEFAULT_CLOSE = lambda coord1, coord2: np.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)
DEFAULT_ACCEPT = 10

frame_offset = 20

def overlap(box1:tuple, box2:tuple) -> float:
    # box is x1, y1, x2, y2

    left_x = max(box1[0],box2[0])
    right_x = min(box1[2], box2[2])
    bottom_y = max(box1[1], box2[1])
    top_y = min(box1[3],box2[3])
    area = max(0, (top_y-bottom_y))*max(0,(right_x-left_x))
    calc_area = lambda box:(box[2]-box[0])*(box[3]-box[1])
    max_area = min(calc_area(box1),calc_area(box2))
    return area / max_area

def takeFirstFrameBBs(csv_data, frame_offset):
  relevant_data = csv_data.where(csv_data["frame"] < frame_offset).dropna()
  cur_list = []
  past_list = []


  for index, value in relevant_data.iterrows():
    x1 = value["x1"];x2 = value["x2"];y1 = value["y1"];y2 = value["y2"]
    bb = (x1,y1,x2,y2)
    add_to_list = True
    for pre_bb in past_list:
      if overlap(bb,pre_bb) > 0:
        add_to_list = False
    past_list.append(bb)
    if add_to_list:
      cur_list.append(bb)
  return cur_list


vid_path = "C:/Users/cdkte/Downloads/daily_monitor_all_neural/reduced-frame-healthspan"
csv_path = "C:/Users/cdkte/Downloads/daily_monitor_all_neural/sorted-by-id"


pre_vid_path = os.path.join(vid_path,"3225_day14.avi")
cur_vid_path = os.path.join(vid_path,"3225_day15.avi")

pre_data_path = os.path.join(csv_path,"3225_day14.csv")
cur_data_path = os.path.join(csv_path, "3225_day15.csv")

pre_vid = cv2.VideoCapture(pre_vid_path)
cur_vid = cv2.VideoCapture(cur_vid_path)

prev_data = pd.read_csv(pre_data_path,names=["frame", "x1", "y1", "x2", "y2","wormID"])
cur_data = pd.read_csv(cur_data_path, names=["frame", "x1", "y1", "x2", "y2","wormID"])

vid_length = pre_vid.get(cv2.CAP_PROP_FRAME_COUNT)

frame_offset = 20

pre_list = takeFirstFrameBBs(prev_data, frame_offset)

cur_list = takeFirstFrameBBs(cur_data, frame_offset)


ret, frame = cur_vid.read()




for i,cur_worm in enumerate(cur_list):
  fig, ax = plt.subplots(3,len(pre_list))

  x1, y1, x2, y2 =cur_worm
  x1 = int(x1)-10; x2 = int(x2)+10; y1 = int(y1)-10; y2 = int(y2)+10
  xes = [x1,x1,x2,x2,  x1]
  yes = [y1,y2,y2,y1,  y1]
  #print("\n\n\n\n")
  #"""
  cur_vid.set(cv2.CAP_PROP_POS_FRAMES,1)
  ret, frame = cur_vid.read()
  cur_worm_bb = frame[y1:y2,x1:x2]

  for j, pre_worm in enumerate(pre_list):
    x1, y1, x2, y2 = pre_worm
    x1 = int(x1)-10; x2 = int(x2)+10; y1 = int(y1)-10; y2 = int(y2)+10
    xes = [x1,x1,x2,x2,  x1]
    yes = [y1,y2,y2,y1,  y1]
    #print("\n\n\n\n")
    #"""
    pre_vid.set(cv2.CAP_PROP_POS_FRAMES,1)
    ret, frame = pre_vid.read()
    pre_worm_bb = frame[y1:y2,x1:x2]

    cur_val = deadscriminate.predict_single(pre_worm_bb,cur_worm_bb)

    ax[0,j].imshow(cur_worm_bb)
    ax[0,j].set_xticks([])
    ax[0,j].axes.get_yaxis().set_visible(False)
    ax[1,j].imshow(pre_worm_bb)
    ax[1,j].axes.get_xaxis().set_visible(False)
    ax[1,j].axes.get_yaxis().set_visible(False)
    ax[2,j].imshow(np.full((30,30),cur_val,dtype=np.float32),cmap = "gray", vmin=0, vmax=1)
    ax[2,j].axes.get_xaxis().set_visible(False)
    ax[2,j].axes.get_yaxis().set_visible(False)
  plt.show()
  #pass)

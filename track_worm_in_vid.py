from turtle import left
import tensorflow_sdf as tsdf
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import blot_circle_crop as bcc
import numpy as np

def triple_stack(matrix):
  return np.dstack([matrix,matrix,matrix])

def merge_frames(rgb_frame,grayscale_frame):
  #grayscale_rgb = np.mean(rgb_frame,2)
  int_gs = (grayscale_frame*255).astype(np.uint8)
  rgb_gs = triple_stack(int_gs)
  combo = np.hstack([rgb_frame,rgb_gs])
  #plt.imshow(combo)
  #plt.show()
  return combo

avi = "C:/Users/cdkte/Downloads/641_day6_simple.avi"
csv = "C:/Users/cdkte/Downloads/641_day6_simple_sort.csv"

csv = pd.read_csv(csv, usecols=[0, 1, 2, 3, 4, 5], names=["frame", "wormID", "x1", "y1", "x2", "y2"])
cap:cv2.VideoCapture = cv2.VideoCapture(avi)

offset = 50

select_worm_frame = 211

frame_worms = csv.where(csv["frame"]==select_worm_frame)
frame_worms = frame_worms.dropna()

first_worm = frame_worms.iloc(0)[0]
wormID = first_worm["wormID"]

print(wormID)

cur_x1 = int(first_worm["x1"])-10
cur_y1 = int(first_worm["y1"])-10
cur_x2 = int(first_worm["x2"])+10
cur_y2 = int(first_worm["y2"])+10

#frame_width = (cur_x2-cur_x1+offset*2)*2
#frame_height = cur_y2-cur_y1+offset*2

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*2)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_size = (frame_width,frame_height)
print(frame_size)
fps = cap.get(cv2.CAP_PROP_FPS)

output = cv2.VideoWriter("C:/Users/cdkte/Downloads/641_one_worm.avi", cv2.VideoWriter_fourcc('M','J','P','G'), fps, frame_size)

cap.set(cv2.CAP_PROP_POS_FRAMES,select_worm_frame)
ret, cur_frame = cap.read()
cur_frame_indx = select_worm_frame
while (ret):

  frame_worms = csv.where(csv["frame"]==cur_frame_indx)
  frame_worms = frame_worms.where(frame_worms["wormID"]==wormID)
  frame_worms = frame_worms.dropna()
  try:
    first_worm = frame_worms.iloc(0)[0]
  except:
    cur_frame_indx += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES,cur_frame_indx)
    ret, cur_frame = cap.read()
    continue

  cur_x1 = int(first_worm["x1"])-10
  cur_y1 = int(first_worm["y1"])-10
  cur_x2 = int(first_worm["x2"])+10
  cur_y2 = int(first_worm["y2"])+10


  worm_box = cur_frame[cur_y1:cur_y2,cur_x1:cur_x2]
  print(worm_box.dtype)
  anno_box = tsdf.generate_single_sdf(worm_box)
  #plt.imshow(anno_box)
  #plt.show()
  anno_box = bcc.dropSmall(anno_box)

  total_off = frame_width/2 - (cur_x2-cur_x1)
  left_off = int(np.floor(total_off/2))
  right_off = int(np.ceil(total_off/2))

  total_off = frame_height - (cur_y2-cur_y1)
  bottom_off = int(np.floor(total_off/2))
  top_off = int(np.ceil(total_off/2))
  #print(top_off)

  newer_box = cur_frame[cur_y1-bottom_off:cur_y2+top_off,cur_x1-left_off:cur_x2+right_off]
  new_anno_box = np.zeros((newer_box.shape[0],newer_box.shape[1]))
  #print(newer_box.shape,new_anno_box.shape,offset,cur_x1,cur_x2,new_anno_box[offset:offset+cur_y2-cur_y1,offset:offset+cur_x2-cur_x1].shape,cur_x2-cur_x1,anno_box.shape)

  #new_anno_box[offset:offset+cur_y2-cur_y1,offset:offset+cur_x2-cur_x1] = anno_box
  #both_boxes = merge_frames(newer_box,new_anno_box)

  new_anno_box = np.zeros((cur_frame.shape[0],cur_frame.shape[1]))
  new_anno_box[cur_y1:cur_y2,cur_x1:cur_x2] = anno_box
  both_boxes = merge_frames(cur_frame,new_anno_box)

  #print(both_boxes.shape,frame_size)



  #plt.imshow(anno_box)
  #plt.show()

  cv2.imshow("frame",both_boxes)
  #print("WRote")
  output.write(both_boxes)

  cur_frame_indx += 1

  cap.set(cv2.CAP_PROP_POS_FRAMES,cur_frame_indx)
  ret, cur_frame = cap.read()

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
#print("Release")
output.release()



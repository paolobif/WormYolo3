import cv2
import pandas as pd
import numpy as np
import os
import segmentation as seg


cur_frame = 0

vid_path = "C:/Users/cdkte/Downloads/320(1).avi"

csv_path = "C:/Users/cdkte/Downloads/320.csv"
out_vid = "C:/Users/cdkte/Downloads/320_anno.avi"


folder_path = "/media/mlcomp/DrugAge/gui_test/320/orig"

anno_folder_path = "/media/mlcomp/DrugAge/gui_test/320/anno"


"""
cap = cv2.VideoCapture(vid_path)
csv = pd.read_csv(csv_path, usecols=[0, 1, 2, 3, 4], names=["frame", "x1", "y1", "width", "height"])

ret, frame = cap.read()


while ret:
  copy_frame = np.copy(frame)
  cur_frame+=1

  #print(ret,frame)
  if cur_frame % 100 == 0:
    print(cur_frame)

  cur_worms = csv.where(csv["frame"]==cur_frame)
  cur_worms = cur_worms.dropna()

  for index, row in cur_worms.iterrows():
    x1=int(row["x1"])
    x2=x1+int(row["width"])
    y1=int(row["y1"])
    y2=y1+int(row["height"])
    worm_box = copy_frame[y1:y2,x1:x2,:]

    file_name = [str(row["x1"]),str(row["width"]),str(row["y1"]),str(row["height"]),str(cur_frame),".png"]
    file_name = "_".join(file_name)
    file_path = os.path.join(folder_path,file_name)

    cv2.imwrite(file_path,worm_box)


    #cv2.rectangle(frame, (x1,y1), (x2,y2), color = (0,0,255), thickness =2)


  ret, frame = cap.read()


"""
print("Test")
seg.annotateFolder(folder_path,anno_folder_path)

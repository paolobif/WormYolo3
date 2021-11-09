import os
import numpy as np
import cv2
import csv
from PIL import Image
import yolo_to_pixellib as ytp
import convert_image as ci
import skeleton_cleaner as sc
import time as t
import sys

"""
Handles unsorted worm videos. The lack of ids makes this more difficult, as the worms cannot be seperated into different folders.
"""


# The padding for worm images
PADDING = 10

# Number of frames to move at a time (I recommend reducing min_frames when you increase this)
# 1 skips no frames, 2 takes every second frame
SKIP_FRAMES = 2

# Number of folders to create
FOLDER_NUM = 50 * SKIP_FRAMES

def createImagesUnsorted(vid_path,csv_path,out_path):
  """
  Creates folders of images sorted by worm

  vid_path: The path to the raw video
  csv_path: The path to the csv file that matches the video with the sorted worms
  out_path: The folder where the results should be stored
  """

  if not os.path.exists(out_path):
    os.mkdir(out_path)
  np.set_printoptions(threshold=np.inf)
  vid = cv2.VideoCapture(vid_path)
  video_id = os.path.basename(vid_path).strip('.avi').split("_")[0]
  total_frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
  all_frames = {}

  for i in range(0,int(total_frame_count)+1,SKIP_FRAMES):
    all_frames[float(i)] = []
  with open(csv_path) as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    for row in reader:

      #row = [row[0],row[2],row[3],row[4],row[5],row[1]]
      frame = float(row[0])
      if not (frame%SKIP_FRAMES == 0):
        continue
      if frame <= total_frame_count:
        # frame, x1, y1, x2, y2, worm_id
        all_frames[frame].append([frame,float(row[1]),float(row[2]),float(row[3]),float(row[4]),0])
      else:
        print("Invalid Frame")
        break
  working_worms = []
  for i in all_frames:
    if not i%FOLDER_NUM in working_worms:
            working_worms.append(i%FOLDER_NUM)
    if not os.path.exists(out_path+"/"+str(int(i%FOLDER_NUM))):
      os.mkdir(out_path+"/"+str(int(i%FOLDER_NUM)))

  for i in range(0,int(total_frame_count)):
    ret, frame = vid.read()
    frame_count = vid.get(cv2.CAP_PROP_POS_FRAMES)
    if frame_count == 1:
      height, width, channels = frame.shape
      fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    frame_cur = int(frame_count)
    quarter_frame = round(total_frame_count/4)
    if frame_cur == 1:
      print("Starting Video:",video_id)
    elif frame_cur == quarter_frame:
      print("25% Complete: ",frame_cur," frames out of ",total_frame_count)
    elif frame_cur == quarter_frame*2:
      print("50% Complete: ",frame_cur," frames out of ",total_frame_count)
    elif frame_cur == quarter_frame*3:
      print("75% Complete: ",frame_cur," frames out of ",total_frame_count)
    if frame_cur in all_frames:
      all_worms_in_frame = all_frames[frame_cur]
    else:
      all_worms_in_frame = []

    for worm in all_worms_in_frame:
      # Because this is a video, we have to use integers to pull pixels.
      if worm[0]%FOLDER_NUM in working_worms:

        x1 = int(worm[1]); x2 = x1 + int(worm[3])
        y1 = int(worm[2]); y2 = y1 + int(worm[4])
        if x1 > x2:
          temp = x2
          x2 = x1
          x1 = temp
        if y1 > y2:
          temp = y2
          y2 = y1
          y1 = temp
        if x1 <= 0:
          x1 = 1
        if y1 <= 0:
          y1 = 1
        if x2 > width:
          x2 = width
        if y2 > height:
          y2 = height
        im = frame[y1: y2, x1: x2]
        data = Image.fromarray(im)
        file_name = "_".join([str(video_id),str(frame_cur),"1","x1y1x2y2",str(x1),str(y1),str(x2),str(y2)])+".png"
        data.save(out_path+"/"+str(frame_cur%FOLDER_NUM)+"/"+file_name)
  print("100% Complete!")

def runAllPixellibUnsorted(vid_path,csv_path,folder_path):
  """
  Takes a video and a sorted csv file and turns it entirely into data

  vid_path: The path to the video
  csv_path: The sorted csv file
  folder_path: The path to store outputs
  """
  start = t.time()
  if not SKIP_ANNO:
    createImagesUnsorted(vid_path,csv_path,folder_path)
    ytp.createAnnotatedFolders(folder_path)
  func_list = [ci.getArea, ci.getAverageShade,sc.getCmlAngle,sc.getCmlDistance,ci.getMaxWidth,ci.getMidWidth,sc.getDiagonalNum]
  ytp.storeAllData(folder_path, func_list)
  ytp.storeVideos(folder_path)
  stop = t.time()
  print("Complete!")
  print("Time taken:", stop-start)

if __name__ == "__main__":
  try:
    avi_path = sys.argv[1]
  except IndexError:
    print("No AVI path detected.")
    raise

  try:
    csv_path = sys.argv[2]
  except IndexError:
    print("No CSV path detected.")
    raise

  try:
    output_path = sys.argv[3]
  except:
    print("No output path detected")
    raise
  runAllPixellibUnsorted(avi_path,csv_path,output_path)
  #runAllPixellibUnsorted("C:/Users/cdkte/Downloads/timelapse/344.avi","C:/Users/cdkte/Downloads/timelapse/344.csv","C:/Users/cdkte/Downloads/timelapse/photos")
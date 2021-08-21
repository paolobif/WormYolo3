import numpy as np
import cv2
import csv
import os
from PIL import Image
import segmentation as seg
import image_analysis as ia
import multiprocessing
import skeleton_cleaner as sc
import convert_image as ci
import time as t
import shutil
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
Handles the transition between YOLO's output of CSV files and Pixellib's input of images
"""

# The padding for worm images
PADDING = 10

# The prefixes for annotated images
DATA_PATH = "Anno"
SKELE_PATH = "Skele"

# The model that should be used for semantic segmentation
MODEL_PATH = "C:/Users/cdkte/Downloads/worm_segmentation/model_folder/mask_rcnn_model.052-0.130289.h5"

# The number of Tensorflow processes that can be run at one time
PROCESS_NUM = 1

def createImages(vid_path,csv_path,out_path):
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
  video_id = os.path.basename(vid_path).strip('.avi')
  total_frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
  all_frames = {}
  unique_worms = {}
  for i in range(int(total_frame_count)):
    all_frames[int(i)] = []
  with open(csv_path) as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    for row in reader:
      frame = float(row[0])
      if frame <= total_frame_count:
        # frame, x1, y1, x2, y2, worm_id
        all_frames[frame].append([frame,float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5])])
        if float(row[5]) not in unique_worms:
          unique_worms[float(row[5])] = 1
        else:
          unique_worms[float(row[5])] += 1
      else:
        print("Invalid Frame")
        break

  working_worms = []
  for i in unique_worms:
    if not os.path.exists(out_path+"/"+str(i)):
      if unique_worms[i]>=45:
        working_worms.append(i)
        os.mkdir(out_path+"/"+str(i))
  for i in range(int(total_frame_count)):
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

      if float(worm[5]) in working_worms:
        x1 = int(worm[1])
        y1 = int(worm[2])
        x2 = int(worm[3])
        y2 = int(worm[4])
        im = frame[y1-PADDING: y2+PADDING, x1-PADDING: x2+PADDING]
        data = Image.fromarray(im)
        file_name = "_".join([str(video_id),str(frame_cur),str(worm[5]),"x1y1x2y2",str(x1),str(y1),str(x2),str(y2)])+".png"
        data.save(out_path+"/"+str(worm[5])+"/"+file_name)
  print("100% Complete!")

def createAnnotatedFolders(folder_path):
  all_files = os.listdir(folder_path)
  i = 0
  total_length = len(all_files)
  multi_list = []
  #anno_folder = lambda file: seg.annotateFolder(folder_path+"/"+file,MODEL_PATH,folder_path+"/"+DATA_PATH+file)
  for file in all_files:
    i+=1
    if os.path.isdir(folder_path+"/"+file) and file.split("_")[0]!=DATA_PATH:
      p = multiprocessing.Process(target = seg.annotateFolder, args=(folder_path+"/"+file,MODEL_PATH,folder_path+"/"+DATA_PATH+"_"+file))
      p.start()
      multi_list.append(p)
      #p.join()
      #cleanFolder(folder_path)
    print(i," out of ",total_length," started")
    if len(multi_list) >= PROCESS_NUM:
      for j in range(len(multi_list)):
        proc = multi_list[0]
        proc.join()
        multi_list.remove(proc)

def gatherAllData(folder_path, function_list):
  all_files = os.listdir(folder_path)
  i = 0
  total_length = len(all_files)/2
  # Tensorflow has memory issues
  all_data=np.empty((0,6+len(function_list)))
  for file in all_files:

    if os.path.isdir(folder_path+"/"+file) and file.split("_")[0] == DATA_PATH:
      print(i," out of ",total_length)
      i+=1
      file_data = ia.folder_data(folder_path+"/"+file, function_list)
      all_data = np.vstack((all_data,file_data))

  return all_data

def storeAllData(folder_path, function_list, match_dict = {}):
  titles =   titles = ia.functionNames(function_list,match_dict)
  all_data = gatherAllData(folder_path, function_list)
  folder_name = os.path.basename(folder_path)

  np.savetxt(folder_path+"/"+folder_name+".csv", all_data, header = titles, delimiter = ",")

def storeVideos(folder_path):
  print("Creating videos!")
  all_files = os.listdir(folder_path)
  i = 0
  total_length = len(all_files)/2
  for folder in all_files:
    if os.path.isdir(folder_path+"/"+folder) and folder.split("_")[0] == DATA_PATH:
      orig_folder = folder.split("_")[-1]
      for file in os.listdir(folder_path+"/"+orig_folder):
        os.remove(folder_path+"/"+orig_folder+"/"+file)
      ia.makeSkeleVideo(folder_path+"/"+folder,folder_path+"/"+orig_folder+"/"+SKELE_PATH+"_"+orig_folder+".avi")
      ia.makeAnnoVideo(folder_path+"/"+folder,folder_path+"/"+orig_folder+"/"+DATA_PATH+"_"+orig_folder+".avi")
      i += 1
      print(i,"out of",total_length,"complete!")
      shutil.rmtree(folder_path+"/"+folder)

def runAllPixellib(vid_path,csv_path,folder_path):
  start = t.time()
  createImages(vid_path,csv_path,folder_path)
  createAnnotatedFolders(folder_path)
  func_list = [ci.getArea, ci.getAverageShade,sc.getCmlAngle,sc.getCmlDistance,ci.getMaxWidth,ci.getMidWidth,sc.getDiagonalNum]
  storeAllData(folder_path, func_list)
  storeVideos(folder_path)
  stop = t.time()
  print("Complete!")
  print("Time taken:", stop-start)

if __name__ == "__main__":
  func_list = [ci.getArea, ci.getAverageShade,sc.getCmlAngle,sc.getCmlDistance,ci.getMaxWidth,ci.getMidWidth,sc.getDiagonalNum]

  #storeAllData("206",func_list)
  runAllPixellib("C:/Users/cdkte/Downloads/Mot_Single/206.avi","C:/Users/cdkte/Downloads/Mot_Single/206_sort.csv","206")
  #storeVideos("206")





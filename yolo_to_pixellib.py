import numpy as np
import cv2
import csv
import os
from PIL import Image
import image_analysis as ia
import skeleton_cleaner as sc
import convert_image as ci
import time as t
import shutil
import os
import sys
import subprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
Handles the transition between YOLO's output of CSV  les and Pixellib's input of images
"""

# The padding for worm images
PADDING = 10

# The prefixes for annotated images
DATA_PATH = "Anno"
SKELE_PATH = "Skele"

# The model that should be used for semantic segmentation
MODEL_PATH = "C:/Users/cdkte/Downloads/worm_segmentation/model_folder/mask_rcnn_model.008-0.419976.h5"

# The number of Tensorflow processes that can be run at one time
PROCESS_NUM = 1

# Delete images and annotated images after creating the video
DELETE_FRAMES = False

# Delete folders created in processing
DELETE_FOLDERS = False

# The minimum number of frames for a unique worm to appear in for it to be tracked
MIN_FRAMES = 10

# Number of frames to move at a time (I recommend reducing min_frames when you increase this)
# 1 skips no frames, 2 takes every second frame
SKIP_FRAMES = 2

# Skip creation and annotation of images
SKIP_ANNO = False

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
  video_id = os.path.basename(vid_path).strip('.avi').split("_")[0]
  total_frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
  all_frames = {}
  unique_worms = {}
  for i in range(0,int(total_frame_count)+1,SKIP_FRAMES):
    all_frames[float(i)] = []
  with open(csv_path) as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    for row in reader:

      row = [row[0],row[2],row[3],row[4],row[5],row[1]]
      #row = [row[0],row[1],row[2],row[3],row[4],row[5]]

      frame = float(row[0])
      if not (frame%SKIP_FRAMES == 0):
        continue
      if frame <= total_frame_count:
        # frame, x1, y1, x2, y2, worm_id
        all_frames[frame].append([frame,float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5])])
        if float(row[5]) not in unique_worms:
          unique_worms[float(row[5])] = 1
        else:
          if float(row[5]) == 81.0:
            pass
          unique_worms[float(row[5])] += 1
      else:
        print("Invalid Frame")
        break
  working_worms = []
  for i in unique_worms:
    if not os.path.exists(out_path+"/"+str(i)):
      if unique_worms[i]>=MIN_FRAMES:
        working_worms.append(i)
        os.mkdir(out_path+"/"+str(i))

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

      if float(worm[5]) in working_worms:
        x1 = int(worm[1]) - PADDING
        y1 = int(worm[2]) - PADDING
        x2 = int(worm[3]) + PADDING
        y2 = int(worm[4]) + PADDING
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
        file_name = "_".join([str(video_id),str(frame_cur),str(worm[5]),"x1y1x2y2",str(x1),str(y1),str(x2),str(y2)])+".png"
        data.save(out_path+"/"+str(worm[5])+"/"+file_name)
  print("100% Complete!")

def createAnnotatedFolders(folder_path):
  """
  For each folder of images within folder_path, creates a new folder for highlighted images
  and fills it. The new folders are also in folder_path
  Uses subprocess to clear Tensorflow memory

  folder_path: The directory containing the folders of images
  """
  all_files = os.listdir(folder_path)
  i = 0
  total_length = len(all_files)
  multi_list = []
  for file in all_files:
    #"""
    i+=1
    print(i,"started out of",total_length)
    if os.path.isdir(folder_path+"/"+file) and file.split("_")[0]!=DATA_PATH:
      first_input = "1"
      second_input = folder_path+"/"+file
      third_input = folder_path+"/"+DATA_PATH+"_"+file
      segmentation_file = "/".join(__file__.split("/")[0:-1]) + "segmentation.py"
      subprocess.call(["python3",segmentation_file,first_input,second_input,third_input])

def gatherAllData(folder_path):
  """
  Collects the information from individual images and returns it as a matrix.
  Run on the directory containing multiple worm ID folders, each of which contains images of that specific worm.

  Args:
    folder_path: The path to the folder of folders of highlighted images to get data from

  Returns:
    all_data: Numpy array of the data taken from all the worms in the folder
  """
  all_files = os.listdir(folder_path)
  i = 0
  total_length = len(all_files)/2
  all_data=np.empty((0,23))
  for file in all_files:

    if os.path.isdir(folder_path+"/"+file) and file.split("_")[0] == DATA_PATH:
      print(i," out of ",total_length)
      i+=1
      file_data = ia.folder_data(folder_path+"/"+file)
      all_data = np.vstack((all_data,file_data))

  return all_data

def storeAllData(folder_path):
  """
  Takes highlighted images from subfolders of folder_path and gathers data, storing the results to a csv file.

  Args:
    folder_path: The directory where the id folders are stored
  """

  # Get Labels for each column
  titles =   titles = ia.functionNames()

  # Collect Data
  all_data = gatherAllData(folder_path)

  # Write data to csv
  folder_name = os.path.basename(folder_path)
  np.savetxt(folder_path+"/"+folder_name+".csv", all_data, header = titles, delimiter = ",")

def storeVideos(folder_path):
  """
  Creates a skeleton video and a highlighted video for each folder in folder_path

  Args:
    folder_path: The folder of folders with highlighted images
  """
  print("Creating videos!")
  all_files = os.listdir(folder_path)
  i = 0
  total_length = len(all_files)/2
  for folder in all_files:
    if os.path.isdir(folder_path+"/"+folder) and folder.split("_")[0] == DATA_PATH:
      orig_folder = folder.split("_")[-1]
      if DELETE_FRAMES:
        for file in os.listdir(folder_path+"/"+orig_folder):
          os.remove(folder_path+"/"+orig_folder+"/"+file)
      ia.makeSkeleVideo(folder_path+"/"+folder,folder_path+"/"+orig_folder+"/"+SKELE_PATH+"_"+orig_folder+".avi")
      ia.makeAnnoVideo(folder_path+"/"+folder,folder_path+"/"+orig_folder+"/"+DATA_PATH+"_"+orig_folder+".avi")
      i += 1
      print(i,"out of",total_length,"complete!")
      if DELETE_FRAMES:
        shutil.rmtree(folder_path+"/"+folder)

def clearFolders(folder_path):
  """
  Deletes all folders within a folder. Don't point this at anything you care about!

  Args:
    folder_path: The folder containing folders to delete

  Returns:
    List of the names of deleted directories
  """
  del_arr = []
  if DELETE_FOLDERS:
    for file in os.listdir(folder_path):
      if os.path.isdir(folder_path+"/"+file):
        del_arr.append()
        shutil.rmtree(folder_path+"/"+file)

def runAllPixellib(vid_path,csv_path,folder_path):
  """
  Takes a video and a sorted csv file and turns it entirely into data

  Args:
    vid_path: The path to the video
    csv_path: The csv file that tells where sorted worms are
    folder_path: The path to store outputs
  """
  start = t.time()
  if not SKIP_ANNO:
    createImages(vid_path,csv_path,folder_path)
    createAnnotatedFolders(folder_path)
  storeAllData(folder_path)
  storeVideos(folder_path)
  clearFolders(folder_path)
  stop = t.time()
  print("Complete!")
  print("Time taken:", stop-start)

if __name__ == "__main__":

  try:
    avi_path = sys.argv[1]
  except IndexError:
    print("No AVI path detected. Using default path")
    avi_path = "C:/Users/cdkte/Downloads/641_day4_simple.avi"

  try:
    csv_path = sys.argv[2]
  except IndexError:
    print("No CSV path detected. Using default path")
    csv_path = "C:/Users/cdkte/Downloads/641_day4_simple_sort.csv"

  try:
    output_path = sys.argv[3]
  except:
    print("No output path detected")
    output_path = "C:/641"

  try:
    MODEL_PATH = sys.argv[4]
  except IndexError:
    print("No model: Using default model")

  try:
    MIN_FRAMES = int(sys.argv[5])
  except IndexError:
    print("No minimum number of frames: Using default value =",MIN_FRAMES)
  except ValueError:
    print("Invalid value for minimum number of frames: Using default value =",MIN_FRAMES)

  try:
    DELETE_FRAMES = bool(sys.argv[6])
  except ValueError:
    print("Invalid bool value for deleting frames: Using default setting")
  except IndexError:
    print("No bool for deleting frames: Using default setting")

  try:
    PROCESS_NUM = int(sys.argv[7])
  except ValueError:
    print("Invalid int value for number of Tensorflow processes: Using default setting")
  except IndexError:
    print("No int for number of Tensorflow procceses: Using default setting")

  runAllPixellib(avi_path,csv_path,output_path)




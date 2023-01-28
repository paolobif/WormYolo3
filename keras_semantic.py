import tensorflow_sdf as tsdf
import cv2
from matplotlib import pyplot as plt
import os
import time as t
import all_image_analysis as aia
import pandas as pd
import numpy as np
from tqdm import tqdm
import blot_circle_crop

# TODO: Change Sort parameters?
#   1-3 hits before being lost originally, if you have it go out more frames it'll assume the worm moved a large distance
# Do this, change parameters on a vid w/ good detection
#   Specifically downsample vids to test whether that's the problem
#   Current # of hits as 0, propagate 0 or 1 frames
#   Current ToD links tracks back to each other

# TODO: Once finish, get behavior data
# Compile and send behavior and morphology to group

# TODO: Can we do lifespan off the videos?
# Link worms between videos - same or not
# Head/tail changes
# Start w/ all dead animals as "ground truth" like jitter
# Ideally dead animals are fixed
# Degrading? Can we detect worms that "show up" when in reverse?
#   Try using moving window to update persistent bounding boxes
# "Constellation of Death"
# Once we have consistent bbs, check head/tail twitches

# Make a folder of one wormbot with all the days
# Start on back
# Ignore IDs and only look at position. Range at start, and range at end


# Lab meeting Friday this week joint w/ Mendenhall Lab
# Facetime(Zoom!) in for lab meeting
# TODO: Set calendar for Sept16 meeting(every 2 weeks)

# TODO: Validate and Compile

# RERUN Motility Pixellib

# TODO: Check type of experiments and send to Renu

# TODO: Show small part of focused video with original and anno, maybe add measurements next to it
# Try tracking bounding box seperate from YOLO detections?

# TODO: Validation targeted for end of life?
# Consider how to match worms with the outline

# TODO: Ideate on compare vids between days to detect dead
# Jitter detection?

# TODO: DrugAge hard drive for experiments with good detection off time-lapse
# Pull those and analyze Daily monitor movies

DO_SORT = True

def data_from_vid(vid_path,in_csv, out_csv,vid_id = 0,fail_path = None):
  start = t.time()
  # Prep vid and data
  cap = cv2.VideoCapture(vid_path)
  if not DO_SORT:
    csv = pd.read_csv(in_csv, usecols=[0, 1, 2, 3, 4], names=["frame", "x1", "y1", "width", "height"])
  else:
    csv = pd.read_csv(in_csv, usecols=[0, 1, 2, 3, 4, 5], names=["frame", "x1", "y1", "x2", "y2", "wormID"])

  ret, frame = cap.read()
  cur_frame = 0


  frame_width = cap.get(3)
  frame_height = cap.get(4)

  size = (int(frame_width),int(frame_height))

  all_data = np.empty((23,))


  while ret:
    #1080 1920 3
    #worm_data = np.array()

    cur_frame+=1

    # Get worms in this frame
    cur_worms = csv.where(csv["frame"]==cur_frame)
    cur_worms = cur_worms.dropna()

    for index, row in cur_worms.iterrows():
      x1=int(row["x1"])
      if not DO_SORT:
        x2=x1+int(row["width"])
      else:
        x2 = int(row["x2"])
      y1=int(row["y1"])
      if not DO_SORT:
        y2=y1+int(row["height"])
      else:
        y2 = int(row["y2"])

      # Replace with actual worm id when using SORT
      if not DO_SORT:
        worm_id = 0
      else:
        worm_id = row["wormID"]

      orig_x1 = x1
      orig_x2 = x2
      orig_y1 = y1
      orig_y2 = y2
      if x1-10 > 0:
        x1-=10
      else:
        x1=0
      if y1-10 > 0:
        y1-=10
      else:
        y1 = 0
      if x2+11 < frame_width:
        x2+=10
      if y2+11 < frame_height:
        y2+=10

      worm_box = frame[y1:y2,x1:x2,:]

      try:
        anno_img = tsdf.generate_single_sdf(worm_box)
      except Exception as e:
        print(y1,y2,x1,x2)
        raise e
      #figs, ax = plt.subplots(2)
      #ax[0].imshow(worm_box)
      #ax[1].imshow(anno_img)
      #plt.show()
      worm_box = frame[y1:y2,x1:x2,0]

      anno_box = worm_box[0]+anno_img*100

      anno_img = np.where(anno_img >= 0.5, 1, 0)
      if np.sum(anno_img) < 10:
        if not fail_path is None:
          file_name = str(cur_frame)+"_"+str(orig_x1)+"_"+str(orig_x2)+"_"+str(orig_y1)+"_"+str(orig_y2)+".png"
          file_path = os.path.join(fail_path,file_name)
          #print(file_path)
          cv2.imwrite(file_path,worm_box)
        continue
      #plt.imshow(anno_img)
      #plt.show()
      #anno_img = np.array([anno_img,anno_img,anno_img])
      #anno_img = np.swapaxes(anno_img,0,2)
      #print(anno_img.shape)

      if fail_path is None:
        worm_dict = makeDict(anno_img)
        if not DO_SORT:
          worm_val = aia.matrixAnalysis(worm_box,worm_dict,vid_id,cur_frame,worm_id,x1,y1,x2,y2)
        else:
          worm_val = aia.matrixAnalysis(worm_box,worm_dict,vid_id,cur_frame,worm_id,x1,y1,x2,y2)
      else:
        pass
      try:
        worm_dict = makeDict(anno_img)
        worm_val = aia.matrixAnalysis(worm_box,worm_dict,vid_id,cur_frame,worm_id,x1,y1,x2,y2)
      except:
        file_name = "curiss_"+str(cur_frame)+"_"+str(x1)+"_"+str(x2)+"_"+str(y1)+"_"+str(y2)+".png"
        print(file_name)
        file_path = os.path.join("C:/Users/cdkte/Downloads/new_error_2",file_name)
        anno_name = str(cur_frame)+"_"+str(x1)+"_"+str(x2)+"_"+str(y1)+"_"+str(y2)+"_anno"+".png"
        anno_f_path = os.path.join("C:/Users/cdkte/Downloads/new_error_2",anno_name)
        cv2.imwrite("C:/Users/cdkte/Downloads/new_error_2",worm_box)
        cv2.imwrite(anno_f_path,anno_box)
        continue

      # Ignore invalid images
      if np.sum(worm_val[7:]) == 0:
        if not fail_path is None:
          file_name = str(cur_frame)+"_"+str(x1)+"_"+str(x2)+"_"+str(y1)+"_"+str(y2)+".png"
          file_path = os.path.join(fail_path,file_name)
          anno_name = str(cur_frame)+"_"+str(x1)+"_"+str(x2)+"_"+str(y1)+"_"+str(y2)+"_anno"+".png"
          anno_f_path = os.path.join(fail_path,anno_name)
          #print(file_path)
          cv2.imwrite(file_path,worm_box)
          #plt.imshow(copy_img)
          #plt.show()
          cv2.imwrite(anno_f_path,anno_box)
        continue

      all_data = np.vstack((all_data,worm_val))
      #cv2.rectangle(frame, (x1,y1), (x2,y2), color = (0,0,255), thickness =2)
    #plt.imshow(frame)
    #plt.show()
    #print(all_data.shape)
    ret, frame = cap.read()
  stop = t.time()
  print((stop-start)/60)

  # Drop Worm-ID column
  if not DO_SORT:
    all_data = np.delete(all_data,5,1)

  header = ",".join(["Video Frame","x1","y1","x2","y2","Area","Shade","Cumulative Angle","Length","Max Width","Mid Width","Diagonals","Point1_x","Point1_y","Point2_x","Point2_y","Point3_x","Point3_y","Point4_x","Point4_y","Point5_x","Point5_y"])
  np.savetxt(out_csv,all_data,header=header,delimiter=",")

def makeDictCol(matx_list):
  out_list = []
  for matx in matx_list:
    bool_matx = np.copy(matx)
    col_size = np.sum(bool_matx,0)
    worm_in = np.where(col_size > 0)[0]
    zero_matrix = np.zeros(bool_matx.shape)
    for i in range(1,len(worm_in)):
      if worm_in[i] > worm_in[i-1]+1:
        left = np.copy(bool_matx)

        left[:,i:] = 0

        bool_matx[:,:i] = 0
        out_list.append(left)
    out_list.append(bool_matx)
  return out_list

def makeDictRow(matx_list):
  out_list = []
  for matx in matx_list:
    bool_matx = np.copy(matx)
    col_size = np.sum(bool_matx,1)
    worm_in = np.where(col_size > 0)[0]
    zero_matrix = np.zeros(bool_matx.shape)
    for i in range(1,len(worm_in)):
      if worm_in[i] > worm_in[i-1]+1:
        left = np.copy(bool_matx)

        left[i:,:] = 0

        bool_matx[i:,:] = 0
        out_list.append(left)
    out_list.append(bool_matx)
  return out_list



def makeDictCol(matx_list):
  out_list = []
  for matx in matx_list:
    bool_matx = np.copy(matx)
    col_size = np.sum(bool_matx,0)
    worm_in = np.where(col_size > 0)[0]
    zero_matrix = np.zeros(bool_matx.shape)
    for i in range(1,len(worm_in)):
      col_v = worm_in[i]
      if col_v > worm_in[i-1]+1:

        left = np.copy(bool_matx)

        left[:,col_v:] = 0

        bool_matx[:,:col_v] = 0
        out_list.append(left)
    out_list.append(bool_matx)
  return out_list

def makeDictRow(matx_list):
  out_list = []
  for matx in matx_list:
    bool_matx = np.copy(matx)
    row_size = np.sum(bool_matx,1)
    worm_in = np.where(row_size > 0)[0]
    zero_matrix = np.zeros(bool_matx.shape)
    for i in range(1,len(worm_in)):
      row_v = worm_in[i]
      if row_v > worm_in[i-1]+1:

        left = np.copy(bool_matx)

        left[row_v:,:] = 0

        bool_matx[:row_v,:] = 0
        out_list.append(left)
    out_list.append(bool_matx)
  return out_list


def makeDictDiag(matx_list,direction = 0):
  out_list = []
  if direction!= 0 and direction!= 1:
    raise ValueError("Direction is not 0 or 1")
  for matx in matx_list:
    num_of_diags = matx.shape[0]+matx.shape[1]-1
    low=-int(np.floor(num_of_diags/2))
    high = int(np.ceil(num_of_diags/2))
    if direction == 0:
      matx_trace = lambda x: np.trace(matx,x)
    elif direction == 1:
      flipped_matx = np.fliplr(matx)
      matx_trace = lambda x: np.trace(flipped_matx,x)

    diag_size = np.array([matx_trace(x) for x in range(low,high+1)])
    worm_in = np.where(diag_size > 0)[0]
    for i in range(1, len(worm_in)):
      row_v = worm_in[i]
      if row_v > worm_in[i-1]+1:
        try:
          mask = constructZeroDiag(matx.shape,low+row_v,direction)
        except Exception as E:
          print(E)
          plt.imsave("C:/Users/cdkte/Downloads/stupid_worm_not_work.png",matx)
          plt.imshow(matx)
          plt.show()

        left = np.copy(matx)

        left = np.multiply(left,mask)
        matx = np.multiply(matx,1-mask)

        out_list.append(left)
    out_list.append(matx)
    return out_list



def constructZeroDiag(shape, offset, direction=0):
  base_arr = np.zeros(shape)
  height = shape[0]
  wid = shape[1]
  for i in range(offset+1):
    size = i+1
    if size > wid:
      size = wid
    ones = np.ones((size))
    if direction == 0:
      base_arr[height-offset-1+i,0:size] += ones
    elif direction == 1:
      base_arr[height-offset-1+i,wid-size:wid] += ones
      #base_arr[offset-i,0:size] += ones
    else:
      raise ValueError("Not 0 or 1")
  return base_arr

def makeDict(bool_matrix):
  bool_matrix = np.copy(bool_matrix)
  bool_matrix = blot_circle_crop.dropSmall(bool_matrix)
  worm_matrices = makeDictCol([bool_matrix])
  worm_matrices = makeDictRow(worm_matrices)
  worm_matrices = makeDictDiag(worm_matrices)
  #print(len(worm_matrices))

  #worm_matrices = makeDictDiag(worm_matrices,1)
  #print(len(worm_matrices))
  #worm_matrices =  makeDictCol(worm_matrices)

  return dict(zip(range(len(worm_matrices)),worm_matrices))

if __name__ == "__main__":
  """
  vid_folder = "C:/Users/cdkte/Downloads/N2_controls/raw"
  csv_folder = "C:/Users/cdkte/Downloads/N2_controls/neural"
  out_folder = "C:/Users/cdkte/Downloads/N2_controls/out"
  fail_path = "C:/Users/cdkte/Downloads/new_error_4"

  for vid_file in tqdm(os.listdir(vid_folder)):
    vid_id = vid_file.split(".")[0]
    vid_path = os.path.join(vid_folder, vid_file)
    csv_path = os.path.join(csv_folder,vid_id+".csv")
    if not os.path.exists(csv_path):
      continue

    out_path = os.path.join(out_folder,vid_id+"_pix.csv")

    data_from_vid(vid_path,csv_path,out_path,int(vid_id),fail_path = fail_path)
  """
  vid_path = "C:/Users/cdkte/Downloads/641_day6_simple.avi"
  csv_path = "C:/Users/cdkte/Downloads/641_day6_simple_sort.csv"
  out_csv = "C:/Users/cdkte/Downloads/641_written.csv"
  fail_path = "C:/Users/cdkte/Downloads/error_imgs_5"

  data_from_vid(vid_path,csv_path,out_csv,vid_id = 320)#,fail_path = fail_path)
  #"""



import tensorflow_sdf as tsdf
import cv2
from matplotlib import pyplot as plt
import os
import time as t
import all_image_analysis as aia
import pandas as pd
import numpy as np
from tqdm import tqdm

WORM_IDs = False

def data_from_vid(vid_path,in_csv, out_csv,vid_id = 0,drop_worm_ids = True,fail_path = None):
  start = t.time()
  # Prep vid and data
  cap = cv2.VideoCapture(vid_path)
  csv = pd.read_csv(in_csv, usecols=[0, 1, 2, 3, 4], names=["frame", "x1", "y1", "width", "height"])

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
      x2=x1+int(row["width"])
      y1=int(row["y1"])
      y2=y1+int(row["height"])

      # Replace with actual worm id when using SORT
      worm_id = 0

      orig_x1 = x1
      orig_x2 = x2
      orig_y1 = y1
      orig_y2 = y2
      if x1-10 > 0:
        x1-=10
      if y1-10 > 0:
        y1-=10
      if x2+11 < frame_width:
        x2+=10
      if y2+11 < frame_height:
        y2+=10

      worm_box = frame[y1:y2,x1:x2,:]


      anno_img = tsdf.generate_single_sdf(worm_box)
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

      worm_dict = makeDict(anno_img)
      worm_val = aia.matrixAnalysis(worm_box,worm_dict,vid_id,cur_frame,worm_id,x1,y1,x2,y2)

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
  if drop_worm_ids:
    all_data = np.delete(all_data,5,1)

  header = ",".join(["frame","x1","y1","x2","y2","area","shade","Cumulative Angle","Length","Max Width","Mid Width","Diagonals","Point1_x","Point1_y","Point2_x","Point2_y","Point3_x","Point3_y","Point4_x","Point4_y","Point5_x","Point5_y"])
  np.savetxt(out_csv,all_data,header="")

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
        mask = constructZeroDiag(matx.shape,low+row_v,direction)
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
  worm_matrices = makeDictCol([bool_matrix])
  worm_matrices = makeDictRow(worm_matrices)
  worm_matrices = makeDictDiag(worm_matrices)
  print(len(worm_matrices))

  worm_matrices = makeDictDiag(worm_matrices,1)
  print(len(worm_matrices))
  worm_matrices = makeDictCol(worm_matrices)

  return dict(zip(range(len(worm_matrices)),worm_matrices))

if __name__ == "__main__":
  vid_folder = "C:/Users/cdkte/Downloads/vid"
  csv_folder = "C:/Users/cdkte/Downloads/csv"
  out_folder = "C:/Users/cdkte/Downloads/out"

  for vid_file in tqdm(os.listdir(vid_folder)):
    vid_id = vid_file.split(".")[0]
    vid_path = os.path.join(vid_folder, vid_file)
    csv_path = os.path.join(csv_folder,vid_id+".csv")
    if not os.path.exists(csv_path):
      continue

    out_path = os.path.join(out_folder,vid_id+"_pix.csv")

    data_from_vid(vid_path,csv_path,out_path,int(vid_id))
  """
  vid_path = "C:/Users/cdkte/Downloads/320(1).avi"
  csv_path = "C:/Users/cdkte/Downloads/320.csv"
  out_csv = "C:/Users/cdkte/Downloads/320_written.csv"
  fail_path = "C:/Users/cdkte/Downloads/error_imgs"

  data_from_vid(vid_path,csv_path,out_csv,vid_id = 320)#,fail_path = fail_path)
  """



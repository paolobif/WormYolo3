import convert_image as ci
import segmentation as seg
import skeleton_cleaner as sc
import numpy as np
import os
import time as t
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt

"""
Requires Tensorflow, cv2, and Pixellib
"""

DATA_PATH = "Anno_"


def single_data(folder_path,img_name, function_list):
  """
  Creates a list of data for the given image
  ---
  folder_path: The folder which holds the image
  img_name: The file name of the image
  function_list: The list of functions that the image should be run through
  """
  #Correct img_name format: Annotated_344_469_4967.0_x1y1x2y2_909_835_966_855.png
  worm_dict, grayscale_matrix = ci.getWormMatrices(folder_path+"/"+img_name)
  select_worm = ci.findCenterWorm(worm_dict)
  parsed_name = img_name.split("_")
  #annotated = parsed_name[0]
  vid_id = parsed_name[1]
  frame = parsed_name[2]
  worm_id = parsed_name[3]
  outnumpy = np.array([worm_id, frame])
  for func in function_list:
    outnumpy = np.append(outnumpy, func(select_worm, grayscale_matrix))
  return outnumpy

def folder_data(folder_path, function_list):
  """
  Creates a matrix of data for all files in the given folder
  ---
  folder_path: The folder to be prpocessed
  function_list: The list of functions to be run on each file
  """
  files = os.listdir(folder_path)
  arr_shape = (len(files), 2+len(func_list))
  percent_check = [round(len(files)/4), round(len(files)/2), round(3*len(files)/4)]
  out_arr = np.empty(arr_shape)
  for i in range(len(files)):
    out_arr[i] = single_data(folder_path, files[i], function_list)
    if i in percent_check:
      print(str(int(i*100/len(files))) + "% complete on folder "+folder_path)

  sorted_array = out_arr[np.argsort(out_arr[:, 0])]
  sorted_array = sorted_array[np.argsort(sorted_array[:, 1])]
  return sorted_array

def save_folder(folder_path, out_file, function_list):
  """
  Runs folder_data and saves it in a file
  ---
  folder_path: The folder to be processed
  out_file: The file to store the matrix in
  function_list: The list of functions to be run on each file
  """
  sorted_array = folder_data(folder_path, function_list)
  np.savetxt(out_file, sorted_array, delimiter=",")

def data_folder(folder_path, func_list):
  """
  Highlights the images in a folder and then saves it in a file
  ---
  folder_path: The folder to be processed (currently only works with folders within the cwd)
  func_list: The list of functions to be run on each file
  """
  seg.annotateFolder(folder_path, "models\\mask_rcnn_model.002-0.633895.h5", DATA_PATH+folder_path)
  save_folder("Anno_"+folder_path, DATA_PATH+folder_path+".csv", func_list)

def makeSkeleVideo(folder_path, output_path):
  """
  Creates a video of the annotated worm as it moves with a middle line the entire time
  """
  files = os.listdir(folder_path)
  # Find largest image
  max_width = 0
  max_height = 0
  for file in files:
    img = cv2.imread(folder_path+"/"+file)
    height, width, colors = img.shape
    if height > max_height:
      max_height = height
    if width > max_width:
      max_width = width


  writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, (max_width+1, max_height+1), True)
  for file in files:
    empty_image = np.zeros((max_height+1,max_width+1,3),'uint8')
    worm_dict, grayscale_matrix = ci.getWormMatrices(folder_path+"/"+file)
    selectWorm = ci.findCenterWorm(worm_dict)
    wormFront = ci.findFront(selectWorm)
    skelList = ci.createMiddleSkeleton((wormFront[1],wormFront[0]),selectWorm)
    img = ci.makeSkelImg(selectWorm, grayscale_matrix, skelList)
    height, width, colors = img.shape
    h_padding = round((max_height - height)/2)
    w_padding = round((max_width - width)/2)
    empty_image[h_padding:h_padding+height,w_padding:w_padding+width] = img
    #plt.imshow(empty_image)
    #plt.show()
    writer.write(empty_image)

  del writer

def makeSkeleEstimate(folder_path, output_path):

  """
  Creates a video of the annotated worm as it moves with an approximate middle line the entire time
  """
  files = os.listdir(folder_path)
  # Find largest image
  max_width = 0
  max_height = 0
  for file in files:
    img = cv2.imread(folder_path+"/"+file)
    height, width, colors = img.shape
    if height > max_height:
      max_height = height
    if width > max_width:
      max_width = width

  writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, (max_width+1, max_height+1), True)
  for file in files:
    empty_image = np.zeros((max_height+1,max_width+1,3),'uint8')
    worm_dict, grayscale_matrix = ci.getWormMatrices(folder_path+"/"+file)
    selectWorm = ci.findCenterWorm(worm_dict)
    wormFront = ci.findFront(selectWorm)
    skelList = ci.createMiddleSkeleton((wormFront[1],wormFront[0]),selectWorm)
    shortenSkel = sc.make_clusters(skelList)
    try:
      img = ci.makeSkelLines(selectWorm, grayscale_matrix, shortenSkel)
    except:
      raise Exception(file)
    height, width, colors = img.shape
    h_padding = round((max_height - height)/2)
    w_padding = round((max_width - width)/2)
    empty_image[h_padding:h_padding+height,w_padding:w_padding+width] = img
    writer.write(empty_image)

  del writer

if __name__=="__main__":
  first_angle = lambda worm_matrix, grayscale_matrix: sc.getSegmentAngle(worm_matrix,grayscale_matrix,point_num=10,angle_index = 1)
  func_list = [ci.getArea, ci.getAverageShade,first_angle]
  #test = single_data("Annotated_4967","Annotated_344_469_4967.0_x1y1x2y2_909_835_966_855.png", func_list)
  #test2 = folder_data("Annotated_4967", func_list)
  start = t.time()
  #data_folder("5240", func_list)
  makeSkeleEstimate("C:/Users/cdkte/Downloads/worm_segmentation/Annotated_4967","C:/Users/cdkte/Downloads/worm_segmentation/SkeletonVids/v2/4967.avi")
  stop = t.time()
  print(str(stop - start) + " seconds passed")

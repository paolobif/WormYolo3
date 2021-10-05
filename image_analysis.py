from numpy.lib.function_base import select
import convert_image as ci
import skeleton_cleaner as sc
import numpy as np
import os
import time as t
import cv2
from matplotlib import pyplot as plt
from multiprocessing import Pool
import all_image_analysis as aia

"""
Requires Tensorflow, cv2, and Pixellib
"""

DATA_PATH = "Anno_"
MODEL_PATH = "C:/Users/cdkte/Downloads/worm_segmentation/model_folder/mask_rcnn_model.003-0.442767.h5"


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
  lambda_func = lambda func: func(worm_dict, grayscale_matrix)
  try:
    select_worm = ci.findCenterWorm(worm_dict)
    parsed_name = img_name.split("_")
    vid_id = parsed_name[1]
    frame = parsed_name[2]
    worm_id = parsed_name[3]
    x1=parsed_name[5];y1=parsed_name[6];x2=parsed_name[7];y2=parsed_name[8].split(".")[0]

    outnumpy = np.array([frame,x1,y1,x2,y2,worm_id])

    for func in function_list:
      outnumpy = np.append(outnumpy, func(select_worm, grayscale_matrix))
  except:
    print(folder_path+"/"+img_name + " is invalid")
    if len(worm_dict) != 0:
      raise Exception
    return None
  return outnumpy

def folder_data(folder_path, function_list):
  """
  Creates a matrix of data for all files in the given folder
  ---
  folder_path: The folder to be prpocessed
  function_list: The list of functions to be run on each file
  """
  files = os.listdir(folder_path)
  arr_shape = (len(files), 6+len(function_list)+10)
  percent_check = [round(len(files)/4), round(len(files)/2), round(3*len(files)/4)]
  out_arr = np.empty(arr_shape)

  for i in range(len(files)):
    out_arr[i] = aia.allSingleAnalysis(folder_path, files[i])

    if i in percent_check:
      print(str(int(i*100/len(files))) + "% complete on folder "+folder_path)

  sorted_array = out_arr[np.argsort(out_arr[:, 5])]
  sorted_array = sorted_array[np.argsort(sorted_array[:, 0])]
  return sorted_array

def save_folder(folder_path, out_file, function_list,match_dict={}):
  """
  Runs folder_data and saves it in a file
  ---
  folder_path: The folder to be processed
  out_file: The file to store the matrix in
  function_list: The list of functions to be run on each file
  """
  sorted_array = folder_data(folder_path, function_list)
  titles = functionNames(function_list,match_dict)
  np.savetxt(out_file, sorted_array, header = titles, delimiter = ",")

def functionNames(function_list,match_dict={}):
  """
  Creates an array that stores the names of the functions used in gathering Worm data
  Matches the order created in single_data
  """
  out_arr = []

  out_arr.append("Video Frame")
  out_arr.append("x1");out_arr.append("y1");out_arr.append("x2");out_arr.append("y2")
  out_arr.append("Worm ID")
  for i in range(len(function_list)):
    cur_func = function_list[i]
    if cur_func in match_dict:
      out_arr.append(match_dict[cur_func])
    elif cur_func == sc.getDiagonalNum:
      out_arr.append("Diagonals")
    elif cur_func == sc.getCmlAngle:
      out_arr.append("Cumulative Angle")
    elif cur_func == sc.getCmlDistance:
      out_arr.append("Length")
    elif cur_func == ci.getArea:
      out_arr.append("Area")
    elif cur_func == ci.getAverageShade:
      out_arr.append("Shade")
    elif cur_func == ci.getMaxWidth:
      out_arr.append("Max Width")
    elif cur_func == ci.getMidWidth:
      out_arr.append("Mid Width")
    else:
      out_arr.append("Unassigned Function")
  return ",".join(out_arr)


def isalambda(v):
  """
  Copied from Alex Martelli on
  https://stackoverflow.com/questions/3655842/how-can-i-test-whether-a-variable-holds-a-lambda
  """
  LAMBDA = lambda:0
  return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__



def makeAnnoVideo(folder_path, output_path):
  """
  Creates a video of the annotated worm as it moves
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
    try:
      selectWorm = ci.findCenterWorm(worm_dict)
    except:
      # If no worm was detected, skip this frame
      continue
    img = ci.createHighlight(selectWorm,grayscale_matrix)
    height, width, colors = img.shape
    h_padding = round((max_height - height)/2)
    w_padding = round((max_width - width)/2)
    empty_image[h_padding:h_padding+height,w_padding:w_padding+width] = img
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
    skelList = sc.lazySkeleton(selectWorm)
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

def makeSkeleVideo(folder_path, output_path):
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
    try:
      selectWorm = ci.findCenterWorm(worm_dict)
    except:
      continue

    skelList = sc.lazySkeleton(selectWorm)
    shortenSkel = sc.make_clusters(skelList)
    try:
      img = ci.makeSkelLines(selectWorm, grayscale_matrix, shortenSkel)
    except:
      continue
    height, width, colors = img.shape
    h_padding = round((max_height - height)/2)
    w_padding = round((max_width - width)/2)
    empty_image[h_padding:h_padding+height,w_padding:w_padding+width] = img
    writer.write(empty_image)

  del writer


if __name__=="__main__":
  first_angle = lambda worm_matrix, grayscale_matrix: sc.getSegmentAngle(worm_matrix,grayscale_matrix,point_num=10,angle_index = 1)
  last_angle = lambda worm_matrix, grayscale_matrix: sc.getSegmentAngle(worm_matrix,grayscale_matrix,point_num=7,angle_index = 4)
  func_list = [ci.getArea, ci.getAverageShade,sc.getCmlAngle,sc.getCmlDistance,ci.getMaxWidth,ci.getMidWidth,sc.getDiagonalNum]
  #test = single_data("Annotated_4967","Annotated_344_469_4967.0_x1y1x2y2_909_835_966_855.png", func_list)
  start = t.time()
  match_dict = {last_angle:"Last Angle"}
  #print(functionNames(func_list,match_dict))
  test2 = folder_data("C:/Users/cdkte/Downloads/Anno_50.0/Anno_50.0", func_list)
  #save_folder("Anno_5518.0","Anno_5518.0.csv", func_list,match_dict = match_dict)
  #makeSkeleEstimate("C:/Users/cdkte/Downloads/worm_segmentation/Annotated_4967","C:/Users/cdkte/Downloads/worm_segmentation/SkeletonVids/v2/4967.avi")
  stop = t.time()
  print(str(stop - start) + " seconds passed")

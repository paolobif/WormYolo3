import numpy as np
import convert_image as ci
import skeleton_cleaner as sc
import image_analysis as ia
import time as t
from matplotlib import pyplot as plt
import worm_checker as wc

#TODO: Look up MaskRCNN to see if it would be more efficient.

# Print invalid images in console
SHOW_INVALID = False

def matrixAnalysis(grayscale_matrix, worm_dict,vid_id,frame,worm_id,x1,y1,x2,y2):
  try:
    worm_matrix = ci.findCenterWorm(worm_dict)
    skel_list = sc.lazySkeleton(worm_matrix)

    area = np.sum(worm_matrix)

    # Create a matrix with only the color values of worm pixels
    mult_matrix = np.multiply(worm_matrix,grayscale_matrix).astype(float)
    #plt.imshow(mult_matrix)
    #plt.show()
    # Set all pixels that aren't worm to nan
    mult_matrix[mult_matrix==0]=np.nan
    # Take the average ignoring nan
    shade = np.nanmean(mult_matrix)


    # Create Simplified worm skeleton
    skel_simple = sc.makeFractionedClusters(skel_list, 7)
    # Take cumulative angle of simplified skeleton
    cml_angle = sc.getCmlAnglePoints(skel_simple)

    length = 0
    for i in range(0,len(skel_list)-1,1):
      length += ci.pointDistance(skel_list[i],skel_list[i+1])

    # Find Max Width
    width_point = {}
    for i in range(0,len(skel_list)):
      point = skel_list[i]
      x = point[0]; y = point[1]
      width_point[point] = estimateMaxWidth(point,worm_matrix)
    max_width = width_point[max(width_point, key=width_point.get)]

    mid_width = width_point[skel_list[round(len(skel_list)/2)]]

    # Get Diagonals
    diagonals = sc.getDiagonalNum(worm_matrix,grayscale_matrix)

    # Get simplified skeleton coordinates
    skel_simple = sc.makeFractionedClusters(skel_list, 5)
    sorted_list = sortPoints(skel_simple)

    point1_x = sorted_list[0][0];point1_y = sorted_list[0][1]
    point2_x = sorted_list[1][0];point2_y = sorted_list[1][1]
    point3_x = sorted_list[2][0];point3_y = sorted_list[2][1]
    point4_x = sorted_list[3][0];point4_y = sorted_list[3][1]
    point5_x = sorted_list[4][0];point5_y = sorted_list[4][1]

    outnumpy = np.array([frame,x1,y1,x2,y2,worm_id,area,shade,cml_angle,length,max_width,mid_width,diagonals,point1_x,point1_y,point2_x,point2_y,point3_x,point3_y,point4_x,point4_y,point5_x,point5_y])
    wc.check_worm(worm_dict, worm_matrix, outnumpy,skel_list)

  except Exception as E:
    # Show that something went wrong in console
    if SHOW_INVALID:
      print(vid_id,frame,worm_id," is invalid")
      print(E)
      """
      plt.imshow(grayscale_matrix)
      plt.show()
      plt.imshow(worm_matrix)
      plt.show()
      #"""

    # Show that the data is invalid
    outnumpy = np.array([frame,x1,y1,x2,y2,worm_id,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  return outnumpy

def filelessAnalysis(header, image_matrix):
  return allSingleAnalysis(image_matrix, header.join("_"), use_ci = False)

def allSingleAnalysis(worm_location,img_name, use_ci = True):
  """
  Collects all data from a worm image

  Args:
    worm_location: The folder that the image is located in
    img_name: The image of the worm
    use_ci: Whether to use worm_location as a file path or as a raw image

  Returns:
    outnumpy: Numpy array of data
  """
  try:
    # Get basic information about the image
    parsed_name = img_name.split("_")
    vid_id = parsed_name[1]
    frame = parsed_name[2]
    worm_id = parsed_name[3]
    x1=parsed_name[5];y1=parsed_name[6];x2=parsed_name[7];y2=parsed_name[8].split(".")[0]

    # Make the basic information: worm matrix, grayscale matrix, and skeleton
    if use_ci:
      worm_dict, grayscale_matrix = ci.getWormMatrices(worm_location+"/"+img_name)
    else:
      worm_dict, grayscale_matrix = ci.getFilelessWormMatrices(worm_location)
    worm_matrix = ci.findCenterWorm(worm_dict)
    skel_list = sc.lazySkeleton(worm_matrix)

    area = np.sum(worm_matrix)

    # Create a matrix with only the color values of worm pixels
    mult_matrix = np.multiply(worm_matrix,grayscale_matrix)
    # Set all pixels that aren't worm to nan
    mult_matrix[mult_matrix==0]=np.nan
    # Take the average ignoring nan
    shade = np.nanmean(mult_matrix)

    # Create Simplified worm skeleton
    skel_simple = sc.makeFractionedClusters(skel_list, 7)
    # Take cumulative angle of simplified skeleton
    cml_angle = sc.getCmlAnglePoints(skel_simple)

    length = 0
    for i in range(0,len(skel_list)-1,1):
      length += ci.pointDistance(skel_list[i],skel_list[i+1])

    # Find Max Width
    width_point = {}
    for i in range(0,len(skel_list)):
      point = skel_list[i]
      x = point[0]; y = point[1]
      width_point[point] = estimateMaxWidth(point,worm_matrix)
    max_width = width_point[max(width_point, key=width_point.get)]

    mid_width = width_point[skel_list[round(len(skel_list)/2)]]

    # Get Diagonals
    diagonals = sc.getDiagonalNum(worm_matrix,grayscale_matrix)

    # Get simplified skeleton coordinates
    skel_simple = sc.makeFractionedClusters(skel_list, 5)
    sorted_list = sortPoints(skel_simple)

    point1_x = sorted_list[0][0];point1_y = sorted_list[0][1]
    point2_x = sorted_list[1][0];point2_y = sorted_list[1][1]
    point3_x = sorted_list[2][0];point3_y = sorted_list[2][1]
    point4_x = sorted_list[3][0];point4_y = sorted_list[3][1]
    point5_x = sorted_list[4][0];point5_y = sorted_list[4][1]

    outnumpy = np.array([frame,x1,y1,x2,y2,worm_id,area,shade,cml_angle,length,max_width,mid_width,diagonals,point1_x,point1_y,point2_x,point2_y,point3_x,point3_y,point4_x,point4_y,point5_x,point5_y])
    wc.check_worm(worm_dict, worm_matrix, outnumpy,skel_list)

  except Exception as E:
    # Show that something went wrong in console
    if SHOW_INVALID:
      print(img_name,"is invalid")
      print(E)

    # Show that the data is invalid
    outnumpy = np.array([frame,x1,y1,x2,y2,worm_id,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  return outnumpy

def sortPoints(point_list):
  """
  Put points in a recognizable order. The first point should be closest to (0,0)

  Args:
    point_list: The list of points to sort

  Returns:
    point_list: The sorted list of points
  """
  first_distance = ci.pointDistance(point_list[0],(0,0))
  last_distance = ci.pointDistance(point_list[-1],(0,0))
  if first_distance < last_distance:
    return point_list
  else:
    point_list.reverse()
    return point_list

def estimateMaxWidth(point,worm_matrix):
  """
  Approximates maximum width at a point on the worm

  Args:
    point: The (y,x) coordinate to look for the width along
    worm_matrix: The boolean matrix showing which pixels are worm

  Returns:
    min(distances): The smallest distance along all 45-degree axes
  """
  x = point[1]; y = point[0]
  #up/down, left/right, down-right, down-left
  distances = [0,0,0,0]

  # up down
  worm_matrix = np.pad(worm_matrix,[(1,),(1,)],mode='constant')

  yTest = y
  while worm_matrix[x,yTest]:
    yTest+=1
  distances[0] = yTest - y
  if yTest!=y:
    yTest-=1

  yTest = y
  while worm_matrix[x,yTest]:
    yTest-=1
  if yTest!=y:
    yTest+=1
  distances[0] = distances[0] + y - yTest

  xTest = x
  while worm_matrix[xTest,y]:
    xTest+=1
  if xTest!=x:
    xTest-=1
  distances[1] = xTest - x

  xTest = x
  while worm_matrix[xTest,y]:
    xTest-=1
  if xTest!=x:
    xTest+=1
  distances[1] = distances[1] + x -xTest

  upDif = 0
  while worm_matrix[x+upDif,y+upDif]:
    upDif+=1
  if upDif!=0:
    upDif-=1
  distances[2] = np.sqrt(2)*upDif

  upDif = 0
  while worm_matrix[x-upDif,y-upDif]:
    upDif+=1
  if upDif!=0:
    upDif-=1
  distances[2] += np.sqrt(2)*upDif

  upDif = 0
  while worm_matrix[x+upDif,y-upDif]:
    upDif+=1
  distances[3] = np.sqrt(2)*upDif

  upDif = 0
  while worm_matrix[x-upDif,y+upDif]:
    upDif+=1
  if upDif!=0:
    upDif-=1
  distances[3] += np.sqrt(2)*upDif

  return min(distances)


if __name__ == "__main__":
  ci.makeSkelImg()
  print(allSingleAnalysis( "C:/Users/cdkte/Downloads/Mot_Single/Day10/Anno_21","Annotated_681_day10_simple_151_21_x1y1x2y2_561_745_596_760.png"))
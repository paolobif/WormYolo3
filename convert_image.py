

from matplotlib.pyplot import draw
import numpy as np
import cv2

import skeleton_cleaner as sc

# How far to look when finding the front of the worm(in wormNearPixel)
DENSITY_SEARCH = 2

# How close we expect the skeleton points to be (in checkClosePoint)
SKELETON_DELTA = 1

# How far to move along an angle in findAngle
ANGLE_DELTA = 0.05

# How many angles to check (evenly spaced between 0 and 2pi) in findAngle
ANGLE_NUM = 8

def imageToMatrices(image_path):
  """
  Turns a highlighted image of a worm into two numpy matrices

  Args:
    image_path: The highlighted image of the worm

  Returns: (grayscale_matrix, class_matrix)
    grayscale_matrix: The original version of the image
    class_matrix: The matrix separating 'Not Worm'(0) from 'Worm'(1)
  """
  img = cv2.imread(image_path)
  height, width, colors = img.shape
  grayscale_matrix = np.zeros((height, width))
  class_matrix = np.zeros((height, width))

  for y in range(height):
    for x in range(width):
      rgb_values = img[y,x]
      if (np.all(rgb_values == rgb_values[0])):
        grayscale_matrix[y,x] = rgb_values[0]
        class_matrix[y,x] = -1
      else:
        grayscale_matrix[y,x] = min(rgb_values)
        class_matrix[y,x] = 1

  return grayscale_matrix, class_matrix

def getWormMatrices(image_path):
  """
  Turns a highlighted image into it's grayscale and a dictionary of matrices that represent each individual worm.

  Args:
    image_path: The highlighted image of the worm

  Returns: (worm_dict, grayscale_matrix)
    grayscale_matrix: The original version of the image
    worm_dict: The dictionary containg each matrix separating 'Not this worm' from 'This worm'
  """

  img = cv2.imread(image_path)
  return getFilelessWormMatrices(img)


def getFilelessWormMatrices(img):
  """
  Turns a highlighted image matrix into it's grayscale and a dictionary of matrices that represent each individual worm.

  Args:
    img: The highlighted image matrix of the worm

  Returns: (worm_dict, grayscale_matrix)
    grayscale_matrix: The original version of the image
    worm_dict: The dictionary containg each matrix separating 'Not this worm' from 'This worm'

  """
  worm_dict = {}
  h, w, colors = img.shape
  grayscale_matrix = np.zeros((h, w))

  for y in range(h):
    for x in range(w):
      rgb_values = img[y,x]
      grayscale_matrix[y,x] = np.min(rgb_values)
      continu = False
      for item in rgb_values:
        if abs(item-rgb_values[0])>5:
          continu = True
      if continu:#not (np.all(rgb_values == rgb_values[0])):
        grayscale_matrix[y,x] = np.min(rgb_values)*2
        rgb_values = 10*round((int(rgb_values[0])-int(rgb_values[1]))/10)
        if rgb_values in worm_dict:
          worm_dict[rgb_values][y,x] = 1
        else:
          worm_dict[rgb_values] = np.zeros((h,w))
          worm_dict[rgb_values][y,x] = 1
  return worm_dict, grayscale_matrix

def findFront(worm_matrix, return_pixel = True):
  """
  Identifies the pixel of the worm that is around as little of the worm as possible

  Args:
    worm_matrix: The matrix of which pixels are part of the worm
    return_pixel: If True, returns the pixel near the least worm possible. If False, returns dictionary of how much worm near all pixels

  Returns:
    if return_pixel:
      A worm pixel near as few worm pixels as possible
    else:
      ret_array: The list of worm pixels with as few worm pixels near them as possible
  """
  height, width = worm_matrix.shape
  pixel_dense = {}
  for y in range(height):
    for x in range(width):
      if worm_matrix[y,x]:
        pixel_dense[(y,x)] = wormNearPixel(worm_matrix,y,x)
  if return_pixel:
    return min(pixel_dense,key = pixel_dense.get)
  else:
    ret_array = []
    min_val =  pixel_dense[min(pixel_dense,key = pixel_dense.get)]
    for item in pixel_dense:
      if pixel_dense[item] == min_val:
        ret_array.append(item)
    return ret_array

def wormNearPixel(worm_matrix, y, x, search_range = DENSITY_SEARCH):
  """
  Determines how many pixels near the given x and y are worm with a range of DENSITY_SEARCH

  Args:
    worm_matrix: The matrix showing which parts of the image are worm pixels
    y: The y coordinate for the pixel to look around
    x: The x coordinate for the pixel to look around
    search_range: How far out to look for worm pixels

  Returns:
    sumV: The total number of pixels near the (X,y) coordinate on the image that are worm pixels
  """
  height, width = worm_matrix.shape
  sumV = 0
  for i in range(round(y) - search_range, round(y) + search_range + 1):
    for j in range(round(x) - search_range, round(x) + search_range + 1):
      if (i >= 0 and i < height and j >= 0 and j < width):
        sumV += worm_matrix[i, j]
  return sumV

def moveAlongAngle(y, x, angle, distance):
  """
  Moves from (x,y) a distance at given angle.

  Args:
    y: Part of an (x,y) coordinate
    x: Part of an (x,y) coordinate
    angle: The angle to move from (x,y) towards
    distance: How far to move along the given angle

  Returns:
    (x,y) coordinate
  """
  return moveAlongAxis(y, x, np.sin(angle), np.cos(angle), distance)

def moveAlongAxis(x, y, xSlope, ySlope, distance):
  """
  Moves from (x,y) at a slope of xSlope/ySlope by the given distance

  Args:
    x: Part of an (x,y) coordinate
    y: Part of an (x,y) coordinate
    xSlope: The change in x for y/x to calculate the line to move along
    ySlope: The change in y for y/x
    distance: How far to move along y/x

  Returns:
    (x,y) coordinate
  """
  newX = x + xSlope/np.sqrt(xSlope**2+ySlope**2)*distance
  newY = y + ySlope/np.sqrt(xSlope**2+ySlope**2)*distance
  return (newX, newY)

def findAngle(x, y, worm_matrix):
  """
  Finds the line that goes through the least worm possible drawn through point (x,y)

  Args:
    x: Part of the coordinate (x,y)
    y: Part of the ecoordinate (x,y)
    worm_matrix: The matrix showing which coordinates are worm pixels

  Returns:
    An angle between 0 and 2pi
  """
  angleList = np.arange(0,np.pi,np.pi/ANGLE_NUM)
  angle_dict = {}
  for angle in angleList:
    sumV = 0
    distance = ANGLE_DELTA
    coord = moveAlongAngle(x,y,angle,distance)
    while getValue(coord, worm_matrix):
      distance+=ANGLE_DELTA
      coord = moveAlongAngle(x,y,angle,distance)
    sumV += distance

    distance = ANGLE_DELTA
    coord = moveAlongAngle(x,y,angle-np.pi,distance)
    while getValue(coord, worm_matrix):
      distance+=ANGLE_DELTA
      coord = moveAlongAngle(x,y,angle-np.pi,distance)
    sumV += distance

    angle_dict[angle] = sumV


  return(min(angle_dict, key=angle_dict.get))

def findAngleDict(x, y, worm_matrix):
  """
  Finds the line that goes through the least worm possible drawn through point (x,y)

  Args:
    x: (x,y) coordinate
    y: (x,y) coordinate
    worm_matrix: Matrix showing which pixels are part of the worm

  Returns:
    angle_dict: Describes how much worm each angle through the point goes through
  """
  angleList = np.arange(0,np.pi,np.pi/ANGLE_NUM)
  angle_dict = {}
  for angle in angleList:
    sumV = 0
    distance = ANGLE_DELTA
    coord = moveAlongAngle(x,y,angle,distance)
    while getValue(coord, worm_matrix):
      distance+=ANGLE_DELTA
      coord = moveAlongAngle(x,y,angle,distance)
    sumV += distance

    distance = ANGLE_DELTA
    coord = moveAlongAngle(x,y,angle-np.pi,distance)
    while getValue(coord, worm_matrix):
      distance+=ANGLE_DELTA
      coord = moveAlongAngle(x,y,angle-np.pi,distance)
    sumV += distance

    angle_dict[angle] = sumV
  return angle_dict


def pointDistance(coord1, coord2):
  """
  Determines the euclidean distance between (x,y) of coord1 and (x,y) of coord2

  Args:
    coord1: (x,y) coordinate
    coord2: (x,y) coordinate

  Returns:
    Float value of the distance between the two coordinates
  """
  return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)


def checkClosePoint(coord,point_list):
  """
  Checks whether a coordinate is too close to any of the points in point_list

  Args:
    coord: (x,y) coordinate
    point_list: List of coordinates

  Returns:
    Boolean whether the point is too close to another point
  """
  copy_list = point_list.copy()

  if coord in point_list:
    copy_list.remove(coord)

  for point in copy_list:
    if pointDistance(coord, point) < SKELETON_DELTA/2:
      return True
  return False

def getValue(coord, grayscale):
  """
  Gets the shade of worm at (x,y) and returns 0 if (x,y) is not in the matrix

  Args:
    coord: (x,y) coordinate
    grayscale: The grayscale matrix of the worm

  Returns:
    0-255 integer of the shade of the worm at the given coordinate
  """
  try:
    return grayscale[round(coord[1]),round(coord[0])]
  except:
    return 0

def makeImg(data):
  """
  Turns a grayscale matrix into a [r,g,b] matrix

  Args:
    data: The grayscale matrix

  Returns:
    returnMatrix: The [r,g,b] version of the grayscale matrix
  """
  width, height = np.shape(data)
  returnMatrix = np.zeros((width, height, 3), int)
  for i in range(width):
    for j in range(height):
      curVal = data[i,j]
      for k in range(3):
        returnMatrix[i, j, k] = curVal
  return returnMatrix

def findCenterWorm(worm_dict):
  """
  Finds the worm that is nearest the middle

  Args:
    worm_dict: The dictionary describing which worms are where on the image.

  Return:
    worm_matrix: The matrix showing which pixels are part of the worm in the middle of the image
  """
  dist_dict = {}
  if worm_dict == {}:
    raise Exception("No worms detected")
  for key in worm_dict:
    cur_worm = worm_dict[key]

    radius = 0
    while not checkRadius(cur_worm, radius):
      radius+=1
      if radius > cur_worm.shape[0] and radius > cur_worm.shape[1]:
        break

    dist_dict[radius] = key
  return worm_dict[dist_dict[min(dist_dict)]]

def checkRadius(worm_matrix, radius):
  """
  Looks in the circumference given radius to see if any of the pixels are 'worm'.

  Args:
    worm_matrix: The matrix describing which pixels are part of the worm
    radius: How far to look for worm pixels from the center

  Returns:
    Boolean whether there are worm pixels in that radius
  """
  height, width = worm_matrix.shape
  midy, midx = height/2, width/2
  cosV = np.arange(0,2*np.pi, 0.1)
  for item in cosV:
    x = int(midx+np.round(radius*np.cos(item)))
    y = int(midy+np.round(radius*np.sin(item)))
    try:
      if worm_matrix[y, x] == 1:
        return True
    except:
      pass
  return False

def createHighlight(worm_matrix, grayscale_matrix):
  """
  Creates a highlighted [r,g,b] image of a worm

  Args:
    worm_matrix: The matrix determining worm from not worm
    grayscale_matrix: The grayscale version of the image

  Returns:
    returnMatrix: The grayscale matrix but with color added to worm pixels
  """
  width, height = np.shape(grayscale_matrix)
  returnMatrix = np.zeros((width, height, 3), int)
  for i in range(width):
    for j in range(height):
      curVal = grayscale_matrix[i,j]
      for k in range(3):
        returnMatrix[i, j, k] = curVal
      if worm_matrix[i,j] == 1:
        returnMatrix[i, j, 0] = returnMatrix[i, j, 0] + 50
  return returnMatrix

def makeSkelImg(worm_matrix, grayscale_matrix, point_list):
  """
  Creates a rgb image where the worm is highlighted and points are a different color
  Used to show skeleton

  Args:
    worm_matrix: The matrix that says what is worm
    grayscale_matrix: The grayscale image
    point_list: The series of points to place on the worms image

  Returns:
    rgb: The rgb matrix highlighting worm pixels
  """
  rgb = createHighlight(worm_matrix,grayscale_matrix)
  for point in point_list:
    rgb[round(point[1])][round(point[0])][1] += 100
  return rgb

def makeSkelLines(worm_matrix, grayscale_matrix, point_list):
  """
  Makes an rgb matrix where the worm is highlighted and the skeleton is clearly marked as lines

  Args:
    worm_matrix: The matrix that says what is worm
    grayscale_matrix: The grayscale image
    point_list: The series of points to connect with lines in the image

  Returns:
    rgb: The rgb matrix with both the worm highlighted and pixels along the skeleton highlighted again
  """
  rgb = createHighlight(worm_matrix,grayscale_matrix)
  for i in range(len(point_list)-1):
    point1 = point_list[i]
    point2 = point_list[i+1]
    xSlope = -(point1[0]-point2[0])
    ySlope = -(point1[1]-point2[1])
    draw_on = point_list[i]
    distance = 1

    while not (round(draw_on[0]) == round(point2[0]) and round(draw_on[1]) == round(point2[1])):
      draw_on = moveAlongAxis(point1[0],point1[1],xSlope,ySlope,distance)
      rgb[round(draw_on[1])][round(draw_on[0])][1] += 100

      distance+=0.01

  return rgb

def makeWormOutline(worm_matrix):
  """
  Creates an outline of the worm

  Args:
      worm_matrix (Numpy array): The array showing which pixels are worm

  Returns:
      Numpy array: Matrix showing which pixels lie on the border of the worm
  """
  copy_matrix = worm_matrix.copy()
  # For each point
  for i in range(len(worm_matrix)):
    for j in range(len(worm_matrix[i])):
      coord = (j,i)
      if worm_matrix[i,j]==1:
        if not checkIfNotEdge(coord, worm_matrix):
          copy_matrix[i,j] = 0
  return copy_matrix

def checkIfNotEdge(coord, worm_matrix):
  """
  Checks whether a certain coordinate is on the edge of the worm

  Args:
    coord: (x,y) pixel
    worm_matrix: The matrix showing which pixel is part of the worm
  Returns:
    Boolean if the coordinate is on the edge
  """
  y = coord[1]
  x = coord[0]
  # Check each side
  if getValue((x,y+1),worm_matrix) and getValue((x,y-1),worm_matrix) and getValue((x+1,y),worm_matrix) and getValue((x-1,y),worm_matrix):
    return False
  else:
    return True

"""
Data Handling Functions

These were condensed into all_image_analysis to run faster and remove redundancies,
but are left here in case theyneed to be used specifically.

Accepts input in the form worm_matrix, grayscale_matrix
"""

def getArea(worm_matrix, grayscale_matrix=None):
  """
  Calculates the amount of the image that is worm
  """
  area=0
  for row in worm_matrix:
    for column in row:
      if column==1:
        area+=1
  return area

def getAverageShade(worm_matrix, grayscale_matrix):
  """
  Calculates the average shade of the pixels that are worm
  """
  height, width = worm_matrix.shape
  endArray=[]
  for y in range(height):
    for x in range(width):
      if worm_matrix[y,x]==1:
        endArray.append(grayscale_matrix[y,x])
  return np.mean(endArray)

def getMaxWidth(worm_matrix,grayscale_matrix, path=None):
  """
  Finds the maximum width of the provided worm
  """
  #wormFront = findFront(worm_matrix)
  #skelList = createMiddleSkeleton((wormFront[1],wormFront[0]),worm_matrix)
  #skelList = sc.betterMiddleSkel(worm_matrix)
  skelList = sc.fastMiddleSkel(worm_matrix)
  width_point = {}
  # Find Max Width
  for i in range(len(skelList)):
    point = skelList[i]
    x = point[0]; y = point[1]

    angle_dict = findAngleDict(x,y,worm_matrix)
    width = angle_dict[min(angle_dict,key=angle_dict.get)]
    width_point[point] = width

  return width_point[max(width_point, key=width_point.get)]

def getMidWidth(worm_matrix,grayscale_matrix, path=None):
  """
  Finds the width of the provided worm in the approximate middle
  """
  skelList = sc.lazySkeleton(worm_matrix)
  width_point = {}
  # Find Mid Width
  point = skelList[round(len(skelList)/2)]
  x = point[0]; y = point[1]

  angle_dict = findAngleDict(x,y,worm_matrix)
  width = angle_dict[min(angle_dict,key=angle_dict.get)]
  width_point[point] = width

  return width_point[max(width_point, key=width_point.get)]


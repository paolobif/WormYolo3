try:
  from matplotlib import pyplot as plt
  from matplotlib import patches
  GRAPHING = True
except:
  GRAPHING = False

from matplotlib.pyplot import draw
import numpy as np
import cv2

import time as t
import sys

import skeleton_cleaner as sc


DENSITY_SEARCH = 2
SEARCH_DELTA = 1.5
ANGLE_DELTA = 0.05
SKELETON_DELTA = 1
ANGLE_NUM = 8

def imageToMatrices(image_path):
  """
  Turns a highlighted image of a worm into two matrices

  image_path: The highlighted image of the worm

  grayscale_matrix: The original version of the image
  class_matrix: The matrix separating 'Not Worm' from 'Worm'
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
  Turns a highlighted image into it's grayscale and matrices that represent each individual worm.

  image_path: The highlighted image of the worm
  grayscale_matrix: The original version of the image
  worm_dict: The dictionary containg each matrix separating 'Not this worm' from 'This worm'
  """
  worm_dict = {}
  img = cv2.imread(image_path)
  h, w, colors = img.shape
  grayscale_matrix = np.zeros((h, w))
  #print(img)

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

def getArea(worm_matrix, grayscale_matrix=None, worm_path=None):
  """
  Calculates the amount of the image that is worm
  """
  area=0
  for row in worm_matrix:
    for column in row:
      if column==1:
        area+=1
  return area

def getAverageShade(worm_matrix, grayscale_matrix, worm_path=None):
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

def findFront(worm_matrix):
  """
  Identifies the pixel of the worm that is around as little of the worm as possible
  """
  height, width = worm_matrix.shape
  pixel_dense = {}
  for y in range(height):
    for x in range(width):
      if worm_matrix[y,x]:
        pixel_dense[(y,x)] = wormNearPixel(worm_matrix,y,x)
  return min(pixel_dense,key = pixel_dense.get)

def wormNearPixel(worm_matrix, y, x):
  """
  Determines how many pixels near the given x and y are worm
  """
  height, width = worm_matrix.shape
  sumV = 0
  for i in range(round(y) - DENSITY_SEARCH, round(y) + DENSITY_SEARCH + 1):
    for j in range(round(x) - DENSITY_SEARCH, round(x) + DENSITY_SEARCH + 1):
      if (i >= 0 and i < height and j >= 0 and j < width):
        sumV += worm_matrix[i, j]
  return sumV

def moveAlongAngle(x, y, angle, distance):
  """
  Moves from (x,y) a distance at given angle.
  """
  return moveAlongAxis(x, y, np.sin(angle), np.cos(angle), distance)

def moveAlongAxis(x, y, xSlope, ySlope, distance):
  """
  Moves from (x,y) at a slope of xSlope/ySlope by the given distance
  """
  newX = x + xSlope/np.sqrt(xSlope**2+ySlope**2)*distance
  newY = y + ySlope/np.sqrt(xSlope**2+ySlope**2)*distance
  return (newX, newY)

def findFarthestAngle(x, y, worm_matrix):
  angleList = np.arange(-np.pi,np.pi,np.pi/ANGLE_NUM)
  angle_dict = {}
  for angle in angleList:
    coord = moveAlongAngle(x,y,angle,SEARCH_DELTA/2)
    angle_dict[angle] = wormNearPixel(worm_matrix, coord[0],coord[1])

  return(min(angle_dict, key=angle_dict.get))
def findAngle(x, y, worm_matrix):
  """
  Finds the line that goes through the least worm possible drawn through point (x,y)
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




def createSkeleton(start_coord, worm_matrix):
  """
  Creates a general middle line through the worm.
  This is done by finding the line through the least of the worm and moving perpendicular to that.
  This was discarded due to bad results on low quality images

  start_coord: The point of the worm where to start
  worm_matrix: The matrix determining what is/is not worm
  """
  point_list = [start_coord]
  cur_coord = start_coord

  ortho_angle = findAngle(cur_coord[0], cur_coord[1], worm_matrix)
  coord1 = moveAlongAngle(cur_coord[0], cur_coord[1], ortho_angle + np.pi/2, 1)

  # Determine which path takes us towards the worm
  if getValue(coord1, worm_matrix):
    prev_angle = ortho_angle + np.pi/2
  else:
    prev_angle = ortho_angle - np.pi/2


  while getValue(cur_coord, worm_matrix):
    ortho_angle = findAngle(cur_coord[0], cur_coord[1], worm_matrix)
    if abs(ortho_angle + np.pi/2 - prev_angle) < abs(ortho_angle - np.pi/2 - prev_angle):
      cur_coord = moveAlongAngle(cur_coord[0], cur_coord[1], ortho_angle + np.pi/2, SKELETON_DELTA)
    else:
      cur_coord = moveAlongAngle(cur_coord[0], cur_coord[1], ortho_angle - np.pi/2, SKELETON_DELTA)
    point_list.append(cur_coord)
  return point_list

def pointDistance(coord1, coord2):
  """
  Determines the euclidean distance between (x,y) of coord1 and (x,y) of coord2
  """
  return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def createMiddleSkeleton(start_coord, worm_matrix):
  """
  Creates a skeleton generally through the middle.
  This is done by finding the angle that aims toward as much worm as possible.

  start_coord: The point of the worm where to start
  worm_matrix: The matrix determining what is/is not worm

  point_list: The list of points in form (x,y) that form the middle line.
  """
  point_list = [start_coord]
  cur_coord = start_coord

  ortho_angle = findAngle(cur_coord[0], cur_coord[1], worm_matrix)
  coord1 = moveAlongAngle(cur_coord[0], cur_coord[1], ortho_angle + np.pi/2, SKELETON_DELTA)
  # Determine which path takes us towards the worm
  if getValue(coord1, worm_matrix):
    prev_angle = ortho_angle + np.pi/2
  else:
    prev_angle = ortho_angle - np.pi/2

  if not getValue(moveAlongAngle(cur_coord[0], cur_coord[1], prev_angle, SKELETON_DELTA),worm_matrix):
    prev_angle = getAngleMax(cur_coord, ortho_angle, worm_matrix)
    prev_angle = getAngleMax(cur_coord, prev_angle, worm_matrix)
    prev_angle = getAngleMax(cur_coord, prev_angle, worm_matrix)
  if not getValue(moveAlongAngle(cur_coord[0], cur_coord[1], prev_angle, SKELETON_DELTA),worm_matrix):
    for angle in np.arange(0,2*np.pi,np.pi/4):
      if getValue(moveAlongAngle(cur_coord[0], cur_coord[1], angle, SKELETON_DELTA),worm_matrix):
        prev_angle = angle
        break

  while getValue(cur_coord, worm_matrix):
    towards_angle = getAngleMax(cur_coord, prev_angle, worm_matrix)
    cur_coord = moveAlongAngle(cur_coord[0], cur_coord[1], towards_angle, SKELETON_DELTA)
    prev_angle = towards_angle
    point_list.append(cur_coord)
    if checkClosePoint(cur_coord,point_list):
      break


  point_list = sc.cleanSmallLoops(point_list)
  return point_list

def checkClosePoint(coord,point_list):
  """
  Checks whether a coordinate is too close to any of the points in point_list
  """
  copy_list = point_list.copy()

  if coord in point_list:
    copy_list.remove(coord)

  for point in copy_list:
    if pointDistance(coord, point) < SKELETON_DELTA/2:
      return True
  return False

def getAngleMax(coord, prev_angle, worm_matrix):
  """
  Finds the angle that leads toward as much worm as possible.
  coord: The coord we are moving away from
  prev_angle: The previous angle, limits the possibilities to 45 degrees in either direction.
  worm_matrix: The matrix that determines what is and isn't worm

  cur_angle: The angle that points toward as much worm as possible.
  """
  # Look 90 degrees around previous angle
  angleList = np.arange(prev_angle-np.pi/4,np.pi/4+prev_angle+np.pi/ANGLE_NUM,np.pi/ANGLE_NUM)
  angle_dict = {}
  for angle in angleList:
    coordNew = moveAlongAngle(coord[0],coord[1],angle, SEARCH_DELTA)
    if getValue(coordNew, worm_matrix):
      angle_dict[angle] = wormNearPixel(worm_matrix, coordNew[1],coordNew[0])
  try:
    max_size = angle_dict[max(angle_dict, key=angle_dict.get)]
    cur_angle = max(angle_dict, key=angle_dict.get)
  except:
    cur_angle=prev_angle
  for angle in angle_dict:
    if angle_dict[angle] == max_size:
      if abs(prev_angle - cur_angle) > abs(prev_angle - angle):
        cur_angle = angle
  return cur_angle

def getValue(coord, worm_matrix):
  """
  Gets the shade of worm at (x,y)
  """
  try:
    return worm_matrix[round(coord[1]),round(coord[0])]
  except:
    return 0

def makeImg(data):
  """
  Turns a grayscale matrix into a [r,g,b] matrix
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

  worm_dict: The dictionary describing which worms are where on the image.
  """
  dist_dict = {}
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
  worm_matrix: The matrix determining worm from not worm
  grayscale_matrix: The grayscale version of the image
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

  worm_matrix: The matrix that says what is worm
  grayscale_matrix: The grayscale image
  point_list: The series of points
  """
  rgb = createHighlight(worm_matrix,grayscale_matrix)
  for point in point_list:
    rgb[round(point[1])][round(point[0])][1] += 100
  return rgb

def makeSkelLines(worm_matrix, grayscale_matrix, point_list):
  """
  Makes an rgb matrix where the worm is highlighted and the skeleton is clearly marked as lines

  worm_matrix: The matrix that says what is worm
  grayscale_matrix: The grayscale image
  point_list: The series of points to connect with lines in the image
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
  #wormFront = findFront(worm_matrix)
  #skelList = createMiddleSkeleton((wormFront[1],wormFront[0]),worm_matrix)
  #skelList = sc.betterMiddleSkel(worm_matrix)
  skelList = sc.fastMiddleSkel(worm_matrix)
  width_point = {}
  # Find Mid Width
  point = skelList[round(len(skelList)/2)]
  x = point[0]; y = point[1]

  angle_dict = findAngleDict(x,y,worm_matrix)
  width = angle_dict[min(angle_dict,key=angle_dict.get)]
  width_point[point] = width

  return width_point[max(width_point, key=width_point.get)]

def makeWormOutline(worm_matrix):
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
  y = coord[1]
  x = coord[0]
  height, width = worm_matrix.shape
  # Check each side
  if getValue((x,y+1),worm_matrix) and getValue((x,y-1),worm_matrix) and getValue((x+1,y),worm_matrix) and getValue((x-1,y),worm_matrix):
    return False
  else:
    return True

if __name__ == "__main__":
  #worm_dict, grayscale_matrix = getWormMatrices("C:/Users/cdkte/Downloads/worm_segmentation/Anno_5240/Annotated_344_532_5240.0_x1y1x2y2_1011_153_1053_179.png")
  #worm_dict2, grayscale_matrix2 = getWormMatrices("C:/Users/cdkte/Downloads/worm_segmentation/Anno_5240/Annotated_344_535_5240.0_x1y1x2y2_1014_151_1053_177.png")
  #worm_dict, grayscale_matrix = getWormMatrices("C:/Users/cdkte/Downloads/worm_segmentation/Anno_5018/Annotated_344_477_5018.0_x1y1x2y2_1132_408_1188_431.png")
  #worm_dict2, grayscale_matrix2 = getWormMatrices("C:/Users/cdkte/Downloads/worm_segmentation/Anno_5018/Annotated_344_478_5018.0_x1y1x2y2_1133_408_1188_429.png")
  worm_dict, grayscale_matrix = getWormMatrices("C:/Users/cdkte/Downloads/yolo3/Worm-Yolo3/Anno_5515.0/Annotated_344_1083_5515.0_x1y1x2y2_927_477_962_514.png")
  #worm_dict, grayscale_matrix = getWormMatrices("C:/Users/cdkte/Downloads/yolo3/Worm-Yolo3/Anno_5518.0/Annotated_344_838_5518.0_x1y1x2y2_722_404_746_446.png")
  worm_dict, grayscale_matrix = getWormMatrices("C:/Users/cdkte/Downloads/yolo3/Worm-Yolo3/Day4/Anno_2/Annotated_681_day4_simple_2_2_x1y1x2y2_562_276_606_298.png")
  worm_dict, grayscale_matrix = getWormMatrices("C:/Users/cdkte/Downloads/yolo3/Worm-Yolo3/Day10/Anno_10/Annotated_681_day10_simple_9_10_x1y1x2y2_440_781_462_828.png")

  selectWorm = findCenterWorm(worm_dict)
  print(worm_dict)
  #selectWorm2 = findCenterWorm(worm_dict2)

  #print(getArea(selectWorm))
  #print(getAverageShade(selectWorm,grayscale_matrix))
  POINT_NUM = 7
  wormFront = findFront(selectWorm)
  skelList = createMiddleSkeleton((wormFront[1],wormFront[0]),selectWorm)
  skelList = sc.fastMiddleSkel(selectWorm)
  shortenSkel = sc.makeFractionedClusters(skelList,5)
  #wormFront2 = findFront(selectWorm2)
  #skelList2 = createMiddleSkeleton((wormFront2[1],wormFront2[0]),selectWorm2)
  #shortenSkel2 = sc.makeFractionedClusters(skelList2,POINT_NUM)
  print("width",getMaxWidth(selectWorm,grayscale_matrix))
  print(sc.getCmlAngle(selectWorm,grayscale_matrix))
  """
  plt.plot(x,y)
  plt.plot(x,y2)
  plt.show()
  """

  plt.imshow(makeSkelLines(selectWorm, grayscale_matrix, shortenSkel))
  i=0
  for item in skelList:
    plt.plot(item[0],item[1],'bo',markersize=3)
    i+=1
  for item in shortenSkel:
    plt.plot(item[0],item[1],'bo',markersize=9)
    i+=1

  outline_matrix = makeWormOutline(selectWorm)
  #plt.imshow(createHighlight(outline_matrix,grayscale_matrix))
  #print(sc.getDiagonalNum(selectWorm,grayscale_matrix))
  '''
  plt.imshow(selectWorm)
  plt.plot(wormFront[1], wormFront[0],'bo')
  #print(moveAlongAngle(17,18,np.pi/4,1))
  #print(findAngle(wormFront[1], wormFront[0],selectWorm))
  skelList = createMiddleSkeleton((wormFront[1],wormFront[0]),selectWorm)
  #skelList = createSkeleton((wormFront[1],wormFront[0]),selectWorm)
  i=1
  for item in skelList:
    plt.plot(item[0],item[1],'bo',markersize=i)
    #i-=1
  '''
  plt.show()
  '''
  plt.imshow(makeSkelLines(selectWorm2, grayscale_matrix2, shortenSkel2))
  i=0
  for item in skelList2:
    plt.plot(item[0],item[1],'bo',markersize=3)
    i+=1
  for item in shortenSkel2:
    plt.plot(item[0],item[1],'bo',markersize=9)
    i+=1
  #plt.show()
  '''

  #"""
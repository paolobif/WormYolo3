from numpy.core.fromnumeric import take
from numpy.lib.function_base import select
import convert_image as ci
import numpy as np
import time as t
try:
    from matplotlib import pyplot as plt
except:
  pass

MAX_SIZE = 10
MAX_DISTANCE = 10
SMALLEST_LOOP = 20
SEARCH_DELTA = 1.5
SKELETON_DELTA = 1
ANGLE_DELTA = 0.05
ANGLE_NUM = 2

class cluster():
  """
  A cluster of points that keeps track of cumulative distance and angle even as it reduces the number of points
  Should keep the same general shape
  """
  def __init__(self,point_list,size=1,cml_distance=0,cml_angle = 0):
    self.point_list=point_list
    self.size = size
    self.cml_distance = cml_distance
    self.cml_angle = cml_angle
  def __str__(self):
    return str(self.point_list)
  def midpoint(self):
    """
    Returns an (x,y) that is equally distant from either end of the cluster, moving along points in the cluster
    """
    mes_distance = 0
    i = 1
    point= self.point_list[0]
    prev_point = point
    while mes_distance < self.cml_distance/2:
      point = self.point_list[i]
      mes_distance += ci.pointDistance(prev_point,point)
      i+=1
    return point
  def connection(self,other_point):
    """
    Determines whether two clusters can form a valid connection
    ---
    other_point: The following cluster
    """
    total_size = self.size + other_point.size
    total_distance = self.cml_distance + other_point.cml_distance
    total_distance += ci.pointDistance(self.point_list[-1], other_point.point_list[0])
    if total_size <= MAX_SIZE and total_distance <= MAX_DISTANCE:
      return True
    else:
      return False
  def total_distance(self, other_cluster):
    """
    Finds the total distance two clusters cover
    The first point of the next cluster will be linked to the last point of this cluster

    other_cluster: The cluster to attach at the end of this one
    """
    return self.cml_distance + other_cluster.cml_distance + ci.pointDistance(self.point_list[-1], other_cluster.point_list[0])
  def connect(self, other_point):
    """
    Creates and returns a new cluster with the new point added to the end of this cluster
    """
    total_size = self.size + other_point.size
    point_list = self.point_list + other_point.point_list

    total_distance = self.cml_distance + other_point.cml_distance + ci.pointDistance(self.point_list[-1], other_point.point_list[0])

    return cluster(point_list,total_size,total_distance)

def makeFractionedClusters(point_list, cluster_num):
  """
  Makes a number of points evenly distributed throughout the distance

  Args:
    point_list: A list of points
    cluster_num: The number of points to reduce to

  Returns:
    return_list: a new list of points with cluster_num points
  """
  cluster_list = []
  first_point = point_list[0]
  max_cluster = cluster([first_point])
  for point in point_list:
    if not point == first_point:
      max_cluster = max_cluster.connect(cluster([point]))
    cluster_list.append(cluster([point]))
  try:
    section_length = round(max_cluster.cml_distance) / (cluster_num - 1)
  except:
    section_length = 0

  new_list = []
  prev_cluster = cluster_list[0]
  clusterV = cluster_list[0]
  for i in range(1,len(cluster_list)):
    clusterV = cluster_list[i]
    if prev_cluster.total_distance(clusterV) >= section_length:

      new_list.append(prev_cluster.connect(clusterV))
      i += 1
      if i < len(cluster_list):
        prev_cluster = cluster_list[i]

    prev_cluster = prev_cluster.connect(clusterV)
  if len(new_list) < cluster_num - 1:
    new_list.append(prev_cluster.connect(clusterV))
  return_list = []
  for item in new_list:
    return_list.append(item.point_list[0])
  try:
    return_list.append(new_list[-1].point_list[-1])
  except:
    raise Exception
  return return_list

def cleanSmallLoops(point_list):
  """
  Removes small groupings of points from the point_list and returns the list

  Args:
    point_list: A list of points

  Returns:
    point_list: The list with any small loops at the end closed
  """

  #Find the point that starts the circle
  for i in range(len(point_list)):

    point1 = point_list[i]
    if ci.checkClosePoint(point1,point_list):
      break
  point_dict = {}

  #Look for the most distant point
  for j in range(i,len(point_list)):
    point_dict[j] = ci.pointDistance(point1,point_list[j])
  indx = max(point_dict, key=point_dict.get)

  #Remove the points following that distant point
  if indx - i < SMALLEST_LOOP:

    return point_list[0:indx]
  else:
    return point_list

def make_numbered_clusters(point_list, cluster_num):
  """
  Makes a number of clusters from the given points by connecting the smallest clusters first

  Args:
    point_list: The points to seperate into clusters
    cluster_num: The number of points to finish with

  Returns:
    return_list: the points in the cluster
  """
  cluster_list = []
  for point in point_list:
    cluster_list.append(cluster([point]))
  while len(cluster_list) > cluster_num:
    clust_dist = {}
    for i in range(len(cluster_list)-1):
      cluster1 = cluster_list[i]
      cluster2 = cluster_list[i+1]
      clust_dist[i] = cluster1.total_distance(cluster2)
    shortest = min(clust_dist, key = clust_dist.get)
    new_list = []
    i = 0
    while i < len(cluster_list):
      if i!=shortest:
        new_list.append(cluster_list[i])
      else:
        new_list.append(cluster_list[i].connect(cluster_list[i+1]))
        i+=1
      i+=1
    cluster_list = new_list
  return_list = []
  for item in cluster_list:
    return_list.append(item.midpoint())
  return return_list

def make_clusters(point_list):
  """
  Makes clusters of points while matching the constants at the top of this file

  Args:
    point_list: A list of points

  Returns:
    return_list: A list of clusters matching the constants
  """
  cluster_list = []
  for point in point_list:
    cluster_list.append(cluster([point]))

  new_list = []
  while not new_list==cluster_list:
    if new_list:
      cluster_list = new_list.copy()
    new_list = []
    for i in range(0,len(cluster_list),2):
      if len(cluster_list) == i+1:
        new_list.append(cluster_list[i])
      elif cluster_list[i].connection(cluster_list[i+1]):
        new_list.append(cluster_list[i].connect(cluster_list[i+1]))
      else:
        new_list.append(cluster_list[i])
        new_list.append(cluster_list[i+1])

  return_list = []
  for item in cluster_list:
    return_list.append(item.midpoint())
  return return_list

def getAngle(point1,point2,point3):
  """
  Finds the angle offset formed by three points
  To be more precise:
  If a line was drawn with the slope from point1 to point2 centered on (0,0)
  and another line was drawn from point2 to point3 centered on (0,0) you would then look
  at the difference of angles formed by those two lines.
  This makes this a good measure of how much the slope *changed* rather than
  the actual angle at those 3 points.

  Args:
    point1: The endpoint of the first line segment
    point2: The connecting point where the line segments meet
    point3: The endpoint of the second line segment

  Returns:
    The float angle between the points
  """
  x1 = point1[0]; y1 = point1[1]
  x2 = point2[0]; y2 = point2[1]
  x3 = point3[0]; y3 = point3[1]
  if x1==x2:
    angle1 = np.pi/2
  else:
    slope1 = (y1-y2)/(x1-x2)
    angle1 = np.arctan(slope1)
  if x2==x3:
    angle2 = np.pi/2
  else:
    slope2 = (y2-y3)/(x2-x3)
    angle2 = np.arctan(slope2)
  return abs(abs(angle1) - abs(angle2))

def getCmlAnglePoints(point_list):
  """
  Finds how much the angle must have changed while moving along the points in point_list

  Args:
    point_list: A list of points

  Returns:
    sumV: The float sum of the angles along the point_list
  """
  sumV = 0
  for i in range(len(point_list)-2):
    sumV += getAngle(point_list[i],point_list[i+1],point_list[i+2])
  return sumV

def findNextEdge(cur_point, not_list, outline_matrix):
  """
  Finds an adjacent point that isn't on the not_list and is in the outline_matrix

  Args:
    cur_point: An (x,y) point in outline_matrix
    not_list: A list of (x,y) points that are not acceptable return values
    outline_matrix: The matrix showing which pixels are on the edge of the worm

  Returns: [(x,y),0 or 1]
    (x,y) is thet next point
    0 or 1 is whether that point is an acceptable value to return
  """
  cur_x = cur_point[0]
  cur_y = cur_point[1]
  for x in range(cur_x-1,cur_x+2):
    for y in range(cur_y-1, cur_y+2):
      if ci.getValue((x,y),outline_matrix):
        if not ((x,y) in not_list):
          if x!=cur_x and y!=cur_y:
            return [(x,y),1]
          else:
            return [(x,y),0]
  return [None,0]

def lazySkeleton(worm_matrix):
  """
  Currently, the most accurate method for creating a skeleton of the given worm.
  Effectively does the same as betterMiddleSkel except sticking solely to pi/4 angles
  and integer coordinates.

  Finds the most isolated point on the worm and uses that as the 'front'.
    Determines which combination of (1,0),(0,1),(1,1),(1,-1) goes through the least worm pixels
    Moves pi/2 radians from that excluding the option closest to the prior angle.
    Repeat until not on worm

  Args:
    worm_matrix: The matrix describing which pixels are part of the worm

  Returns:
    point_list: The list of points that form the skeleton
  """
  height, width = worm_matrix.shape
  pad = False
  if np.any(worm_matrix[0,:]) or np.any(worm_matrix[:,0]) or np.any(worm_matrix[:,-1]) or np.any(worm_matrix[-1,:]):
    worm_matrix = np.pad(worm_matrix,[(1,),(1,)],mode='constant')
    pad = True

  first_point = ci.findFront(worm_matrix)
  bad_angle = badAngle(first_point,worm_matrix)
  point_list = []
  if min(bad_angle) == bad_angle[0]:
    prior_direct = 2
    next_point = (first_point[0]+1,first_point[1])
    if not worm_matrix[next_point[0],next_point[1]]:
      prior_direct = 6
      next_point = (first_point[0]-1,first_point[1])
  elif min(bad_angle) == bad_angle[1]:
    prior_direct = 0
    next_point = (first_point[0],first_point[1]+1)
    if not worm_matrix[next_point[0],next_point[1]]:
      prior_direct = 4
      next_point = (first_point[0],first_point[1]-1)
  elif min(bad_angle) == bad_angle[2]:
    prior_direct = 3
    next_point = (first_point[0]+1,first_point[1]-1)
    if not worm_matrix[next_point[0],next_point[1]]:
      prior_direct = 7
      next_point = (first_point[0]-1,first_point[1]+1)
  elif min(bad_angle) == bad_angle[3]:
    prior_direct = 1
    next_point = (first_point[0]+1,first_point[1]+1)
    if not worm_matrix[next_point[0],next_point[1]]:
      prior_direct = 5
      next_point = (first_point[0]-1,first_point[1]-1)
  point_list.append(next_point)
  reverse = False
  while worm_matrix[next_point[0],next_point[1]]:
    direction_list = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
    distances = badAngle(next_point,worm_matrix)
    direction = translateDistances(distances,prior_direct,next_point,worm_matrix,point_list)

    prior_direct = direction
    change = direction_list[direction]
    next_point = (next_point[0]+change[0],next_point[1]+change[1])
    next_point = getMiddlePoint(next_point,direction,worm_matrix)
    if next_point in point_list:
      if len(point_list) > 5:
        break
      else:
        if worm_matrix[next_point[0]+1,next_point[1]]:
          next_point = (next_point[0]+1,next_point[1])
        elif worm_matrix[next_point[0]-1,next_point[1]]:
          next_point = (next_point[0]-1,next_point[1])
        elif worm_matrix[next_point[0],next_point[1]+1]:
          next_point = (next_point[0],next_point[1]+1)
        elif worm_matrix[next_point[0],next_point[1]-1]:
          next_point = (next_point[0],next_point[1]-1)
        else:
          break
    if not worm_matrix[next_point[0],next_point[1]] and len(point_list) < 5 and not reverse:
      prior_direct = prior_direct + 4
      if prior_direct > 7:
        prior_direct-=7
      reverse = True
      next_point = point_list[0]
      point_list = []
    else:
      point_list.append(next_point)

  point_list = [(point[1],point[0]) for point in point_list]
  if pad:
    copy_list = []
    for point in point_list:
      copy_list.append((point[0]-1,point[1]-1))
    point_list = copy_list
    for point in point_list:
      if point[0] >= width or point[1] >= height:
        point_list.remove(point)

  return point_list

def getMiddlePoint(point,direction,worm_matrix):
  """
  Shifts a point to be as much in the skeleton as possible while moving as little as possible

  Args
    point: The point to move to the middle of the worm
    direction: The direction moved to reach this point
    worm_matrix: The matrix showing which pixels are part of the worm

  Returns:
    (x,y) coordinate
  """

  test_change = [0,0]
  direction2 = np.argmin(badAngle(point,worm_matrix))

  if direction2 == 0:
    test_change[1] = 1
  elif direction2 == 1:
    test_change[0] = 1
  elif direction2 == 3:
    test_change[0] = 1; test_change[1] = -1
  elif direction2 == 2:
    test_change[0] = 1; test_change[1] = 1
  test_point = [point[0],point[1]]
  change_sum = 0
  while worm_matrix[test_point[0],test_point[1]]:
    test_point[0] += test_change[0]; test_point[1] += test_change[1]
    change_sum += 1

  test_point = [point[0],point[1]]
  while worm_matrix[test_point[0],test_point[1]]:
    test_point[0] -= test_change[0]; test_point[1] -= test_change[1]
    change_sum -= 1
  change_sum = int(change_sum/2)
  return_point = (point[0] + test_change[0]*change_sum, point[1] + test_change[1]*change_sum)
  return return_point

def intersection(list1,list2):
  """
  Gives the values that are both in list1 and list2

  Args:
      list1: A list
      list2: A list

  Returns:
      The list of items in both lists
  """
  return [direction for direction in list1 if direction in list2]
  # That... should work?

def translateDistances(distance_list,prior_direction,next_point,worm_matrix,point_list):
  """
  Determines which possible directions could be possible to move in based on inputs

  Args:
    distance_list: A list of values with the index representing a direction and the value representing how 'good' of a path this is
    prior_direction: The direction moved in previously
    next_point: The point to move from
    worm_matrix: The matrix describing what is 'worm' and what isn't
    point_list: The list of all previous points

  Returns:
    A direction in which to move in based on an evenly segmented circle
  """
  possible_directions=[]
  if distance_list[0]==min(distance_list):
    possible_directions.extend([2,6])
  if distance_list[1]==min(distance_list):
    possible_directions.extend([0,4])
  if distance_list[3]==min(distance_list):
    possible_directions.extend([1,5])
  if distance_list[2]==min(distance_list):
    possible_directions.extend([3,7])

  limited_directions = []
  for i in range(-2,3):
    v =prior_direction+i
    if v<0:
      v+=8
    if v>7:
      v-=8
    limited_directions.append(v)
  select_direct = intersection(possible_directions,limited_directions)
  backup = []
  for item in select_direct:
    direction_list = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
    test_change = direction_list[item]
    if not worm_matrix[next_point[0]+test_change[0],next_point[1]+test_change[1]]:
      select_direct.remove(item)
      backup.append(item)

  if len(select_direct) == 0:
    return backup[0]

  elif len(select_direct) > 1:
    direction_list = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
    point_distance = {}
    for item in select_direct:
      test_change = direction_list[item]
      coord1 = (next_point[0]+test_change[0],next_point[1]+test_change[1])
      test_change = direction_list
      sumv=0
      for coord2 in point_list:
        sumv+=ci.pointDistance(coord1,coord2)
      point_distance[item] = sumv
    return max(point_distance,key=point_distance.get)
  return select_direct[0]

def badAngle(point,worm_matrix):
  """
  Determines which angle to move in would be the worst by moving on the x, y, xy, and yx axes and seeing how many pixels it goes through
  before being outside of the worm.

  Args:
    point: A (x,y) point on the worm
    worm_matrix: The matrix describing which pixels are part of the worm

  Returns:
    A list of values that say how many pixels lie on each axis
  """
  x = point[0]; y = point[1]
  #up/down, left/right, down-right, down-left
  distances = [0,0,0,0]

  # up down
  yTest = y
  while worm_matrix[x,yTest]:
    yTest+=1

  distances[0] = yTest - y
  yTest = y
  while worm_matrix[x,yTest]:
    yTest-=1

  distances[0] = distances[0] + y - yTest

  xTest = x
  while worm_matrix[xTest,y]:
    xTest+=1

  distances[1] = xTest - x
  xTest = x
  while worm_matrix[xTest,y]:
    xTest-=1

  distances[1] = distances[1] + x -xTest

  upDif = 0
  while worm_matrix[x+upDif,y+upDif]:
    upDif+=1


  distances[2] = np.sqrt(2)*upDif
  upDif = 0
  while worm_matrix[x-upDif,y-upDif]:
    upDif+=1

  distances[2] += np.sqrt(2)*upDif


  upDif = 0
  while worm_matrix[x+upDif,y-upDif]:
    upDif+=1
  distances[3] = np.sqrt(2)*upDif
  upDif = 0
  while worm_matrix[x-upDif,y+upDif]:
    upDif+=1
  distances[3] += np.sqrt(2)*upDif

  distances[0] = distances[0]-1
  distances[1] = distances[1] - 1
  distances[2] = distances[2] - np.sqrt(2)
  distances[3] = distances[3] - np.sqrt(2)

  return distances

"""
Data Handling Functions

These were condensed into all_image_analysis to run faster and remove redundancies,
but are left here in case theyneed to be used specifically.

Accepts input in the form worm_matrix, grayscale_matrix
"""

def getCmlAngle(worm_matrix, grayscale_matrix, worm_path=None, point_num = 7):
  """
  Segments the worm into point_num clusters, then finds the total change in slope.

  point_num: The number of points to create for taking the cumulative angle
  """
  skelList = lazySkeleton(worm_matrix)
  skelSimple = makeFractionedClusters(skelList, point_num)
  return getCmlAnglePoints(skelSimple)

def getCmlDistance(worm_matrix, grayscale_matrix, worm_path=None):
  """
  Segments the worm into point_num clusters, then finds the total change in slope.
  """
  wormFront = ci.findFront(worm_matrix)
  skelList = lazySkeleton(worm_matrix)
  sum = 0
  for i in range(0,len(skelList)-1,1):
    sum += ci.pointDistance(skelList[i],skelList[i+1])
  return sum

def getDiagonalNum(worm_matrix, grayscale_matrix):
  """
  Finds the number of diagonal movements made along the edge of the worm
  """
  outline_matrix = ci.makeWormOutline(worm_matrix)
  # Find starting point
  height, width = outline_matrix.shape
  continu = False
  for y in range(height):
    for x in range(width):
      if ci.getValue((x,y),outline_matrix):
        continu = True
        break
    if continu:
      break
  point_list = [(x,y)]
  cur_point = (x,y)
  curvature = 0
  while True:
    if cur_point == None:
      break
    else:
      cur_point, dCurvature = findNextEdge(cur_point,point_list,outline_matrix)
      point_list.append(cur_point)
      curvature += dCurvature
  return curvature

def getSegmentAngle(worm_matrix, grayscale_matrix, point_num = 10, angle_index = 0):
  """
  Segments the worm into point_num clusters, then finds the change in slope at the given angle.
  If angle_index is 0, it takes the first angle: The one formed by the first, second, and third points.

  point_num = The number of points to take
  angle_index = Which angle we should look at
  """
  skelList = lazySkeleton(worm_matrix)
  skelSimple = makeFractionedClusters(skelList, point_num)
  return getAngle(skelSimple[angle_index],skelSimple[angle_index+1],skelSimple[angle_index+2])


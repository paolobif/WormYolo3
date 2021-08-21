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
  def __init__(self,point_list,size=1,cml_distance=0,cml_angle = 0):
    self.point_list=point_list
    self.size = size
    self.cml_distance = cml_distance
    self.cml_angle = cml_angle
  def __str__(self):
    return str(self.point_list)
  def midpoint(self):
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
    """
    return self.cml_distance + other_cluster.cml_distance + ci.pointDistance(self.point_list[-1], other_cluster.point_list[0])
  def connect(self, other_point):
    total_size = self.size + other_point.size
    point_list = self.point_list + other_point.point_list

    total_distance = self.cml_distance + other_point.cml_distance + ci.pointDistance(self.point_list[-1], other_point.point_list[0])
    return cluster(point_list,total_size,total_distance)

def makeFractionedClusters(point_list, cluster_num):
  """
  Makes a number of points evenly distributed throughout the distance
  """
  cluster_list = []
  first_point = point_list[0]
  max_cluster = cluster([first_point])
  for point in point_list:
    if not point == first_point:
      max_cluster = max_cluster.connect(cluster([point]))
    cluster_list.append(cluster([point]))
  section_length = round(max_cluster.cml_distance) / (cluster_num - 1)
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
  ---
  point_list: The points to seperate into clusters
  cluster_num: The number of points to finish with

  Returns the points in the cluster
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
  Makes clusters of points while matching the CONSTANTs at the top of this file
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
  ---
  point1: The endpoint of the first line segment
  point2: The connecting point where the line segments meet
  point3: The endpoint of the second line segment
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
  """
  sumV = 0
  for i in range(len(point_list)-2):
    sumV += getAngle(point_list[i],point_list[i+1],point_list[i+2])
  return sumV

def getSegmentAngle(worm_matrix, grayscale_matrix, point_num = 10, angle_index = 0):
  """
  Segments the worm into point_num clusters, then finds the change in slope at the given angle.
  If angle_index is 0, it takes the first angle: The one formed by the first, second, and third points.
  """
  #wormFront = ci.findFront(worm_matrix)
  #skelList = ci.createMiddleSkeleton((wormFront[1],wormFront[0]),worm_matrix)
  skelList = fastMiddleSkel(worm_matrix)
  skelSimple = makeFractionedClusters(skelList, point_num)
  return getAngle(skelSimple[angle_index],skelSimple[angle_index+1],skelSimple[angle_index+2])

def getCmlAngle(worm_matrix, grayscale_matrix, worm_path=None, point_num = 5):
  """
  Segments the worm into point_num clusters, then finds the total change in slope.
  """
  #wormFront = ci.findFront(worm_matrix)
  #skelList = ci.createMiddleSkeleton((wormFront[1],wormFront[0]),worm_matrix)
  skelList = fastMiddleSkel(worm_matrix)
  #skelList = betterMiddleSkel(worm_matrix)
  skelSimple = makeFractionedClusters(skelList, point_num)
  return getCmlAnglePoints(skelSimple)
def getCmlDistance(worm_matrix, grayscale_matrix, worm_path=None):
  """
  Segments the worm into point_num clusters, then finds the total change in slope.
  """
  wormFront = ci.findFront(worm_matrix)
  skelList = ci.createMiddleSkeleton((wormFront[1],wormFront[0]),worm_matrix)
  skelList = fastMiddleSkel(worm_matrix)
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
def findNextEdge(cur_point, not_list, outline_matrix):
  """
  Finds an adjacent point that isn't on the not_list and is in the outline_matrix
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

def findMidpoint(x, y, worm_matrix):
  """
  Given a point on the worm, finds the midpoint across the smallest cross-line
  """
  best_angle = ci.findAngle(x,y, worm_matrix)
  distance = 0.05
  d1 = distance
  coord = ci.moveAlongAngle(x,y,best_angle,distance)
  while ci.getValue(coord, worm_matrix):
    d1+=distance
    coord = ci.moveAlongAngle(x,y,best_angle,d1)

  d2 = distance
  distance = 0.05
  coord = ci.moveAlongAngle(x,y,best_angle-np.pi,d2)
  while ci.getValue(coord, worm_matrix):
    d2 += distance
    coord = ci.moveAlongAngle(x,y,best_angle-np.pi,d2)
  avg_distance = (d1 - d2)/2
  mid_point = ci.moveAlongAngle(x,y,best_angle,avg_distance)
  return mid_point

def betterMiddleSkel(worm_matrix):
  """
  Creates an accurate middle line. Takes substantially more time
  """
  front_coord = ci.findFront(worm_matrix) # Remember findFront returns (y,x)
  front_coord = findMidpoint(front_coord[1],front_coord[0],worm_matrix)
  front_coord = (front_coord[1], front_coord[0])
  point_list = []
  prev_angle = findWorstAngle(front_coord[1],front_coord[0],worm_matrix)
  new_coord = ci.moveAlongAngle(front_coord[1],front_coord[0],prev_angle,SKELETON_DELTA)

  while not ci.checkClosePoint(new_coord,point_list) and ci.getValue(new_coord,worm_matrix):
    point_list.append(new_coord)
    newWorstRange = np.arange(prev_angle-np.pi/8,prev_angle+np.pi/8,np.pi/4/ANGLE_NUM)
    prev_angle = findWorstAngle(new_coord[0],new_coord[1],worm_matrix,newWorstRange)
    new_coord = ci.moveAlongAngle(new_coord[0],new_coord[1],prev_angle,SKELETON_DELTA)
    new_coord = findMidpoint(new_coord[0],new_coord[1],worm_matrix)
    if len(point_list) > 100:
      print(len(point_list))
  return point_list


def findWorstAngle(x,y,worm_matrix,angleList=np.arange(0,2*np.pi,np.pi/ANGLE_NUM)):
  angle_dict = {}
  for angle in angleList:
    sumV = 0
    distance = ANGLE_DELTA
    coord = ci.moveAlongAngle(x,y,angle,distance)
    while ci.getValue(coord, worm_matrix):
      distance+=ANGLE_DELTA
      coord = ci.moveAlongAngle(x,y,angle,distance)
    sumV += distance

    angle_dict[angle] = round(sumV,4)
  return(max(angle_dict, key=angle_dict.get))

def fastMiddleSkel(worm_matrix):
  point_list = []
  for i in range(len(worm_matrix)):
    row = worm_matrix[i]
    on_worm = False
    for j in range(len(row)):
      if row[j] == 1:
        if on_worm:
          on_count += 1
        else:
          on_worm = True
          on_count = 1
      if row[j] != 1 and on_worm:
        point_list.append((j-on_count/2,i))
        on_worm = False
  #Iterate through columns instead of rows
  wm_t = worm_matrix.transpose()
  point_list2=[]
  for i in range(len(wm_t)):
    row = wm_t[i]
    on_worm = False
    for j in range(len(row)):
      if row[j] == 1:
        if on_worm:
          on_count += 1
        else:
          on_worm = True
          on_count = 1
      if row[j] != 1 and on_worm:
        point_list2.append((i,j-on_count/2))
        on_worm = False
  if len(point_list)<len(point_list2):
    point_list = point_list2
    del point_list2
    take_vert = 0
  else:
    take_vert = 1
  # Place in order
  mid_point = getMiddle(point_list)
  new_point_list = [mid_point]
  point_list.remove(mid_point)
  left_lim = np.inf; left_point = None
  right_lim = np.inf; rightPoint = None
  for point in point_list:
    if point[take_vert] < new_point_list[0][take_vert]:
      if left_lim > ci.pointDistance(point,new_point_list[0]):
        left_lim = ci.pointDistance(point,new_point_list[0])
        left_point = point
    else:
       if right_lim > ci.pointDistance(point,new_point_list[0]):
        right_lim = ci.pointDistance(point,new_point_list[0])
        right_point = point
  if right_point:
    new_point_list.insert(0,right_point); point_list.remove(right_point)
  if left_point:
    new_point_list.append(left_point); point_list.remove(left_point)
  while len(point_list) > 0:
    left_lim = np.inf; left_point = None
    right_lim = np.inf; rightPoint = None
    for point in point_list:
      if left_lim > ci.pointDistance(point,new_point_list[-1]):
        left_lim = ci.pointDistance(point,new_point_list[-1])
        left_point = point
      if right_lim > ci.pointDistance(point,new_point_list[0]):
        right_lim = ci.pointDistance(point,new_point_list[0])
        right_point = point
    #if ci.pointDistance()
    if right_point!=left_point:
      if ci.pointDistance(right_point,new_point_list[0]) <= ci.pointDistance(right_point,new_point_list[-1]):
        new_point_list.insert(0,right_point); point_list.remove(right_point)
      if ci.pointDistance(left_point,new_point_list[0]) >= ci.pointDistance(left_point,new_point_list[-1]):
        new_point_list.append(left_point); point_list.remove(left_point)

    else:
      if ci.pointDistance(right_point,new_point_list[0]) <= ci.pointDistance(right_point,new_point_list[-1]):
        new_point_list.insert(0,right_point); point_list.remove(right_point)
      else:
        new_point_list.append(right_point); point_list.remove(right_point)


  return new_point_list

def getMiddle(point_list):
  """
  Gets the point that is as close to as many other point as possible
  """
  point_dict = {}
  for point in point_list:
    sum = 0
    for point2 in point_list:
      if point!=point2:
        sum+=ci.pointDistance(point,point2)
    point_dict[point] = sum
  return min(point_dict, key=point_dict.get)



if __name__ == "__main__":

  #worm_dict, grayscale_matrix = ci.getWormMatrices("C:/Users/cdkte/Downloads/yolo3/Worm-Yolo3/Anno_5518.0/Annotated_344_856_5518.0_x1y1x2y2_720_413_738_467.png")
  #worm_dict, grayscale_matrix = ci.getWormMatrices("C:/Users/cdkte/Downloads/worm_segmentation/Annotated_4967/Annotated_344_688_4967.0_x1y1x2y2_919_834_944_848.png")
  #worm_dict, grayscale_matrix = ci.getWormMatrices("C:/Users/cdkte/Downloads/yolo3/Worm-Yolo3/Anno_5515.0/Annotated_344_1078_5515.0_x1y1x2y2_925_476_960_514.png")
  worm_dict, grayscale_matrix = ci.getWormMatrices("C:/Users/cdkte/Downloads/yolo3/Worm-Yolo3/Anno_6158.0/Annotated_346_611_6158.0_x1y1x2y2_762_369_784_424.png")


  selectWorm = ci.findCenterWorm(worm_dict)
  plt.imshow(ci.createHighlight(selectWorm, grayscale_matrix))
  #plt.plot(20,47,'o')


  start = t.time()
  wormFront = ci.findFront(selectWorm)
  skelList = ci.createMiddleSkeleton((wormFront[1],wormFront[0]),selectWorm)
  stop = t.time()
  print(start-stop)
  start = t.time()
  next_p = fastMiddleSkel(selectWorm)
  stop = t.time()
  print(start-stop)
  clustP = makeFractionedClusters(next_p,5)
  msv = 0.5
  for coord in next_p:
    plt.plot(coord[0],coord[1],'go',ms=msv)
    msv+=0.2
  msv = 1.0
  for coord in clustP:
    plt.plot(coord[0],coord[1],'ro',ms=msv)
    msv+=1
  plt.legend(loc='upper left', fontsize='xx-small')
  print(getCmlAngle(selectWorm,grayscale_matrix))
  plt.show()

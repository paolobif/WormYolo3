import convert_image as ci
import numpy as np

MAX_SIZE = 10
MAX_DISTANCE = 10

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

def make_numbered_clusters(point_list, cluster_num):
  """
  Makes a number of clusters from the given points by connecting the smallest clusters first
  ---
  point_list: The points to seperate into clusters
  cluster_num: The number of points to finish with
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
  return abs(angle1 - angle2)

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
  wormFront = ci.findFront(worm_matrix)
  skelList = ci.createMiddleSkeleton((wormFront[1],wormFront[0]),worm_matrix)
  skelSimple = make_numbered_clusters(skelList, point_num)
  return getAngle(skelSimple[angle_index],skelSimple[angle_index+1],skelSimple[angle_index+2])

def getCmlAngle(worm_matrix, grayscale_matrix, worm_path=None, point_num = 10):
  """
  Segments the worm into point_num clusters, then finds the total change in slope.
  """
  wormFront = ci.findFront(worm_matrix)
  skelList = ci.createMiddleSkeleton((wormFront[1],wormFront[0]),worm_matrix)
  skelSimple = make_numbered_clusters(skelList, point_num)
  return getCmlAnglePoints(skelSimple)


if __name__ == "__main__":
  test = make_numbered_clusters([(-1,-3),(0,0),(1,2),(2,3),(3,4),(10,30),(11,29)],2)
  print(test)
  print(getCmlAnglePoints([(0,0),(1,1),(0,2),(1,3)]))


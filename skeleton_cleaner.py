import convert_image as ci

MAX_SIZE = 10
MAX_DISTANCE = 10

class cluster():
  def __init__(self,point,size=1,cml_distance=0,cml_angle = 0):
    self.point=point
    self.size = size
    self.cml_distance = cml_distance
    self.cml_angle = cml_angle
  def connection(self,other_point):
    total_size = self.size + other_point.size
    total_distance = self.cml_distance + other_point.cml_distance
    total_distance += ci.pointDistance(self.point, other_point.point)
    if total_size <= MAX_SIZE and total_distance <= MAX_DISTANCE:
      return True
    else:
      return False
  def connect(self, other_point):
    total_size = self.size + other_point.size
    avg_x = (self.point[0]*self.size+other_point.point[0]*other_point.size)/total_size
    avg_y = (self.point[1]*self.size+other_point.point[1]*other_point.size)/total_size
    avg_point = (avg_x, avg_y)

    total_distance = self.cml_distance + other_point.cml_distance
    return cluster(avg_point,total_size,total_distance)

def make_clusters(point_list):
  cluster_list = []
  for point in point_list:
    cluster_list.append(cluster(point))

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
    return_list.append(item.point)
  return return_list




if __name__ == "__main__":
  test = make_clusters([(1,2),(2,3),(3,4),(10,30)])
  print(test)


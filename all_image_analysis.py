import numpy as np
import convert_image as ci
import skeleton_cleaner as sc
import image_analysis as ia
import time as t

def allSingleAnalysis(folder_path,img_name):
  try:
    parsed_name = img_name.split("_")
    vid_id = parsed_name[1]
    frame = parsed_name[2]
    worm_id = parsed_name[3]
    x1=parsed_name[5];y1=parsed_name[6];x2=parsed_name[7];y2=parsed_name[8].split(".")[0]

    worm_dict, grayscale_matrix = ci.getWormMatrices(folder_path+"/"+img_name)
    worm_matrix = ci.findCenterWorm(worm_dict)
    skel_list = sc.lazySkeleton(worm_matrix)



    area = np.sum(worm_matrix)



    mult_matrix = np.multiply(worm_matrix,grayscale_matrix)
    mult_matrix[mult_matrix==0]=np.nan
    shade = np.nanmean(mult_matrix)



    skel_simple = sc.makeFractionedClusters(skel_list, 7)
    cml_angle = sc.getCmlAnglePoints(skel_simple)



    length = 0
    for i in range(0,len(skel_list)-1,1):
      length += ci.pointDistance(skel_list[i],skel_list[i+1])


    width_point = {}
    # Find Max Width
    start = t.time()
    for i in range(0,len(skel_list)):
      point = skel_list[i]
      x = point[0]; y = point[1]
      width_point[point] = estimateMaxWidth(point,worm_matrix)
      """
      angle_dict = ci.findAngleDict(x,y,worm_matrix)
      width = angle_dict[min(angle_dict,key=angle_dict.get)]
      width_point[point] = width
      """



    max_width = width_point[max(width_point, key=width_point.get)]

    mid_width = width_point[skel_list[round(len(skel_list)/2)]]

    diagonals = sc.getDiagonalNum(worm_matrix,grayscale_matrix)


    outnumpy = np.array([frame,x1,y1,x2,y2,worm_id,area,shade,cml_angle,length,max_width,mid_width,diagonals])
  except:
    print(img_name,"is invalid")
    outnumpy = np.array([frame,x1,y1,x2,y2,worm_id,0,0,0,0,0,0])
  return outnumpy

def estimateMaxWidth(point,worm_matrix):
  x = point[1]; y = point[0]
  #up/down, left/right, down-right, down-left
  distances = [0,0,0,0]

  # up down
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


if __name__=="__main__":
  func_list = [ci.getArea, ci.getAverageShade,sc.getCmlAngle,sc.getCmlDistance,ci.getMaxWidth,ci.getMidWidth,sc.getDiagonalNum]
  start = t.time()
  test = ia.single_data("C:/Users/cdkte/Downloads/yolo3/Worm-Yolo3/Anno_27172.0","Annotated_206_2224_27172.0_x1y1x2y2_1352_683_1373_725.png", func_list)
  stop = t.time()
  test2 = allSingleAnalysis("C:/Users/cdkte/Downloads/yolo3/Worm-Yolo3/Anno_27172.0","Annotated_206_2224_27172.0_x1y1x2y2_1352_683_1373_725.png")
  stop2 = t.time()
  print("Original 1 Picture")
  print("time",stop-start)
  print(test)
  print("\nNew 1 Picture")
  print("time",stop2-stop)
  print(test2)

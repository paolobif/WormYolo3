import re
import csv
import numpy as np
import convert_image as ci
import traja as trj
from matplotlib import pyplot as plt
import skeleton_cleaner as sc
import os
import tf_best_fit as bst_fit

# Number of different datas that are taken
VALUE_SIZE = 17

# Plot motion of worm
SHOW_MOTION = False

def readCsv(csv_path):
  reader = csv.reader(open(csv_path), delimiter = ',')
  i = 0
  for item in reader:
    if i == 0:
      header = item
      i+=1
    elif i == 1:
      data = np.array(item)
      i+=1
    else:
      data = np.vstack((data,item))



  csv_path = csv_path.replace("\\","/")
  find_day = csv_path.split("/")[-1]
  day_section = find_day.split("day")[1]
  day = ""
  i = 0
  while day_section[i] in [str(x) for x in range(10)]:
    day+=day_section[i]
    i+=1

  return day,header,data

def procCsvData(day,header,data):
  for i in range(len(header)):
    if header[i] == "Worm ID":
      id_head = i


  out_data = np.array([0]*VALUE_SIZE)
  cur_id = False

  for row in data:
    if row[id_head] == cur_id:
      for i in range(len(row)):
        cur_data[i].append(float(row[i]))
    else:
      if cur_id:
        out_data = np.vstack((out_data,getValuesFrom(day,header,cur_data)))
      cur_id = row[id_head]
      cur_data = []
      for value in row:
        cur_data.append([float(value)])

  # Add final worm
  out_data = np.vstack((out_data,getValuesFrom(day,header,cur_data)))

  out_data = out_data[1:,:]
  return out_data
def getValuesFrom(day,header,cur_data):

  for i in range(len(header)):
    if header[i] == "Worm ID":
      id_head = i
    elif header[i] == "Area":
      area_head = i
    elif header[i] == "Shade":
      shade_head = i
    elif header[i] == "Cumulative Angle":
      cml_angle_head = i
    elif header[i] == "Length":
      l_head = i
    elif header[i] == "Max Width":
      mxw_head = i
    elif header[i] == "Mid Width":
      mdw_head = i
    elif header[i] == "Diagonals":
      dgl_head = i
    elif header[i] == "x1":
      x1_head = i
    elif header[i] == "x2":
      x2_head = i
    elif header[i] == "y1":
      y1_head = i
    elif header[i] == "y2":
      y2_head = i
    elif header[i] == "# Video Frame" or header[i] == "Video Frame":
      frame_head = i

  avg_area = sum(cur_data[area_head])/len(cur_data[area_head])
  avg_shade = sum(cur_data[shade_head])/len(cur_data[shade_head])
  angle_variance = np.var(cur_data[cml_angle_head])
  avg_length = sum(cur_data[l_head])/len(cur_data[l_head])
  avg_mxw = sum(cur_data[mxw_head])/len(cur_data[mxw_head])
  avg_mdw = sum(cur_data[mdw_head])/len(cur_data[mdw_head])
  diagonal_variance = np.var(cur_data[dgl_head])

  # Approximate total traveled distance
  # TODO: Threshold of change (<2.5?)
  total_travel = calcTravel(cur_data[x1_head],cur_data[y1_head],cur_data[x2_head],cur_data[y2_head])

  travel_speed = total_travel / (cur_data[0][-1] - cur_data[0][0])

  # TODO: Traja... stuff?
  x_arr = []
  y_arr = []
  time_arr = []
  point_arr = []
  for i in range(len(cur_data[x1_head])):
    x_arr.append((cur_data[x1_head][i] + cur_data[x2_head][i])/2)
    y_arr.append((cur_data[y1_head][i] + cur_data[y2_head][i])/2)
    time_arr.append(cur_data[frame_head][i])
    point_arr.append([x_arr[-1],y_arr[-1]])

  # TODO: Check if able to set noise of movement (extended or smaller groupings of motion)
  df = trj.TrajaDataFrame({'x':x_arr,'y':y_arr,'time':time_arr})
  df_derivs = df.traja.get_derivatives()


  if SHOW_MOTION:
    plt.plot(df['x'],df['y'])

  avg_accel = np.nanmean(df_derivs["acceleration"])
  max_accel = np.nanmax(df_derivs["acceleration"])
  max_speed = np.nanmax(df_derivs["speed"])
  avg_speed = np.nanmean(df_derivs["speed"])

  point0 = ((cur_data[x1_head][0] + cur_data[x2_head][0])/2, (cur_data[y1_head][0] + cur_data[y2_head][0])/2)
  last_point = ((cur_data[x1_head][-1] + cur_data[x2_head][-1])/2, (cur_data[y1_head][-1] + cur_data[y2_head][-1])/2)
  rvr_sinuosity_index = total_travel / ci.pointDistance(point0,last_point)
  #try:
  clustP = sc.makeFractionedClusters(point_arr,8)
  if SHOW_MOTION:
    x, y = zip(*clustP)
    plt.scatter(x,y)
    plt.show()

  sinuosity_index = trajSinIndex(clustP)

  #except:
  #  return np.array([float(day),cur_data[id_head][0],avg_area,avg_shade,angle_variance,avg_length,avg_mxw,avg_mdw,diagonal_variance,total_travel,travel_speed,avg_speed,max_speed,max_accel,avg_accel,rvr_sinuosity_index,sinuosity_index])

  return np.array([float(day),cur_data[id_head][0],avg_area,avg_shade,angle_variance,avg_length,avg_mxw,avg_mdw,diagonal_variance,total_travel,travel_speed,avg_speed,max_speed,max_accel,avg_accel,rvr_sinuosity_index,sinuosity_index])

def trajSinIndex(point_list):
  step_length = []
  turn_angle = []
  for i in range(len(point_list)-1):
    step_length.append(ci.pointDistance(point_list[i],point_list[i+1]))
    if i != 0:
      turn_angle.append(sc.getAngle(point_list[i-1],point_list[i],point_list[i+1]))
  sigma = np.nanstd(np.array(turn_angle))
  q = np.nanmean(np.array(step_length))
  return 1.18 * sigma / np.sqrt(q)

def calcTravel(x1s,y1s,x2s,y2s):
  """!!!!!Find something better for this!!!!!"""
  total_distance = 0
  for i in range(len(x1s)-1):
    travel_dif = ci.pointDistance(((x1s[i]+x2s[i])/2,(y1s[i]+y2s[i])/2),((x1s[i+1]+x2s[i+1])/2,(y1s[i+1]+y2s[i+1])/2))
    if travel_dif > 2:
      total_distance+=travel_dif
  return total_distance

def combineDays(csv_list):
  total_data = np.array([0]*VALUE_SIZE)
  for csv_path in csv_list:
    (day, header, data) = readCsv(csv_path)
    usable_data = procCsvData(day,header,data)
    total_data = np.vstack((total_data,usable_data))
  # Get rid of init zeros
  total_data = total_data[1:,:]

  # Sort by day
  total_data = total_data[np.argsort(total_data[:, 0])]


  return total_data

def direct_folder(folder_path):
  all_files = os.listdir(folder_path)
  file_list = []
  for file in all_files:
    if file.split('.')[-1] == 'csv':
      file_list.append(folder_path+"/"+file)
  return combineDays(file_list)

def create_csv(train_folder,file_out):
  data_set = direct_folder(train_folder)
  data_set = np.delete(data_set,1,axis=1)
  list_size = data_set.shape[0]
  i = 0
  while i < list_size-1:
    row = data_set[i]
    if np.any(np.isnan(row)):
      data_set = np.delete(data_set,i,axis=0)
      i -= 1
      list_size -= 1
    i += 1
  np.savetxt(file_out,data_set,delimiter=",")

def line_fit(train_folder, test_folder):
  # TODO: Principal Component Analysis
  data_set = direct_folder(train_folder)
  list_size = data_set.shape[0]
  i = 0
  while i < list_size-1:
    row = data_set[i]
    if np.any(np.isnan(row)):
      data_set = np.delete(data_set,i,axis=0)
      i -= 1
      list_size -= 1
    i += 1

  x_train = data_set[:,2:VALUE_SIZE]
  y_train = data_set[:,0]

  test_set = direct_folder(test_folder)
  i = 0
  list_size = test_set.shape[0]
  while i < list_size-1:
    row = test_set[i]
    if np.any(np.isnan(row)):
      test_set = np.delete(test_set,i,axis=0)
      i -= 1
      list_size -= 1
    i += 1
  x_test = test_set[:,2:VALUE_SIZE]
  y_test = test_set[:,0]

  mean_label = y_train.mean(axis=0)
  std_label = y_train.std(axis=0)

  mean_feat = x_train.mean(axis=0)
  mean_feat = np.nanmean(x_train,axis=0)

  std_feat = np.nanstd(x_train,axis=0)

  x_train_norm = (x_train-mean_feat)/std_feat
  y_train_norm = (y_train-mean_label)/std_label
  linear_model = bst_fit.SimpleLinearRegression(0)
  linear_model.train(x_train_norm, y_train_norm, learning_rate=0.1, epochs=50)

  # standardize
  x_test = (x_test-mean_feat)/std_feat
  # reverse standardization
  pred = linear_model.predict(x_test)
  pred *= std_label
  pred += mean_label
  plt.plot(y_test,pred,'bo')

  plt.show()

if __name__ == "__main__":
  #(day, header, data) = readCsv("C:/Users/cdkte/Downloads/line_data/641_day12.csv")
  #usable_data = procCsvData(day,header,data)
  #print(usable_data)
  #print(combineDays(["C:/Users/cdkte/Downloads/line_data/641_day12.csv","C:/Users/cdkte/Downloads/line_data/641_day8.csv"]))
  #print(direct_folder("C:/Users/cdkte/Downloads/days"))
  #line_fit("C:/Users/cdkte/Downloads/days/Train","C:/Users/cdkte/Downloads/days/Test")
  create_csv("C:/Users/cdkte/Downloads/days/Train","days4to12.csv")
import numpy as np
import cv2
import pandas as pd
import postproc_summary as proc
import os
import traja as trj
import convert_image as ci
import skeleton_cleaner as sc


def calc_motion_data(file_path,do_wh = False):
  data = None
  if not do_wh:
    data = pd.read_csv(file_path,names=["frame", "x1", "y1", "x2", "y2","wormID"])
  else:
    data = pd.read_csv(data,names=["frame", "x1", "y1", "w", "h","wormID"])
    data["x2"] = data["x1"] + data["w"]; data["y2"] = data["y1"] + data["h"]
  header = np.array("worm_id,first_frame,last_frame,travel_speed,avg_accel,max_accel,avg_speed,max_speed,total_travel,rvr_sinuosity_index,sinuosity_index".split(","))
  all_outputs = None
  worm_ids = data["wormID"].unique()
  for worm_id in worm_ids:
    cur_data = data.where(data["wormID"]==worm_id).dropna()


    x1s = cur_data["x1"].tolist()
    x2s = cur_data["x2"].tolist()
    y1s = cur_data["y1"].tolist()
    y2s = cur_data["y2"].tolist()
    frames = cur_data["frame"].tolist()

    first_frame = frames[0]
    last_frame = frames[-1]

    #print(x1s)
    total_travel = proc.calcTravel(x1s,y1s,x2s,y2s)
    if frames[-1] > frames[0]:
      travel_speed = total_travel/(frames[-1]-frames[0])
    else:
      travel_speed = np.nan

    xs = np.mean(np.array([x1s,x2s]),axis=0)
    ys =  np.mean(np.array([y1s,y2s]),axis=0)


    df = trj.TrajaDataFrame({'x':xs,'y':ys,'time':frames})
    df_derivs = df.traja.get_derivatives()
    #print(df_derivs["acceleration"],"IsNan",np.all(np.isnan(df_derivs[["acceleration"]])))
    if not np.all(np.isnan(df_derivs["acceleration"])):
      avg_accel = np.nanmean(df_derivs["acceleration"])
      max_accel = np.nanmax(df_derivs["acceleration"])
      max_speed = np.nanmax(df_derivs["speed"])
      avg_speed = np.nanmean(df_derivs["speed"])
    else:
      avg_accel = np.nan
      max_accel = np.nan
      max_speed = np.nan
      avg_speed = np.nan

    first_point = (xs[0],ys[0]); last_point = (xs[-1],ys[-1])
    if ci.pointDistance(first_point,last_point)!=0:
      rvr_sinuosity_index = np.divide(total_travel,ci.pointDistance(first_point,last_point))
    else:
      rvr_sinuosity_index = np.nan

    point_arr = [(x, y) for x, y in zip(xs,ys)]
    clustP = sc.makeFractionedClusters(point_arr,8)
    sinuosity_index = proc.trajSinIndex(clustP)
    if not all_outputs is None:
      all_outputs = np.vstack((all_outputs,np.array([worm_id,first_frame,last_frame,travel_speed,avg_accel,max_accel,avg_speed,max_speed,total_travel,rvr_sinuosity_index,sinuosity_index])))
    else:
      all_outputs = np.array([worm_id,first_frame,last_frame,travel_speed,avg_accel,max_accel,avg_speed,max_speed,total_travel,rvr_sinuosity_index,sinuosity_index])
  return header, all_outputs

def get_motion_data(folder,out_file):
  vid_header = np.array(["Vid Id","Day"])
  all_data = None
  for file in os.listdir(folder):
    file_name = file.split(".")[0]
    file_path = os.path.join(folder,file)
    header, outputs = calc_motion_data(file_path)
    vid_id = file_name.split("_day")[0]
    day_num = file_name.split("_day")[1]
    vid_info = np.array([[vid_id,day_num] for i in range(outputs.shape[0])])
    cur_data = np.hstack((vid_info,outputs))
    if all_data is None:
      all_data = cur_data
    else:
      all_data = np.vstack((all_data,cur_data))
  header = np.hstack((vid_header,header))
  final_data = np.vstack((header,all_data))
  np.savetxt(out_file,final_data,delimiter=",",fmt='%s')

if __name__ == "__main__":
  file_path = "C:/Users/cdkte/Downloads/sorted-by-id"
  #print(calc_motion_data(file_path))
  get_motion_data(file_path,"C:/Users/cdkte/Downloads/check_motion.csv")
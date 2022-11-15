import motion_from_sorts as mfs
import PySimpleGUI as sg
import numpy as np
import os
import pandas as pd
import numpy as np

def get_motion_data_all_file(folder,output_fold):
  global window
  window["go"].update(disabled = True)
  vid_header = np.array(["Vid Id","Day"])
  all_data = None
  num_files = len(os.listdir(folder))
  vid_id_dict = dict()
  for i,file in enumerate(os.listdir(folder)):
    window['progbar'].update_bar(int(i/num_files*100))
    #window.read()
    window.refresh()
    file_name = file.split(".")[0]
    file_path = os.path.join(folder,file)
    header, outputs = mfs.calc_motion_data(file_path)
    vid_id = file_name.split("_day")[0]
    day_num = file_name.split("_day")[1]
    vid_info = np.array([[vid_id,day_num] for i in range(outputs.shape[0])])
    cur_data = np.hstack((vid_info,outputs))

    if vid_id in vid_id_dict:
      vid_id_dict[vid_id] = np.vstack((vid_id_dict[vid_id],cur_data))
    else:
      header = np.hstack((vid_header,header))
      vid_id_dict[vid_id] = np.vstack((header,cur_data))
  for vid_id in vid_id_dict:
    cur_data = vid_id_dict[vid_id]
    out_file = os.path.join(output_fold,vid_id+".csv")
    np.savetxt(out_file,cur_data,delimiter=",",fmt='%s')
  window["go"].update(disabled = False)

def get_motion_data_one_file(folder,out_file):
  window["go"].update(disabled = True)
  vid_header = np.array(["Vid Id","Day"])
  all_data = None
  num_files = len(os.listdir(folder))
  for i,file in enumerate(os.listdir(folder)):
    window['progbar'].update_bar(int(i/num_files*100))
    #window.read()
    window.refresh()
    file_name = file.split(".")[0]
    file_path = os.path.join(folder,file)
    header, outputs = mfs.calc_motion_data(file_path)
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
  window["go"].update(disabled = False)



layout = [
  [sg.Text("Select the folder to pull data from (sorted-by-id)")],
  [sg.Text("Folder"),sg.FolderBrowse(key="sort_browse")],

  [sg.Text("Where to put the output file?")],
  [sg.Text("Folder"),sg.FolderBrowse(key="output_browse")],

  [sg.Text("What to name the output file? Does nothing if creating separate videoID files")],
  [sg.Input(key="file_name",default_text="output.csv",size=(20,1))],

  [sg.Button("Go", key = "go")],

  [sg.ProgressBar(100,orientation='h', size=(20,20),key="progbar",border_width=4,bar_color=['Red','Green'])
  ],

  [sg.Checkbox("Create Separate Video ID Files",key = "separate_check",tooltip = "Create separate files for each video id")],


]

window = sg.Window("Extract Behavior Manager", layout)

while True:
    event, values = window.read(timeout=100)

    if event == sg.WIN_CLOSED:
      break

    elif event == "go":
      input_fold = values["sort_browse"]
      output_fold = values["output_browse"]
      output_file = values["file_name"]
      do_seperate_files = values["separate_check"]

      if not do_seperate_files:
        get_motion_data_one_file(input_fold,os.path.join(output_fold,output_file))
      else:
        get_motion_data_all_file(input_fold,output_fold)


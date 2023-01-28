import PySimpleGUI as sg
import numpy as np
import os
import pandas as pd
import numpy as np
from annotated_vid import makeVideo

def make_videos(vid_fold, yolo_fold, tod_fold, output_fold):
  global window
  window["go"].update(disabled = True)
  num_files = len(vid_fold)
  for i, file in enumerate(os.listdir(vid_fold)):
    window['progbar'].update_bar(int(i/num_files*100))
    window.refresh()

    file_id = file.split(".avi")[0]
    csv_file = file_id+".csv"
    yolo_file = os.path.join(yolo_fold, csv_file)
    tod_file = os.path.join(tod_fold, csv_file)
    vid_file = os.path.join(vid_fold,file)
    output_file = os.path.join(output_fold,file)


    if (os.path.exists(yolo_file) and os.path.exists(tod_file)):
      makeVideo(vid_file,yolo_file,tod_file,output_file)
  window["go"].update(disabled = False)

layout = [
  [sg.Text("Select the folder to pull videos from(reduced-frame-healthspan)")],
  [sg.Text("Video"),sg.FolderBrowse(key = "vid_browse")],

  [sg.Text("Select the folder to pull data from (yolo)")],
  [sg.Text("Folder"),sg.FolderBrowse(key = "yolo_browse")],

  [sg.Text("Select the folder to pull data from (timelapse-analysis)")],
  [sg.Text("Folder"),sg.FolderBrowse(key="tod_browse")],

  [sg.Text("Where to put the output file?")],
  [sg.Text("Folder"),sg.FolderBrowse(key="output_browse")],


  [sg.Button("Go", key = "go")],

  [sg.ProgressBar(100,orientation='h', size=(20,20),key="progbar",border_width=4,bar_color=['Red','Green'])
  ],
]

window = sg.Window("Extract Behavior Manager", layout)

while True:
    event, values = window.read(timeout=100)

    if event == sg.WIN_CLOSED:
      break

    elif event == "go":
      tod_fold = values["tod_browse"]
      yolo_fold = values["yolo_browse"]
      vid_fold =  values["vid_browse"]
      output_fold = values["output_browse"]

      make_videos(vid_fold,yolo_fold,tod_fold,output_fold)


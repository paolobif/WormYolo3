import numpy as np
import PySimpleGUI as sg
from PIL import Image
import pandas as pd
import cv2
import io
from matplotlib import pyplot as plt
import os
import random as r

default_path1 = "C:/Users/cdkte/Downloads/daily_monitor_all_neural/reduced-frame-healthspan"
default_path2= "C:/Users/cdkte/Downloads/daily_monitor_all_neural/sorted-by-id"

def overlap(box1:tuple, box2:tuple) -> float:
    # box is x1, y1, x2, y2

    left_x = max(box1[0],box2[0])
    right_x = min(box1[2], box2[2])
    bottom_y = max(box1[1], box2[1])
    top_y = min(box1[3],box2[3])
    area = max(0, (top_y-bottom_y))*max(0,(right_x-left_x))
    calc_area = lambda box:(box[2]-box[0])*(box[3]-box[1])
    max_area = min(calc_area(box1),calc_area(box2))
    return area / max_area

def read_data(do_wh, pre_data_path, cur_data_path):

  if not do_wh:
    prev_data = pd.read_csv(pre_data_path,names=["frame", "x1", "y1", "x2", "y2","wormID"])
    cur_data = pd.read_csv(cur_data_path, names=["frame", "x1", "y1", "x2", "y2","wormID"])

  else:
    data1 = pd.read_csv(pre_data_path,names=["frame", "x1", "y1", "w", "h","worm_label"])
    data2 = pd.read_csv(cur_data_path,names=["frame", "x1", "y1", "w", "h","worm_label"])
    data1["x2"] = data1["x1"] + data1["w"]; data1["y2"] = data1["y1"] + data1["h"]
    data2["x2"] = data2["x1"] + data2["w"]; data2["y2"] = data2["y1"] + data2["h"]
    prev_data = data1
    cur_data = data2

  return prev_data, cur_data

def takeFirstFrameBBs(csv_data, frame_offset):
  relevant_data = csv_data.where(csv_data["frame"] < frame_offset).dropna()
  cur_list = []
  past_list = []


  for index, value in relevant_data.iterrows():
    x1 = value["x1"];x2 = value["x2"];y1 = value["y1"];y2 = value["y2"]
    bb = (x1,y1,x2,y2)
    add_to_list = True
    for pre_bb in past_list:
      if overlap(bb,pre_bb) > 0:
        add_to_list = False
    past_list.append(bb)
    if add_to_list:
      cur_list.append(bb)
  print(cur_list)
  return cur_list

def initImages(vid_path, bbs):

  vid = cv2.VideoCapture(vid_path)
  vid.set(cv2.CAP_PROP_POS_FRAMES,3)
  ret, frame = vid.read()

  images = []

  print(frame.shape)
  h,w, colors = frame.shape

  for bb in bbs:
    x1, y1, x2, y2 = bb

    x1-= 10; y1-=10;x2+=10;y2+=10
    if x1 < 0:
      x1 = 0
    if y1 < 0:
      y1 =0
    if x2>w:
      x2=w-1
    if y2 > h:
      y2=h-1
    x1=int(x1);y1=int(y1);x2=int(x2);y2=int(y2)

    images.append(Image.fromarray(frame[y1:y2,x1:x2]).resize((30,30)))

  return images

offset = 20

gen_image_buttons = [sg.Button(str(cur_id),key=str(cur_id)+"_button",visible=False) for cur_id in range(30)]
gen_image_slots = [sg.Image(key=str(cur_id)+"_slots",visible=False) for cur_id in range(30)]
first_button_half = gen_image_buttons[0:len(gen_image_buttons):2]; second_button_half = gen_image_buttons[1:len(gen_image_buttons):2]
first_images_half = gen_image_slots[0:len(gen_image_slots):2]; second_images_half = gen_image_slots[1:len(gen_image_slots):2]

gen_image_rows = [[button,image,button2,image2] for button, image, button2, image2 in zip(first_button_half,first_images_half,second_button_half,second_images_half)]

layout = [
    [
        sg.Text("Pre Vid File path"),
        sg.Input(key="PRE_INPUT_VID",default_text = default_path1+"/3222_day10.avi"),
        sg.FileBrowse(enable_events = True)
    ],
    [
      sg.Text("Pre Csv File Path"),
      sg.Input(key="PRE_INPUT_FILE",default_text = default_path2+"/3222_day10.csv"),
      sg.FileBrowse(enable_events = True)
    ],
    [
      sg.Text("Cur Vid File path"),
      sg.Input(default_text = default_path1+"/3222_day11.avi",key="CUR_INPUT_VID"),
      sg.FileBrowse(enable_events = True)
    ],
    [
      sg.Text("Cur Csv File Path"),
      sg.Input(default_text =  default_path2+"/3222_day10.csv",key="CUR_INPUT_FILE"),
      sg.FileBrowse(enable_events = True)
    ],
    [
      sg.Checkbox("Uses w,h in csv",key = "-WH-",tooltip = ""),

    ],
    [
      sg.Button("Setup Data",key = "SETUP")
    ],
    [
        sg.Text("Output Good File Path"),
        sg.Input(key="OUTPUT_GOOD"),
        sg.FolderBrowse()
    ],
    [
      sg.Text("Output Bad File Path"),
      sg.Input(key="OUTPUT_BAD"),
      sg.FolderBrowse()
    ],
    [
        sg.Button("Next Image", key = "NEXT_IMAGE_BUTTON")
    ],

    [sg.Text("Current Image"),sg.Image(key="CUR_IMAGE")],
    #gen_image_slots,
    #gen_image_buttons,
    [
      sg.Button("No Match", key = "NoMatch")
    ]
]
layout += gen_image_rows



window = sg.Window("Match Trainer",layout,finalize = True,location=(0,0))

for i in range(30):
  gen_image_slots[i].hide_row()

global cur_folder
cur_folder = None

#window.bind("<Key-0>", '0')
#window.bind("<Key-1>", '1')

num_range = [str(num) for num in range(1,10)]


cur_data = None
prev_data = None
prev_bbs = None
cur_bbs = None

prev_imgs = None
cur_imgs = None

cur_index = 0

image_id = 0
video_id = 0


while True:
    event, values = window.read(timeout=100)
    #print(event, values)

    if event == sg.WIN_CLOSED:
        window.close()
        break
    if event == "SETUP":
      prev_data, cur_data = read_data(values["-WH-"],values["PRE_INPUT_FILE"],values["CUR_INPUT_FILE"])
      prev_bbs = takeFirstFrameBBs(prev_data,offset)
      cur_bbs = takeFirstFrameBBs(cur_data,offset)

      prev_imgs = initImages(values["PRE_INPUT_VID"],prev_bbs)
      cur_imgs = initImages(values["CUR_INPUT_VID"],cur_bbs)
      cur_index = 0
      image_id = 0
      video_id = os.path.split(values["PRE_INPUT_VID"])[-1].split(".")[0]

    if event == "NEXT_IMAGE_BUTTON":
      bio = io.BytesIO()
      if cur_index >= len(cur_imgs):
        continue
      cur_imgs[cur_index].save(bio, format="PNG")
      window["CUR_IMAGE"].update(visible=True)
      window["CUR_IMAGE"].update(data=bio.getvalue())
      for i in range(len(prev_imgs)):
        bio = io.BytesIO()
        prev_imgs[i].save(bio, format="PNG")
        window[str(i)+"_slots"].update(data=bio.getvalue())
        window[str(i)+"_slots"].unhide_row()
        window[str(i)+"_slots"].update(visible=True)
        window[str(i)+"_button"].update(visible=True)
    if "_button" in event or event =="NoMatch":
      window["CUR_IMAGE"].update(visible=False)
      for i in range(len(prev_imgs)):
        window[str(i)+"_slots"].hide_row()
        window[str(i)+"_slots"].update(visible=False)
        window[str(i)+"_button"].update(visible=False)

      pre_img_id = -1
      if event != "NoMatch":
        pre_id = event.split("_button")[0]
        pre_img_id = int(pre_id)
        pre_select_img = prev_imgs[pre_img_id]
        cur_select_img = cur_imgs[cur_index]
        good_path = values["OUTPUT_GOOD"]
        a_name = video_id+"-"+str(image_id)+"a.png"
        b_name = video_id+"-"+str(image_id)+"b.png"
        data_name = video_id+"-"+str(image_id)+"_data.txt"
        pre_select_img.save(os.path.join(good_path,a_name))
        cur_select_img.save(os.path.join(good_path,b_name))

        data = prev_bbs[pre_img_id] + cur_bbs[cur_index]
        data_file = open(os.path.join(good_path,data_name),"w+")
        data_file.write(",".join([str(i) for i in data]))
        data_file.close()
        image_id +=1

      # Now bad image
      bad_img_id = r.randint(0,len(prev_imgs)-1)
      while bad_img_id == pre_img_id:
        bad_img_id = r.randint(0,len(prev_imgs)-1)

      pre_select_img = prev_imgs[bad_img_id]
      cur_select_img = cur_imgs[cur_index]
      bad_path = values["OUTPUT_BAD"]
      a_name = video_id+"-"+str(image_id)+"a.png"
      b_name = video_id+"-"+str(image_id)+"b.png"
      data_name = video_id+"-"+str(image_id)+"_data.txt"
      pre_select_img.save(os.path.join(bad_path,a_name))
      cur_select_img.save(os.path.join(bad_path,b_name))

      data = prev_bbs[bad_img_id] + cur_bbs[cur_index]
      data_file = open(os.path.join(bad_path,data_name),"w+")
      data_file.write(",".join([str(i) for i in data]))
      data_file.close()
      image_id +=1


      cur_index += 1






import numpy as np
import cv2
import os
import pandas as pd
import different_worm_discriminator as dwd
from matplotlib import pyplot as plt

FRAME_OFFSET = 30

# TODO: Check loc (only give bbs nearby, or check if matching is too far away)
# TODO: Try and make vid w/ above
# TODO: Try and normalize data (Between 0 and 1)
# Or matching to another image

def overlap(box1:tuple, box2:tuple) -> float:
    """
    Calculates amount of overlap between bounding boxes

    Args:
        box1 (tuple): A bounding box in the form x1,y1,x2,y2
        box2 (tuple): A bounding box in the form x1,y2,x2,y2

    Returns:
        float: _description_
    """
    # box is x1, y1, x2, y2

    left_x = max(box1[0],box2[0])
    right_x = min(box1[2], box2[2])
    bottom_y = max(box1[1], box2[1])
    top_y = min(box1[3],box2[3])
    area = max(0, (top_y-bottom_y))*max(0,(right_x-left_x))
    calc_area = lambda box:(box[2]-box[0])*(box[3]-box[1])
    max_area = min(calc_area(box1),calc_area(box2))
    if max_area == 0:
      return 0
    return area / max_area

def takeFirstFrameBBs(csv_data, frame_offset = FRAME_OFFSET):
  """
  Takes the bounding boxes from the first few frames and clumps together overlapping ones.

  Args:
      csv_data (pandas.dataframe): The pandas dataframe storing detections. Each row must have a frame, x1, x2, y1, and y2
      frame_offset (int, optional): The number of frames to look at from the start for detections. Defaults to 30 (or  value in FRAME_OFFSET)

  Returns:
      list(tuple): A list of bounding boxes in the form (x1,y1,x2,y2)
  """
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
  return cur_list

def createImages(bb_list:list, vid_path:str):
  """
  Creates a list of images that match the bounding boxes

  Args:
      bb_list (list(tuple)): The list of bounding boxes
      vid_path (str): The file path to the video
  """
  cur_vid = cv2.VideoCapture(vid_path)
  cur_vid.set(cv2.CAP_PROP_POS_FRAMES,3)
  ret, frame = cur_vid.read()
  worm_imgs = []

  for cur_worm in bb_list:
    x1, y1, x2, y2 =cur_worm
    x1 = int(x1)-10; x2 = int(x2)+10; y1 = int(y1)-10; y2 = int(y2)+10
    cur_worm_bb = frame[y1:y2,x1:x2]
    if cur_worm_bb.size != 0:
      worm_imgs.append(cur_worm_bb)
    else:
      bb_list.remove(cur_worm)
  return worm_imgs


def matchBBs(vid_data1, vid_data2, uses_wh:bool = False):
  """
  Matches bounding boxes between videos, assumes vid_data_1 is representing the earlier video, and matches backwards.
  Or: Finds all bounding boxes in vid_data2 that match a bounding box in vid_data1

  Args:
      vid_data1 (str, str): The video path and the data path of the first video. In that order!
      vid_data2 (str, str): The video path and the data path of the second video
      uses_wh (bool):
  Returns:
      list(tuple): A list of the matched images in the form of bb1, img1, bb2, img2
        bb(tuple): A bounding box in the form x1,y1,x2,y2
        img(np.array): An image of the worm
  """
  vid1, data1 = vid_data1
  vid2, data2 = vid_data2
  if not uses_wh:
    data1 = pd.read_csv(data1,names=["frame", "x1", "y1", "x2", "y2","wormID"])
    data2 = pd.read_csv(data2,names=["frame", "x1", "y1", "x2", "y2","wormID"])
  else:
    data1 = pd.read_csv(data1,names=["frame", "x1", "y1", "w", "h","worm_label"])
    data2 = pd.read_csv(data2,names=["frame", "x1", "y1", "w", "h","worm_label"])
    data1["x2"] = data1["x1"] + data1["w"]; data1["y2"] = data1["y1"] + data1["h"]
    data2["x2"] = data2["x1"] + data2["w"]; data2["y2"] = data2["y1"] + data2["h"]


  bb_list1 = takeFirstFrameBBs(data1)
  bb_list2 = takeFirstFrameBBs(data2)


  img_list1 = createImages(bb_list1,vid1)
  img_list2 = createImages(bb_list2,vid2)

  matching_images = []
  unmatched_2 = []
  unmatched_1 = []

  matched_set = dict()

  for bb2, img2 in zip(bb_list2, img_list2):
    match_q = []
    # Predict matches
    #fig,ax = plt.subplots(3,len(bb_list1))
    j=0
    for bb1, img1 in zip(bb_list1, img_list1):

      pred_data = bb1 + bb2
      pred_val = dwd.predict_single(img1, img2, pred_data)[0,0]

      """
      ax[0,j].imshow(img2)
      ax[1,j].imshow(img1)
      ax[2,j].imshow(np.full((30,30),pred_val,dtype=np.float32),cmap = "gray", vmin=0, vmax=1)
      ax[0,j].set_xticks([])
      ax[0,j].axes.get_yaxis().set_visible(False)
      ax[1,j].set_xticks([])
      ax[1,j].set_xlabel(j)
      ax[1,j].axes.get_yaxis().set_visible(False)
      ax[2,j].axes.get_xaxis().set_visible(False)
      ax[2,j].axes.get_yaxis().set_visible(False)
      #"""

      j+=1

      match_q.append(pred_val)
    # If there is a matching image, find it
    #plt.show()
    if np.max(match_q) > 0.7:
      index = np.argmax(match_q)
      bb1, img1 = bb_list1[index], img_list1[index]

      cont = True
      print(bb1, bb1 in matched_set)
      if bb1 in matched_set:
        print(bb1,"AH")
        if matched_set[bb1] > np.max(match_q):
          cont = False
        else:
          print("Removing!")
          for item in matching_images:
            match_bb1 = item[0]
            match_img = item[1]
            if match_bb1 == bb1:
              matching_images.remove(item)
              unmatched_1.append((match_bb1,match_img))
              print("Removed")
      if cont:
        matching_images.append((bb1,img1,bb2,img2))
        matched_set[bb1] = np.max(match_q)
      else:
        unmatched_2.append((bb2,img2))
    else:
      unmatched_2.append((bb2,img2))

  for i, img in enumerate(img_list1):

    if not (img.tolist() in [match[1].tolist() for match in matching_images]):
      unmatched_1.append((bb_list1[i],img))
  unmatched = [unmatched_1,unmatched_2]
  return matching_images, unmatched

def matchSeries(video_list, csv_list):
  matches = []
  unmatches = []
  for i in range(len(video_list)-1,0,-1):
    cur_vid = video_list[i]
    prev_vid = video_list[i-1]
    cur_csv = csv_list[i]
    pre_csv = csv_list[i-1]
    pre = (prev_vid, pre_csv)
    cur = (cur_vid, cur_csv)
    match, unmatch = matchBBs(pre, cur)
    matches.append(match)
    unmatches.append(unmatch)
  matches.reverse()
  unmatch.reverse()
  return matches,unmatch

def makeVid(video_list, csv_list, out_vid):
  get_specs = cv2.VideoCapture(video_list[0])
  w = get_specs.get(3); h= get_specs.get(4); shape = (int(w), int(h))

  fourcc = cv2.VideoWriter_fourcc(*"MJPG")
  out_vid = cv2.VideoWriter(out_vid, fourcc, 15, shape)
  print("Beginning Matches")
  matches = matchSeries(video_list,csv_list)
  for vid_index in range(len(video_list)-1):
    read_vid = cv2.VideoCapture(video_list[vid_index])
    ret, frame = read_vid.read()
    for i in range(10):
      ret, frame = read_vid.read()
      for bb1, img1, bb2, img2 in matches[vid_index]:
        """
        fig,ax = plt.subplots(1,2)
        ax[0].imshow(img1)
        ax[1].imshow(img2)
        plt.show()
        """
        x1,y1,x2,y2 = bb2
        p1 = (int(x1),int(y1))
        p2 = (int(x2),int(y2))
        cv2.rectangle(frame, p1, p2, (255,0,0),2)
      out_vid.write(frame)
    read_vid.release()
  out_vid.release()

def makeVidLists(video_folder,csv_folder,worm_id,max_day=30):
  vid_list = []
  csv_list = []
  for i in range(max_day):
    file_name = str(worm_id) +"_day"+ str(i)
    vid_file = os.path.join(video_folder, file_name+".avi")
    csv_file = os.path.join(csv_folder, file_name+".csv")
    if os.path.exists(vid_file) and os.path.exists(csv_file):
      vid_list.append(vid_file)
      csv_list.append(csv_file)
  print("Finish vid list")

  return vid_list, csv_list

def compareTwo(viddata_one, viddata_two):
  match, unmatch = matchBBs(viddata_one, viddata_two)
  pre_vid = cv2.VideoCapture(viddata_one[0])
  pre_vid.set(cv2.CAP_PROP_POS_FRAMES,3)
  ret, pre_frame = pre_vid.read()
  cur_vid = cv2.VideoCapture(viddata_two[0])
  cur_vid.set(cv2.CAP_PROP_POS_FRAMES,3)
  ret, cur_frame = cur_vid.read()

  import random as r

  for bb1, img1, bb2, img2 in match:
    print(bb1,bb2)
    x1,y1,x2,y2 = bb1
    p1 = (int(x1),int(y1))
    p2 = (int(x2),int(y2))

    rgb = (r.randint(1,255),r.randint(1,255),r.randint(1,255))
    cv2.rectangle(pre_frame, p1, p2, rgb,2)

    x1,y1,x2,y2 = bb2
    p1 = (int(x1),int(y1))
    p2 = (int(x2),int(y2))
    cv2.rectangle(cur_frame, p1, p2, rgb,2)

  unmatch1, unmatch2 = unmatch

  for bb1, img1 in unmatch1:
    x1,y1,x2,y2 = bb1
    p1 = (int(x1),int(y1))
    p2 = (int(x2),int(y2))

    rgb = (255,0,0)
    cv2.rectangle(pre_frame, p1, p2, rgb,2)

  for bb2, img2 in unmatch2:
    x1,y1,x2,y2 = bb2
    p1 = (int(x1),int(y1))
    p2 = (int(x2),int(y2))

    rgb = (255,0,0)
    cv2.rectangle(cur_frame, p1, p2, rgb,2)


  fig, ax = plt.subplots(2,1)
  ax[0].imshow(pre_frame)
  ax[1].imshow(cur_frame)
  plt.show()

def plotDetectionsInFirstFrames(viddata):
  video, data = viddata
  if not True:
    data = pd.read_csv(data,names=["frame", "x1", "y1", "x2", "y2","wormID"])
  else:
    data = pd.read_csv(data,names=["frame", "x1", "y1", "w", "h","worm_label"])
    data["x2"] = data["x1"] + data["w"]; data["y2"] = data["y1"] + data["h"]

  bbs = takeFirstFrameBBs(data)

  video = cv2.VideoCapture(video)

  fig, ax = plt.subplots(5,1)
  div = (FRAME_OFFSET+1)//5
  for i in range(1,FRAME_OFFSET+1,div):
    video.set(cv2.CAP_PROP_POS_FRAMES,i)
    ret, frame = video.read()
    for bb in bbs:
      x1,y1,x2,y2 = bb
      p1 = (int(x1),int(y1))
      p2 = (int(x2),int(y2))
      cv2.rectangle(frame,p1,p2,(255,0,0),2)
    ax[i//div].imshow(frame)
    print(i//div)
  plt.show()

def matchForward(vid_path, csv_path):
  vid_list, csv_list = makeVidLists(vid_path,csv_path)


if __name__ == "__main__":
  """
  vid_path = "C:/Users/cdkte/Downloads/daily_monitor_new_test/vids"
  csv_path = "C:/Users/cdkte/Downloads/daily_monitor_new_test/csvs"
  print("Starting Vid Lists")
  vid_list, csv_list = makeVidLists(vid_path,csv_path,78)
  video_path = "C:/Users/cdkte/Downloads/daily_monitor_new_test/test.avi"
  makeVid(vid_list,csv_list,video_path)
  """
  vid_path = "C:/Users/cdkte/Downloads/daily_monitor_new_test/vids"
  csv_path = "C:/Users/cdkte/Downloads/daily_monitor_new_test/csvs"

  vid_path = "C:/Users/cdkte/Downloads/daily_monitor_all_neural/reduced-frame-healthspan"
  csv_path = "C:/Users/cdkte/Downloads/daily_monitor_all_neural/sorted-by-id"

  pre_vid_path = os.path.join(vid_path,"3270_day15.avi")
  cur_vid_path = os.path.join(vid_path,"3270_day16.avi")

  pre_data_path = os.path.join(csv_path,"3270_day15.csv")
  cur_data_path = os.path.join(csv_path, "3270_day16.csv")

  compareTwo((pre_vid_path,pre_data_path),(cur_vid_path,cur_data_path))
  #plotDetectionsInFirstFrames((cur_vid_path,cur_data_path))
  """

  vid_path = "C:/Users/cdkte/Downloads/daily_monitor_all_neural/reduced-frame-healthspan"
  csv_path = "C:/Users/cdkte/Downloads/daily_monitor_all_neural/sorted-by-id"


  pre_vid_path = os.path.join(vid_path,"3222_day14.avi")
  cur_vid_path = os.path.join(vid_path,"3222_day15.avi")

  pre_data_path = os.path.join(csv_path,"3222_day14.csv")
  cur_data_path = os.path.join(csv_path, "3222_day15.csv")

  pre = (pre_vid_path, pre_data_path)
  cur = (cur_vid_path, cur_data_path)

  matching = matchBBs(pre, cur)

  fig, ax = plt.subplots(2,len(matching))

  for i in range(len(matching)):
    bb1, img1, bb2, img2 = matching[i]
    ax[0,i].imshow(img1)
    ax[1,i].imshow(img2)
  plt.show()
  """
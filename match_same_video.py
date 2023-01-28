import matching_utils as mu
import pandas as pd
import cv2
import os
from matplotlib import pyplot as plt
import random as r

MAX_DISTANCE = 40

# TODO: Similar to sort, create index of matches over time & filter for consistent matches
# TODO: Normalize xs and ys
# NMS stuff

# TODO: Try power save mode and check charger
# All power saving stuff was for while asleep

# TODO: Setup batch process to folder w/ label of method, hyperparameters and same name videos
#   Just timelapse videos in the dropbox, not daily monitor

# TODO: Try to make feature detection for networks seperately i.e. resnet

class tracked_dead_worm:
  def __init__(self,bb, img, frame):
    self.bbs = [bb]
    self.imgs = [img]
    self.time_count = 0
    self.first_frame = frame
    self.rgb = (r.randint(0,255),r.randint(0,255),r.randint(0,255))
  def get_last(self):
    return self.bbs[-1],self.imgs[-1]
  def get_length(self):
    return len(self.bbs)
  def tick_up(self):
    self.time_count += 1
  def check_time(self):
    return self.time_count
  def add_detection(self,bb,img):
    self.bbs.append(bb)
    self.imgs.append(img)
    self.time_count = 0

def progressive_matching(vid_path,csv_path,uses_wh = True,num_divs = 10,time_limit = 2, length_limit = 3, end_allowance = 0):
  """
  Matches worm detections between frames keeping track of matched worms moving forward

  Args:
      vid_path (_type_): Path to pull video from
      csv_path (_type_): Path to pull data from
      uses_wh (bool, optional): Determines how to read the csv data. Defaults to True.
      num_divs (int, optional): Number of frames to look at from the video. Defaults to 10.
      time_limit (int, optional): How long to wait for another detection before removing the worm. Defaults to 2.
      length_limit (int, optional): How many times a worm has to show up before it can't be removed. Defaults to 3.
      end_allowance (int, optional): The amount of forgiveness for worms that show up at the end of the video. Defaults to 0.
  Returns:
      _type_: _description_
  """
  if not uses_wh:
    data = pd.read_csv(csv_path,names=["frame", "x1", "y1", "x2", "y2","wormID"])
  else:
    data = pd.read_csv(csv_path,names=["frame", "x1", "y1", "w", "h","worm_label"])
    data["x2"] = data["x1"] + data["w"]; data["y2"] = data["y1"] + data["h"]

  cur_vid = cv2.VideoCapture(vid_path)
  vid_size = cur_vid.get(cv2.CAP_PROP_FRAME_COUNT)
  division_size = vid_size // num_divs

  division_starts = [int(division_size*i) for i in range(num_divs)]
  div_width = 2

  bbs_each_division = [mu.takeFirstFrameBBs(data,start_frame = cur_div,frame_offset = div_width) for cur_div in division_starts]

  imgs_each_division = [mu.createImages(bbs_this_div,vid_path,starting_frame) for bbs_this_div,starting_frame in zip(bbs_each_division,division_starts)]

  first_bbs = bbs_each_division[0]
  first_imgs = imgs_each_division[0]

  tracked_bbs = [tracked_dead_worm(bb,img,division_starts[0]) for bb, img in zip(first_bbs,first_imgs)]
  div_i = 1

  while div_i < num_divs:
    print(div_i)
    cur_bbs = bbs_each_division[div_i]
    cur_imgs = imgs_each_division[div_i]
    tracked_worms:list[tracked_dead_worm] = [worm.get_last() for worm in tracked_bbs]
    pre_tracked_bbs = [worm[0] for worm in tracked_worms]
    pre_tracked_imgs = [worm[1] for worm in tracked_worms]

    # Match all images
    matching_imgs, unmatched = mu.matchWithDiscriminator(pre_tracked_bbs,pre_tracked_imgs,cur_bbs,cur_imgs,max_distance = MAX_DISTANCE)
    for worm in matching_imgs:
      bb1, img1, bb2, img2 = worm
      # Find matching previous worm
      for i in range(len(tracked_bbs)):
        pre_worm = tracked_bbs[i]
        # Add new detection if matching
        if bb1 == pre_worm.get_last()[0]:
          pre_worm.add_detection(bb2,img2)

    # remove old worms
    for pre_worm in tracked_bbs:
      if pre_worm.check_time() > time_limit and pre_worm.get_length() < length_limit:
        tracked_bbs.remove(pre_worm)

    for unmatched_worm in unmatched[1]:
      tracked_bbs.append(tracked_dead_worm(unmatched_worm[0],unmatched_worm[1],division_starts[div_i]))

    # Uptick time
    for worm in tracked_bbs:
      worm.tick_up()
    div_i += 1

  # Drop worms on the end
  for worm in tracked_bbs:
    if worm.get_length() < length_limit - end_allowance:
      tracked_bbs.remove(worm)
  return tracked_bbs

def plotProgressive(vid_path, tracked_worms:list[tracked_dead_worm],out_path = None):

  vid = cv2.VideoCapture(vid_path)
  if not out_path is None:
    frame_width = vid.get(3)
    frame_height = vid.get(4)
    size = (int(frame_width),int(frame_height))
    fps = vid.get(cv2.CAP_PROP_FPS)
    #fps = 2

    write_vid = cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
  ret, frame = vid.read()
  while ret:
    cur_frame = vid.get(cv2.CAP_PROP_POS_FRAMES)
    for worm in tracked_worms:
      if cur_frame > worm.first_frame:
        pt1, pt2 = mu.getCoords(worm.get_last()[0])
        cv2.rectangle(frame,pt1,pt2,worm.rgb,thickness = 2)
    if not out_path is None:
      write_vid.write(frame)


    ret, frame = vid.read()

  vid.release()
  if not out_path is None:
    write_vid.release()

def load_dataset(vid_path,csv_path,uses_wh = True, num_divs = 10, div_width=2):
  if not uses_wh:
    data = pd.read_csv(csv_path,names=["frame", "x1", "y1", "x2", "y2","wormID"])
  else:
    data = pd.read_csv(csv_path,names=["frame", "x1", "y1", "w", "h","worm_label"])
    data["x2"] = data["x1"] + data["w"]; data["y2"] = data["y1"] + data["h"]

  cur_vid = cv2.VideoCapture(vid_path)
  vid_size = cur_vid.get(cv2.CAP_PROP_FRAME_COUNT)
  division_size = vid_size // num_divs

  division_starts = [int(division_size*i) for i in range(num_divs)]

  bbs_each_division = [mu.takeFirstFrameBBs(data,start_frame = cur_div,frame_offset = div_width) for cur_div in division_starts]

  imgs_each_division = [mu.createImages(bbs_this_div,vid_path,starting_frame) for bbs_this_div,starting_frame in zip(bbs_each_division,division_starts)]

  divisions_match = []
  for i in range(num_divs-1):
    pre_imgs = imgs_each_division[i]
    cur_imgs = imgs_each_division[i+1]
    pre_bbs = bbs_each_division[i]
    cur_bbs = bbs_each_division[i+1]
    print(len(pre_imgs),len(pre_bbs),len(cur_imgs),len(cur_bbs))
    matched, unmatched = mu.matchWithDiscriminator(pre_bbs,pre_imgs,cur_bbs,cur_imgs,max_distance=MAX_DISTANCE)
    for bb1, img1, bb2, img2 in matched:
      if mu.distance(bb1,bb2) > MAX_DISTANCE:
        matched.remove((bb1,img1,bb2,img2))
        unmatched[0].append((bb1,img1))
        unmatched[1].append((bb2,img2))

    divisions_match.append(matched)

  return division_starts, divisions_match

def plotImages(video_path,division_starts,division_matches, out_path = None):

  vid = cv2.VideoCapture(video_path)
  if out_path:
    frame_width = vid.get(3)
    frame_height = vid.get(4)
    size = (int(frame_width),int(frame_height))
    #fps = vid.get(cv2.CAP_PROP_FPS)
    fps = 2

    write_vid = cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

  for i in range(len(division_starts)-1):
    pre_frame = division_starts[i]
    vid.set(cv2.CAP_PROP_POS_FRAMES,pre_frame)
    ret, pre_frame = vid.read()

    cur_frame = division_starts[i+1]
    vid.set(cv2.CAP_PROP_POS_FRAMES,cur_frame)
    ret, cur_frame = vid.read()

    for bb1, img1, bb2, img2 in division_matches[i]:
      if out_path:
        rgb = (255,0,0)
      else:
        rgb = (r.randint(1,255),r.randint(1,255),r.randint(1,255))
      rgb = (r.randint(1,255),r.randint(1,255),r.randint(1,255))
      pt1, pt2 = mu.getCoords(bb1)
      cv2.rectangle(pre_frame,pt1,pt2,rgb)
      pt1, pt2 = mu.getCoords(bb2)
      cv2.rectangle(cur_frame,pt1,pt2,rgb)

    if out_path:
      write_vid.write(pre_frame)
      write_vid.write(cur_frame)
    else:
      fig, ax = plt.subplots(2)
      ax[0].imshow(pre_frame)
      ax[1].imshow(cur_frame)
      plt.show()
  vid.release()
  write_vid.release()

if __name__ == "__main__":

  time_limit = 3
  num_divs = 20
  length_limit = 8
  end = 0
  #"""
  vid_path = "C:/Users/cdkte/Downloads/N2/yolo"
  csv_path = "C:/Users/cdkte/Downloads/N2/csv"
  out_path = "C:/Users/cdkte/Downloads/N2/320_match_prog_divs"+str(num_divs)+"_timelimit"+str(time_limit)+"_lengthlimit"+str(length_limit)+"_end"+str(end)+"_alt2.avi"
  vid_name = "320.avi"
  csv_name = "320.csv"
  csv_path = os.path.join(csv_path,csv_name)
  vid_path = os.path.join(vid_path,vid_name)
  """
  vid_path = "C:/Users/cdkte/Downloads/3924.avi"
  csv_path = "C:/Users/cdkte/Downloads/3924.csv"
  out_path = "C:/Users/cdkte/Downloads/3924_match_div20_time3_length8_end0.avi"
  #"""
  #division_starts, divisions_match = load_dataset(vid_path,csv_path,num_divs = 30)
  #plotImages(vid_path,division_starts,divisions_match,out_path)
  dead_worms = progressive_matching(vid_path,csv_path,num_divs = num_divs,time_limit=time_limit,length_limit=length_limit,end_allowance=end)
  plotProgressive(vid_path,dead_worms,out_path=out_path)
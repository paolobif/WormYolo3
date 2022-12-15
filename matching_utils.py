import numpy as np
import different_worm_discriminator as dwd
import cv2
from matplotlib import pyplot as plt

FRAME_OFFSET = 30

def getCoords(bb):
  return (int(bb[0]),int(bb[1])),(int(bb[2]),int(bb[3]))

def distance(bb1, bb2):
  coord1 = np.array(((bb1[0]+bb1[2])/2,(bb1[1]+bb1[3])/2))
  coord2=np.array(((bb2[0]+bb2[2])/2,(bb2[1]+bb2[3])/2))
  return np.linalg.norm(coord1-coord2)

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

def takeFirstFrameBBs(csv_data, frame_offset = FRAME_OFFSET, start_frame = 0,):
  """
  Takes the bounding boxes from the first few frames and clumps together overlapping ones.

  Args:
      csv_data (pandas.dataframe): The pandas dataframe storing detections. Each row must have a frame, x1, x2, y1, and y2
      frame_offset (int, optional): The number of frames to look at from the start for detections. Defaults to 30 (or  value in FRAME_OFFSET)

  Returns:
      list(tuple): A list of bounding boxes in the form (x1,y1,x2,y2)
  """
  relevant_data = csv_data.where(csv_data["frame"] < start_frame+frame_offset).where(csv_data["frame"] >= start_frame).dropna()
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

def createImages(bb_list:list, vid_path:str,starting_frame = 3):
  """
  Creates a list of images that match the bounding boxes

  Args:
      bb_list (list(tuple)): The list of bounding boxes
      vid_path (str): The file path to the video
  """
  cur_vid = cv2.VideoCapture(vid_path)
  cur_vid.set(cv2.CAP_PROP_POS_FRAMES,starting_frame)
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

DO_PLT = False

def matchWithDiscriminator(bb_list1,img_list1,bb_list2,img_list2,max_distance:float = None):
  matching_images = []
  unmatched_2 = []
  unmatched_1 = []
  if max_distance is None:
    max_distance = np.inf

  matched_set = dict()
  for bb2, img2 in zip(bb_list2, img_list2):
    if DO_PLT:
      fig,ax = plt.subplots(3,len(bb_list1))
    match_q = []
    # Predict matches
    j=0
    for bb1, img1 in zip(bb_list1, img_list1):
      pred_data = bb1 + bb2
      if distance(bb1,bb2) < max_distance:
        pred_val = dwd.predict_single(img1, img2, pred_data)[0,0]
      else:
        pred_val = 0
      if DO_PLT:
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
      j+=1
      match_q.append(pred_val)

    if DO_PLT:
      plt.show()
    if np.max(match_q) > 0.7:
      index = np.argmax(match_q)
      bb1, img1 = bb_list1[index], img_list1[index]

      cont = True
      if bb1 in matched_set:
        if matched_set[bb1] > np.max(match_q):
          cont = False
        else:
          for item in matching_images:
            match_bb1 = item[0]
            match_img = item[1]
            if match_bb1 == bb1:
              matching_images.remove(item)
              unmatched_1.append((match_bb1,match_img))
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


import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from colour import Color
from scipy import stats
from collections import Counter
import tensorflow_sdf as tsdf
import blot_circle_crop as bcc


DEFAULT_SORT = lambda coord: np.sqrt(coord[0]**2+coord[1]**2)
DEFAULT_CLOSE = lambda coord1, coord2: np.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)
DEFAULT_ACCEPT = 10


# How close we should look when taking the mode
RANGE_SIZE_ONE = 3
RANGE_SIZE_TWO = 5


def simplifyFrameLists(list_of_frames,sort_func = DEFAULT_SORT,close_func = DEFAULT_CLOSE,accept_distance = DEFAULT_ACCEPT,direction = 1):
    out_list = []
    out_index = 0
    if direction == 1:
        list_direction = range(len(list_of_frames))
    else:
        list_direction = range(len(list_of_frames)-1,-1,-1)
    for cur_list_index in list_direction:
        cur_list = list_of_frames[cur_list_index]
        cur_index = 0

        while cur_index < len(cur_list):
            cur_coord = cur_list[cur_index]

            if out_index == len(out_list):
                out_list.append(cur_coord)
                out_index += 1
                cur_index += 1


            else:
                sort_val = sort_func(cur_coord) - sort_func(out_list[out_index])
                close_val = close_func(cur_coord, out_list[out_index])

                if abs(close_val) < accept_distance:
                    out_list[out_index] = cur_coord
                    cur_index += 1
                    out_index += 1

                elif sort_val > 0:
                    out_index += 1

                elif sort_val < 0:
                    out_list.insert(out_index,cur_coord)

                    out_index += 1
                    cur_index += 1

        out_index = 0
    return out_list


def getBbOffset(arr1:list, arr2:list):
    x1, y1 = getOffsetModes(arr1, arr2, RANGE_SIZE_ONE)
    x2, y2 = getOffsetModes(arr1, arr2, RANGE_SIZE_TWO)

    list1= getFirstFive(x1)
    list2 = getFirstFive(x2)
    x_val = None
    for coord in list1:
        if x_val is None:
            x_val = coord
        for alt_coord in list2:
            if abs(coord-alt_coord) < 5:
                if abs(coord) < abs(x_val):
                    x_val = coord

    list1= getFirstFive(y1)
    list2 = getFirstFive(y2)
    y_val = None
    for coord in list1:
        if y_val is None:
            y_val = coord
        for alt_coord in list2:
            if abs(coord-alt_coord) < 5:
                if abs(coord) < abs(y_val):
                    y_val = coord



    #diff = (-int(stats.mode(x_arr)[0]),int(-stats.mode(y_arr)[0]))
    return (-int(x_val),-int(y_val))

def getFirstFive(dictionary):
    return sorted(dictionary, key=dictionary.get, reverse = True)[0:5]

def getOffsetModes(arr1, arr2, range_size):
    # Get (x, y) difference between all points in each frame
    p_arr = np.array([[p1-p2 for p2 in np.array(arr2)] for p1 in np.array(arr1)])

    # Buffer x and y so the mode doesn't need to be exact
    x_arr = p_arr[:,:,0].flatten()
    copy_arr = x_arr.copy()
    for i in range(-range_size,range_size+1):
        copy_arr = np.concatenate((copy_arr,x_arr+i))
    x_arr = copy_arr

    y_arr = p_arr[:,:,1].flatten()
    copy_arr = y_arr.copy()
    for i in range(-range_size,range_size+1):
        copy_arr = np.concatenate((copy_arr,y_arr+i))
    y_arr = copy_arr

    return Counter(x_arr), Counter(y_arr)

    """
    # Get mode of differences
    print(Counter(y_arr))
    #print(x_arr,y_arr)


    return diff
    """



def getMatchingWorms(prev_day_vid, cur_day_vid, prev_day_data, cur_day_data):
    # Load Data
    prev_vid = cv2.VideoCapture(prev_day_vid)
    cur_vid = cv2.VideoCapture(cur_day_vid)

    prev_length = prev_vid.get(cv2.CAP_PROP_FRAME_COUNT)
    cur_length = cur_vid.get(cv2.CAP_PROP_FRAME_COUNT)

    prev_data = pd.read_csv(prev_day_data,names=["frame", "x1", "y1", "x2", "y2","wormID"])
    cur_data = pd.read_csv(cur_day_data, names=["frame", "x1", "y1", "x2", "y2","wormID"])

    # How many frames to look at in each
    frame_offset = 30

    prev_end_points = prev_data.where(prev_data["frame"] > prev_length-frame_offset).dropna()
    red = Color("red")
    colors_prev = list(red.range_to(Color("green"),frame_offset))

    cur_start_points = cur_data.where(cur_data["frame"] < frame_offset).dropna()
    red = Color("yellow")
    colors_cur = list(red.range_to(Color("blue"),frame_offset))

    # Starts at last frame of video
    prev_end_frame_points = [[] for i in range(frame_offset)]

    prev_point_dict = dict()

    # Get every bb for each frame
    for index, row in prev_end_points.iterrows():
        frame_off = prev_length - row["frame"]
        color = colors_prev[int(frame_off)]
        x_values = [row["x1"],row["x2"],row["x2"],row["x1"],row["x1"]]
        y_values = [row["y1"],row["y1"],row["y2"],row["y2"],row["y1"]]

        x_avg = int((row["x1"]+row["x2"])/2)
        y_avg = int((row["y1"]+row["y2"])/2)

        out_coord = (x_avg, y_avg)

        prev_point_dict[out_coord] = (x_values, y_values)

        prev_end_frame_points[int(frame_off)].append((x_avg,y_avg))


    [frame_list.sort(key=DEFAULT_SORT) for frame_list in prev_end_frame_points]

    prev_point_list = simplifyFrameLists(prev_end_frame_points)

    # Starts at first frame of video
    cur_start_frame_points = [[] for i in range(frame_offset)]

    cur_point_dict = dict()


    # Get bb for first _offset_ frame
    for index, row in cur_start_points.iterrows():
        frame_off = row["frame"]
        x_values = [row["x1"],row["x2"],row["x2"],row["x1"],row["x1"]]
        y_values = [row["y1"],row["y1"],row["y2"],row["y2"],row["y1"]]

        x_avg = int((row["x1"]+row["x2"])/2)
        y_avg = int((row["y1"]+row["y2"])/2)

        cur_point_dict[(x_avg,y_avg)] = (x_values, y_values)

        cur_start_frame_points[int(frame_off)].append((x_avg,y_avg))

    [frame_list.sort(key=DEFAULT_SORT) for frame_list in cur_start_frame_points]

    cur_point_list = simplifyFrameLists(cur_start_frame_points)

    offset = getBbOffset(prev_point_list, cur_point_list)

    """
    for point in prev_point_list:
        plt.plot(point[0],point[1],'bo')
        xes, yes = prev_point_dict[point]
        plt.plot(xes, yes, 'b-')
    for point in cur_point_list:
        plt.plot(point[0],point[1],'ro')
        xes, yes = cur_point_dict[point]
        plt.plot(xes, yes, 'r-')
    for point in prev_point_list:
        plt.plot(point[0]+offset[0],point[1]+offset[1],'go')
        xes, yes = prev_point_dict[point]
        xes = np.array(xes) + offset[0]
        yes = np.array(yes) + offset[1]
        plt.plot(xes, yes, 'g-')
    plt.show()
    """

    matching_pairs = []
    for prev_coord in prev_point_list:
        matching_coord= None
        match_dist = DEFAULT_ACCEPT*3
        for cur_coord in cur_point_list:
            temp_coord = (prev_coord[0]+offset[0], prev_coord[1]+offset[1])
            #print(DEFAULT_CLOSE(temp_coord, cur_coord))
            if DEFAULT_CLOSE(temp_coord, cur_coord) < match_dist:
                matching_coord = cur_coord
                #print(prev_coord, cur_coord)
                match_dist = DEFAULT_CLOSE(temp_coord, cur_coord)
        if not matching_coord is None:
            matching_pairs.append((prev_point_dict[prev_coord],cur_point_dict[matching_coord]))
    frame_list = []
    for prev_coord, cur_coord in matching_pairs:
        prev_pos = None
        cur_pos = None

        cur_test = cur_data.where(cur_data["x1"]==cur_coord[0][0]).where(cur_data["y1"]==cur_coord[1][0]).dropna()
        if len(cur_test) > 0:
            cur_pos = cur_test.iloc[0]["frame"]

        prev_test = prev_data.where(prev_data["x1"]==prev_coord[0][0]).where(prev_data["y1"]==prev_coord[1][0]).dropna()
        if len(prev_test) > 0:
            prev_pos = prev_test.iloc[-1]["frame"]


        frame_list.append((prev_pos, cur_pos))

    matching_images = []
    img_offset = 10
    for coord_pair, frame_pair in zip(matching_pairs, frame_list):
        pre_coord, cur_coord = coord_pair
        pre_frame, cur_frame = frame_pair

        prev_vid.set(cv2.CAP_PROP_POS_FRAMES,pre_frame-1)
        ret, frame = prev_vid.read()
        x1 = int(pre_coord[0][0])-img_offset; x2 = int(pre_coord[0][2])+img_offset
        y1 = int(pre_coord[1][0])-img_offset; y2 = int(pre_coord[1][2])+img_offset
        pre_worm_bb = frame[y1:y2,x1:x2]

        cur_vid.set(cv2.CAP_PROP_POS_FRAMES,cur_frame-1)
        ret, frame = cur_vid.read()
        x1 = int(cur_coord[0][0])-img_offset; x2 = int(cur_coord[0][2])+img_offset
        y1 = int(cur_coord[1][0])-img_offset; y2 = int(cur_coord[1][2])+img_offset
        cur_worm_bb = frame[y1:y2,x1:x2]


        pre_anno = tsdf.generate_single_sdf(pre_worm_bb)
        pre_anno = bcc.dropSmall(pre_anno)
        cur_anno = tsdf.generate_single_sdf(cur_worm_bb)
        cur_anno = bcc.dropSmall(cur_anno)

        # Get the amount of overlap

        pre_anno_resize = cv2.resize(pre_anno,(cur_anno.shape[1],cur_anno.shape[0]))

        # Try and get max_overlap
        buffer_size = 10
        shape = cur_anno.shape
        new_shape = (shape[0]+2*buffer_size, shape[1]+2*buffer_size)
        largest_anno = max(np.sum(pre_anno),np.sum(cur_anno))

        pre_buff = np.zeros(new_shape)
        print(pre_buff.shape,shape[0])
        pre_buff[buffer_size:buffer_size+shape[0], buffer_size:buffer_size+shape[1]] += pre_anno_resize
        max_overlap = 0

        for x_buf in range(0, buffer_size*2):
            for y_buf in range(0, buffer_size*2):
                cur_buff = np.zeros(new_shape)

                cur_buff[x_buf:x_buf+shape[0], y_buf:y_buf+shape[1]] += cur_anno
                cur_overlap =  np.sum(np.multiply(pre_buff,cur_buff))
                if cur_overlap > max_overlap:
                    max_overlap = cur_overlap
        # Easy overlap
        anno_overlap = np.sum(np.multiply(pre_anno_resize,cur_anno))



        print(anno_overlap/largest_anno,max_overlap/largest_anno)
        # < 0.7 seems to be the bounds

        """
        figs, axs = plt.subplots(2,2)
        axs[0,0].imshow(pre_anno)
        axs[1,0].imshow(cur_anno)
        axs[0,1].imshow(pre_anno_resize)
        plt.show()
        #"""


    return matching_pairs, frame_list

if __name__ == "__main__":

    vid_path = "C:/Users/cdkte/Downloads/mot_backtrack"
    csv_path = "C:/Users/cdkte/Downloads/mot_backtrack"

    prev_day_vid = "693_day10_simple.avi"
    cur_day_vid = "693_day12_simple.avi"

    prev_day_vid = os.path.join(vid_path,prev_day_vid)
    cur_day_vid = os.path.join(vid_path,cur_day_vid)

    prev_day_data = "693_day10_simple.csv"
    cur_day_data = "693_day12_simple.csv"


    prev_day_data = os.path.join(csv_path,prev_day_data)
    cur_day_data = os.path.join(csv_path,cur_day_data)

    results, frame = getMatchingWorms(prev_day_vid, cur_day_vid, prev_day_data, cur_day_data)
    for (box1, box2) in results:
        #print("Plotted")
        xes, yes = box1
        plt.plot(xes, yes, 'r-')
        xes, yes = box2
        plt.plot(xes, yes, 'g-')
    plt.show()
    print()

    # TODO: Plot matched bbs for each day, over time / total worms
    #   Should look like a survival curse

    # TODO: Downsample to 1 fps - already part of GUI

    # New datasets should go to completion, get those done

    # Structure code to download everything
    #   Name structure 643_day10 to seperate folder

    # Stuff at edges shift differently
    # (Try to mimic telecentric lens in code? Probably hardware. Don't spend too much time on this)

    # What aspects can be done w/ just BBs?
    # TODO: Combine w/ box overlap tracker
    # TODO: Remember to do the overlap tacker once the data finishes!!!!

    # TODO: Likelihood that a worm is unmoving for 1 video but visibly moves between that day and the next.
    # Ideally should get a lifespan curve

    # All the data is taken from the daily monitor rather than the timelapse
    # Increase throughput 4-5x, so we want to use these rather than timelapse

    # TODO: Get some slides (1 or 2) for Friday so we can present


    # TODO: Plot day on X, frac of bbs that moved on Y
    # Track through entire video (try to) using same tracking
    # Threshold for "same bb"
    # Possible Lifespan curve
    # Try using area overlap
    # Sum overlap for one worm over frames, get total overlap for stationary worm
    # Compare that w/ margin to other worms to see if they moved




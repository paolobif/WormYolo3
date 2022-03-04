import os 
import numpy as np 
import matplotlib as plt 
import cv2
import fnmatch

if __name__ == "__main__": 
    #set paths
    DIR = "/Users/shreyaramakrishnan/Documents/kaeberlein lab/"
    MOV_DIR = os.path.join(DIR, "Shreya_Test/")
    VID_DIR = os.path.join(DIR, "generator testing/")
    

    #change this to .avi 
    vid_list = fnmatch.filter(os.listdir(MOV_DIR),"*.mp4")

    # grab the video
    for vid in vid_list: 
        directory = VID_DIR 
        vid_num = vid.split(".")
        vid_num = vid_num[0]
        vid_num = vid_num + "/"
        vid_path = MOV_DIR+vid
        vid = cv2.VideoCapture(vid_path)

        img_path = os.path.join(VID_DIR, vid_num)
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
            # initialize a counter to track total frames 
            total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
            frame_count = 0 # current frame 
            packet_frames = 0 # counter to check how many frames are in the packet 
            packet_num = 0; 
            # make packets while there are frames left 
            while (frame_count < total_frames): 
                print(frame_count)
                ret, frame = vid.read()
                frame_count += 1 
                if (packet_frames % 10 == 0): 
                    packet_frames = 1 
                    packet_num += 1
                    directory = "packet " + str(packet_num) 
                    print(directory)
                    print(VID_DIR)
                    curr_path = os.path.join(VID_DIR, vid_num, directory)
                    os.mkdir(curr_path)
                    print(curr_path)  
                im_path = curr_path + "/img" + str(packet_frames) + ".png"
                cv2.imwrite(im_path, frame)
                packet_frames += 1
        else: 
            print("exists")


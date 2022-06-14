import cv2
from matplotlib import image
import tqdm
import PIL
import numpy as np
import skimage
'''
1. take in a file path of videos 
2. go through each video in the file path and downsample them
3. write it out to a new video path
'''

def downsample(vid_path: str, save_path: str):

    # break the video down into frames 
    vid = cv2.VideoCapture(vid_path)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = vid.read()
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(save_path, fourcc, 10, (width, height), True)

    # go through each frame and downsample it 
    for _ in tqdm(range(total_frames)):
        ret, frame = vid.read()
        frame_count = vid.get(cv2.CAP_PROP_POS_FRAMES)
        if not ret:
            continue

        # convert the image into an array
        image_sequence = frame.getdata()
        image_array = np.array(image_sequence)

        # manipulate the image array to downsample 

        downsample = 20
        # first, change to 0-1
        ds_array = image_array/255
        r = skimage.measure.block_reduce(ds_array[:, :, 0],
                                        (downsample, downsample),
                                        np.mean)
        g = skimage.measure.block_reduce(ds_array[:, :, 1],
                                        (downsample, downsample),
                                        np.mean)
        b = skimage.measure.block_reduce(ds_array[:, :, 2],
                                        (downsample, downsample),
                                        np.mean)
        ds_array = np.stack((r, g, b), axis=-1)

        frame = cv2.imwrite(ds_array)
        writer.write(frame)
    pass 

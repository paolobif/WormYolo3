import os
import sys
import cv2
import csv
import tqdm
import pandas as pd
import random
import numpy as np
import time
from skimage.morphology import skeletonize, medial_axis
from matplotlib import pyplot as plt


def draw_on_im(frame, outputs, out_path, writer, text=None):
        """Takes img, then coordinates for bounding box, and optional text as arg"""
        img = frame
        for output in outputs:
            #output = [int(n) for n in output]
            frameN, x, y, text, *_ = output
            
            # Draw rectangles
            if text is not None: 
                text = str(text)
                cv2.putText(img, text, (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,102,102), 2)
        ## write image to path
        #cv2.imwrite(out_path, img)
        writer.write(img)


def skeleton(df,frame,writer):        
    for output in df:
        output = [int(n) for n in output]
        frameN, x1, y1, x2, y2, label, *_ = output
        if int(label) == 4:
            croppedLarge = frame[(y1-30):(y2+30),(x1-30):(x2+30)]
            thresh = grabcut2(croppedLarge, x1, y1, x2, y2) 
            revthresh = cv2.bitwise_not(thresh)
            im_bw = cv2.threshold(revthresh, 254, 255, cv2.THRESH_BINARY)[1]
            im_bw_inv = cv2.bitwise_not(im_bw)
            im_bw_inv = cv2.cvtColor(im_bw_inv, cv2.COLOR_BGR2GRAY)
            #skelmed = medial_axis(bw_bullshit)
            skelmed,distance = medial_axis(im_bw_inv,return_distance=True)
            skeleton_inv = skelmed*distance
            #skeleton_inv = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
            #length = cv2.countNonZero(skeleton_inv)
            #im_bw_inv_grayscale = cv2.cvtColor(im_bw_inv, cv2.COLOR_BGR2GRAY)
            #area = cv2.countNonZero(im_bw_inv_grayscale)
            skel_inv = cv2.bitwise_not(skeleton_inv)
            skel_inv= cv2.cvtColor(skel_inv, cv2.COLOR_GRAY2BGR)
            skel_inv[np.all(skel_inv != [255,255,255], axis=-1)] = [0,0,255]
            comp_imagesmall = cv2.bitwise_and(croppedLarge, skel_inv)
            maxY, maxX,channels = comp_imagesmall.shape
            Xdif = 150-maxX
            Ydif = 150-maxY
            image_padded = cv2.copyMakeBorder(skeleton_inv, 0, Ydif, 0, Xdif, cv2.BORDER_CONSTANT,value=0)
            #print(image_padded.shape)
            writer.write(image_padded)   
        #return(croppedLarge,area,length)
    #skel_padded = cv2.copyMakeBorder(skeleton_inv, (y1-10), (1080-(y2+10)), (x1-10), (1920-(x2+10)), cv2.BORDER_CONSTANT,value=0)
    #skel_padded_inv = cv2.bitwise_not(skel_padded)
    #skel_padded_inv= cv2.cvtColor(skel_padded_inv, cv2.COLOR_GRAY2BGR)
    #skel_padded_inv[np.all(skel_padded_inv != [255,255,255], axis=-1)] = [255,0,0]
    #comp_image = cv2.bitwise_and(img_base, skel_padded_inv)
    #img_base = comp_image        
        
        
def grabcut2(croppedImage,x1,y1,x2,y2):
    image = croppedImage
    #x1, y1, x2, y2, *_ = outputs
    mask = np.zeros(image.shape[:2], dtype="uint8")
    rect = (30,30,abs(x1-x2),abs(y1-y2))
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")
    (mask, bgModel, fgModel) = cv2.grabCut(image, mask, rect, bgModel,
        fgModel, iterCount=10, mode=cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    image2 = image*mask2[:,:,np.newaxis]
    return(image2)        

if __name__ == "__main__":
    # declare source directory and out path
    """
    IMG_DIR is path to the folder with the images
    OUT_PATH is the path to the csv file you would like the output to go to
    i.e './output/sample.csv'
    """
    VID_PATH = sys.argv[1]
    CSV_SORT_PATH = sys.argv[2]
    OUT_PATH = sys.argv[3]
    
    vid = cv2.VideoCapture(VID_PATH)
    total_frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    video_name = os.path.basename(VID_PATH).strip('.avi')
    out_video_path = f"{OUT_PATH}/{os.path.basename(VID_PATH).strip('.avi')}_SORT_worm_4_medthresh.avi"    
    
    #df = pd.read_csv(csv_path,names=('frame','label','x1', 'y1', 'w','h'))
    #df = df.drop([0])
    #df['x1']= df['x1'].astype(int)
    #df['y1']= df['y1'].astype(int)
    #df['w']= df['w'].astype(int)
    #df['h']= df['h'].astype(int)
    #df['x2']=df[['x1','w']].sum(axis=1)
    #df['y2']=df[['y1','h']].sum(axis=1)
    #df = df[['x1', 'y1', 'x2', 'y2']]
    
    
    df = pd.read_csv(CSV_SORT_PATH,names=('frame', 'x1', 'y1', 'x2', 'y2','label'))
    df = df.drop([0])
    df['X']=df[['x1','x2']].mean(axis=1)
    df['Y']=df[['y1','y2']].mean(axis=1)
    df = df[['frame', 'x1', 'y1', 'x2', 'y2','label']]
    unique = df["frame"].unique()
    
    
    for i in range(int(total_frame_count), 500, -1):
        vid.set(1, i-1)
        print(i)
        ret, frame = vid.read()
        if i == total_frame_count:
            height, width, channels = frame.shape
            #print(height, width)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(out_video_path, fourcc, 10, (150, 150), True)
    
        filtval = df['frame'] == i 
        #print(filtval)
        csv_outputs = np.asarray(df[filtval])[:,:]
        
        #print(csv_outputs)
        if csv_outputs is not None: 
            skeleton(csv_outputs,frame,writer)

        
        if i == 610:
            break
    writer.release()       
    
    
    
    
    
    
    
    
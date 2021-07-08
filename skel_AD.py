import os
import sys
import cv2
import csv
import tqdm
import pandas as pd
import random
import numpy as np
import time
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou



def analyzeSORT(df,threshold):
    #threshold = 45
    vc = df.label.value_counts()
    test = vc[vc > threshold].index.tolist()
    csv_outputs = []
    deadboxes = []
    fullboxes = []
    deadcalls = []
    for ID in test:        
        filtval = df['label'] ==ID
        interim = df[filtval]
        interimD = []
        interim2 = np.asarray(interim)
        fill = 0
        deadcount = 0
        deathspots = []
    for ID in test:        
        filtval = df['label'] ==ID
        interim = df[filtval]
        interimD = []
        interim2 = np.asarray(interim)
        fill = 0
        deadcount = 0
        for row in interim2:
            frameNA, x1A, y1A, x2A, y2A, labelA, deltaA, catagoryA, *_ = row
            if fill > 1:
                boxA = [x1A, y1A, x2A, y2A]
                boxB = [x1B, y1B, x2B, y2B]
                deltaA = bb_intersection_over_union(boxA, boxB) 
                #print(deltaA)
                if deltaA > 0.95:
                    deadcount += 1
                    #print(deadcount)
                #print(fill)
                if deadcount > 15:
                    catagoryA = 'dead'
                   # print(deadcount)
                if deadcount == 16:
                    if deadboxes == []:
                        deathspots.append([frameNA, x1A, y1A, x2A, y2A])     
                        deadboxes.append([x1A, y1A, x2A, y2A])  
                        csv_outputs.append((frameNA/72)+2)
                        deadcalls.append(labelA)
                    else:
                        notunique = 0
                        for box in deadboxes:
                            #print(box)
                            x1D, y1D, x2D, y2D, *_ = box
                            boxD = [x1D, y1D, x2D, y2D]
                            deltaD = bb_intersection_over_union(boxA, boxD)  
                            if deltaD > 0.3:
                                notunique = 1
                        if notunique == 0:
                            deathspots.append([frameNA, x1A, y1A, x2A, y2A])  
                            deadboxes.append([x1A, y1A, x2A, y2A])               
                            csv_outputs.append((frameNA/72)+2)

                    #print(deathtime)
                    #print(frameNA)
                #newRow = [frameNA, x1A, y1A, x2A, y2A, labelA, deltaA, catagoryA]
                #interimD.append(newRow)
            frameNB, x1B, y1B, x2B, y2B,labelB, deltaB, catagoryB, *_ = row
            fullboxes.append([frameNA, x1A, y1A, x2A, y2A, labelA, deltaA, catagoryA])  

            fill +=1
            #if deadcount == 16:
                #break

    #csv_outputs = pd.DataFrame(csv_outputs, columns = ['#desc'])
    #csv_outputs['neural'] = '1'
    fullboxes = pd.DataFrame(fullboxes, columns = ['frame','x1','y1','x2','y2','label','delta','catagory'])

    return(fullboxes)





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
        
def skeleton(df,frame,frame_count): 
    img_base = frame
    
    for output in df:
        #print(output)
        #output = [int(n) for n in output]
        frameN, x1, y1, x2, y2, label, *_ = output
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        buffer = 15
        croppedLarge = frame[(y1-buffer):(y2+buffer),(x1-buffer):(x2+buffer)]
        thresh = grabcut2(croppedLarge, x1, y1, x2, y2, buffer) 
        revthresh = cv2.bitwise_not(thresh)
        im_bw = cv2.threshold(revthresh, 254, 255, cv2.THRESH_BINARY)[1]
        im_bw_inv = cv2.bitwise_not(im_bw)
        #skeleton = skeletonize(im_bw_inv, method='lee')
        #skeleton_inv = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
        #length = cv2.countNonZero(skeleton_inv)
        #im_bw_inv_grayscale = cv2.cvtColor(im_bw_inv, cv2.COLOR_BGR2GRAY)
        #area = cv2.countNonZero(im_bw_inv_grayscale)
        #skel_inv = cv2.bitwise_not(skeleton_inv) #cheating change to swap skeleton with volume
        #skel_inv = cv2.bitwise_not(im_bw_inv) #cheating change to swap skeleton with volume
        #skel_inv= cv2.cvtColor(skel_inv, cv2.COLOR_GRAY2BGR)
        #skel_inv[np.all(skel_inv != [255,255,255], axis=-1)] = [0,0,255]
        #
        #im_bw[np.all(im_bw != [255,255,255], axis=-1)] = [100,255,0]
        #comp_imagesmall = cv2.bitwise_and(croppedLarge, im_bw)

        #comp_imagesmall = cv2.bitwise_and(croppedLarge, skel_inv)
        #maxY, maxX,channels = comp_imagesmall.shape
        #Xdif = 150-maxX
        #Ydif = 150-maxY
        #image_padded = cv2.copyMakeBorder(comp_imagesmall, 0, Ydif, 0, Xdif, cv2.BORDER_CONSTANT,value=0)
        # print(image_padded.shape)
        #writer.write(image_padded)   
        #return(croppedLarge,area,length)
        # skel_padded = cv2.copyMakeBorder(skeleton_inv, (y1-10), (1080-(y2+20)), (x1-10), (1920-(x2+20)), cv2.BORDER_CONSTANT,value=0)
        #print(im_bw_inv.shape())
        #print(skeleton_inv.shape())
        #height, width, channels = im_bw_inv.shape
        #print(height, width)
        skel_padded = cv2.copyMakeBorder(im_bw_inv, (y1-buffer), (1080-(y2+buffer)), (x1-buffer), (1920-(x2+buffer)),cv2.BORDER_CONSTANT,value=0)
        skel_padded_inv = cv2.bitwise_not(skel_padded)
        #skel_padded_inv= cv2.cvtColor(skel_padded_inv, cv2.COLOR_GRAY2BGR)
        img_base[np.all(skel_padded_inv != [255,255,255], axis=-1)] = [0,100,255]
        #print(skel_padded_inv.shape)
        #comp_image = cv2.bitwise_and(img_base, skel_padded_inv)
        #img_base = comp_image
        cv2.rectangle(img_base, (x1,y1), (x2,y2), (0,255,255), 2)
        #print(img_base.shape)
    #if frame_count == 1:
        #cv2.imwrite("C:/Users/benja/Desktop/test_skel2.png", img_base)
    #writer.write(frame)   
    return(img_base)
    #writer.write(img_base)   

def bodymorph(df,frame,frame_count): 
    img_base = frame
    
    for output in df:
        frameN, x1, y1, x2, y2, label, *_ = output
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        buffer = 15
        croppedLarge = frame[(y1-buffer):(y2+buffer),(x1-buffer):(x2+buffer)]
        thresh = grabcut2(croppedLarge, x1, y1, x2, y2, buffer) 
        revthresh = cv2.bitwise_not(thresh)
        im_bw = cv2.threshold(revthresh, 254, 255, cv2.THRESH_BINARY)[1]
        im_bw_inv = cv2.bitwise_not(im_bw)
        skel_padded = cv2.copyMakeBorder(im_bw_inv, (y1-buffer), (1080-(y2+buffer)), (x1-buffer), (1920-(x2+buffer)),cv2.BORDER_CONSTANT,value=0)
        skel_padded_inv = cv2.bitwise_not(skel_padded)
        img_base[np.all(skel_padded_inv != [255,255,255], axis=-1)] = [0,100,255]
        cv2.rectangle(img_base, (x1,y1), (x2,y2), (0,255,255), 2)  
    return(img_base)

     
def edgemorph(df,frame,frame_count): 
    img_base = frame
    
    for output in df:
        frameN, x1, y1, x2, y2, label, *_ = output
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        buffer = 15
        croppedLarge = frame[(y1-buffer):(y2+buffer),(x1-buffer):(x2+buffer)]
        v=np.percentile(croppedLarge,99)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(croppedLarge, lower, upper)
        kernel = np.ones((6,6),np.uint8)
        opening = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((3,3),np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(closing, connectivity=4)
        #print(nb_components)
        if nb_components == 0:
            print("no detection")
            continue
        if nb_components > 1:    
            max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])
            img2 = np.zeros(output.shape)
            img2[output == max_label] = 255
        else:
            img2 = closing
        img3 = np.uint8(img2)
        #img3 = cv2.bitwise_not(img3)
        img3 = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)

        skel_padded = cv2.copyMakeBorder(img3, (y1-buffer), (1080-(y2+buffer)), (x1-buffer), (1920-(x2+buffer)),cv2.BORDER_CONSTANT,value=0)
        skel_padded_inv = cv2.bitwise_not(skel_padded)
        img_base[np.all(skel_padded_inv != [255,255,255], axis=-1)] = [0,100,255]
        cv2.rectangle(img_base, (x1,y1), (x2,y2), (0,255,255), 2)  
    return(img_base)    
    
    
    
    
    
def grabcut2(croppedImage,x1,y1,x2,y2,buffer):
    image = croppedImage
    #x1, y1, x2, y2, *_ = outputs
    mask = np.zeros(image.shape[:2], dtype="uint8")
    rect = (buffer,buffer,abs(x1-x2),abs(y1-y2))
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")
    (mask, bgModel, fgModel) = cv2.grabCut(image, mask, rect, bgModel,
        fgModel, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
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
    out_video_path = f"{OUT_PATH}/{os.path.basename(VID_PATH).strip('.avi')}_skel.avi"    
    
          
    ####SORT####
    df = pd.read_csv(CSV_SORT_PATH,names=('frame', 'x1', 'y1', 'x2', 'y2','label','delta'))
    df['catagory'] = 'alive'
    outputs = analyzeSORT(df,threshold = 75)
    
    
    #df = pd.read_csv(CSV_SORT_PATH,names=('frame', 'x1', 'y1', 'w','h','label'))
    #df['x2']=df[['x1','w']].sum(axis=1)
    #df['y2']=df[['y1','h']].sum(axis=1)
    #df = df[['frame','x1', 'y1', 'x2', 'y2']]
    #df['label'] = 'alive'
    
    
    
    while (1):
        ret, frame = vid.read()
        frame_count = vid.get(cv2.CAP_PROP_POS_FRAMES)
        print(frame_count)
        if frame_count == 1:
            height, width, channels = frame.shape
            print(height, width)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(out_video_path, fourcc, 10, (width, height), True)
            print("vid starter")
        filtval = outputs['frame'] == frame_count
        #print(filtval)
        csv_outputs = np.asarray(outputs[filtval])[:,:]
        
        #print(csv_outputs)
        if csv_outputs is not None: 
            img_base = bodymorph(csv_outputs,frame,frame_count)
            writer.write(img_base)
        else:
            writer.write(frame)   
        #if frame_count == 10.0:
        #    break
    writer.release()       
    
    
    
    
    
    
    
    
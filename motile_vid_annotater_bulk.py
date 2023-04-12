import os
import sys
import cv2
import csv
import tqdm
import pandas as pd
from yolov3_core import *
from sort.sort import *



## initialize model
class YoloToCSV():
    def __init__(self, model, frame, frame_count):
        """
        model is a model object from YoloModelLatest class.
        img_path is the complete path to the image
        """
        self.model = model
        #self.img_path = img_path
        self.img = frame
        self.img_path = frame_count

    def get_annotations(self):
        # pass through img processor. Image and cut size.
        img_size = 416
        outputs = self.model.pass_model(self.img)
        self.outputs = outputs
        if outputs:
            #print(outputs)
            outputs = non_max_suppression_post(outputs, overlapThresh=0.1)
        return outputs

    def write_to_csv(self, out_path):
        """Writes outputs to CSV at out_path"""
        img_name = self.img_path
        outputs = self.get_annotations()
        df = self.pd_for_csv(outputs, img_name)
        #outputsSort = self.sort_update(outputs)
        ##print(outputsSort)
        #if self.img_path > 1:
        #    df = self.pd_for_sort_output(outputsSort, img_name)
        df.to_csv(out_path, mode='a', header=False, index=None)
        print(f"Wrote {img_name} to csv!")

    def draw_on_im(self, out_path, writer, text=None):
        """Takes img, then coordinates for bounding box, and optional text as arg"""
        img = self.img
        for output in self.outputs:
            output = [int(n) for n in output]
            x1, y1, x2, y2, *_ = output
            # Draw rectangles
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,0), 2)
            if text is not None:
                cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        ## write image to path
        #cv2.imwrite(out_path, img)
        writer.write(img)
    # creates pandas df for easy csv saving.
    @staticmethod
    def pd_for_csv(outputs, img_name = "name"):
        """Converts tensors to list that is added to pd df for easy writing to csv"""
        csv_outputs = []
        for output in outputs:
            x1, y1, x2, y2, *_ = output
            w = abs(x2-x1)
            h = abs(y2-y1)
            csv_outputs.append([img_name, x1.tolist(), y1.tolist(), w.tolist(), h.tolist(), "worm"]) # ideally change to list earlier bc now outputs is a mix of tensors and lists....
        out_df = pd.DataFrame(csv_outputs)
        # change header to datacells for R-shiny processing
        #out_df = out_df.set_axis(['dataCells1','dataCells2','dataCells3','dataCells4','dataCells5','class'], axis=1)
        return out_df

    def sort_update(self, outputs):
        fullOutputs = np.array(outputs)
        boxes_xyxy = fullOutputs[:,:4]
        track_bbs_ids = mot_tracker1.update(boxes_xyxy)
        return(track_bbs_ids)


    def pd_for_sort_output(self, outputs, img_name = "name"):
        csv_outputs = []
        for worm in outputs:
            worm = worm.astype(np.int32)
            x1 = worm[0]
            y1 = worm[1]
            x2 = worm[2]
            y2 = worm[3]
            name = worm[4]
            csv_outputs.append([img_name, name, x1, y1 ,x2, y2])
        out_df = pd.DataFrame(csv_outputs)
        return out_df

if __name__ == "__main__":
    # declare source directory and out path
    """
    IMG_DIR is path to the folder with the images
    OUT_PATH is the path to the csv file you would like the output to go to
    i.e './output/sample.csv'
    """

    VID_FOLD_PATH_DIR = sys.argv[1]
    print(VID_FOLD_PATH_DIR)
    #print(VID_FOLD_PATH_DIR)
    #OUT_FOLD_PATH = sys.argv[2]
    #VID_FOLD_PATH_DIR = os.listdir(VID_FOLD_PATH_DIR)
    vid_fold =  os.listdir(VID_FOLD_PATH_DIR)
    print(vid_fold)

    for fold_name in vid_fold:
        VID_FOLD_PATH = os.path.join(VID_FOLD_PATH_DIR, fold_name)
        OUT_FOLD_PATH = VID_FOLD_PATH
        vid_list = os.listdir(VID_FOLD_PATH)
        print(vid_list)
        for vid_name in vid_list:

            VID_PATH = os.path.join(VID_FOLD_PATH, vid_name)
            OUT_PATH = OUT_FOLD_PATH
            print(VID_PATH)
            ## Declare settings for nn
            ## make sure to change these prarameters for your work enviroment
            settings = {'model_def': "cfg/yolov3-spp-1cls.cfg",
                        'weights_path': "C:/Users/cdkte/Downloads/weights-20230322T172213Z-001/weights/best.pt",
                        'class_path': "cfg/classes.names",
                        'img_size': 608,
                        'iou_thres': 0.4,
                        'no_gpu': True,
                        'conf_thres': 0.1,
                        'batch_size': 6,
                        'augment': None,
                        'classes': None,
                        'version': 7}

            model = YoloModelLatest(settings)


            #img_list = os.listdir(IMG_DIR)


            vid = cv2.VideoCapture(VID_PATH)
            total_frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
            video_name = os.path.basename(VID_PATH).strip('.avi')


            csv_out_path = f"{os.path.join(OUT_PATH, video_name)}_yolo.csv"
            out_video_path = f"{OUT_PATH}/{os.path.basename(VID_PATH).strip('.avi')}_yolo.avi"


            while (1):
                ret, frame = vid.read()
                frame_count = vid.get(cv2.CAP_PROP_POS_FRAMES)
                if frame_count == 1:
                    height, width, channels = frame.shape
                    print(height, width)
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter(out_video_path, fourcc, 10, (width, height), True)
                if frame_count % 30 == 0:
                    ToCSV = YoloToCSV(model, frame, frame_count)
                    ToCSV.write_to_csv(csv_out_path)
                    img_out_path =  f"{os.path.join(OUT_PATH, video_name)}_{frame_count}.png"
                    ToCSV.draw_on_im(out_video_path,writer)
                if frame_count == total_frame_count:
                    break
            writer.release()

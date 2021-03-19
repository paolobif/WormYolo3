import os
import sys
import cv2
import csv
import tqdm
import pandas as pd
from yolov3_core import *
"""
This program takes path to a directory with images as an argument
then returns a single csv with all the bounding boxes for those images

Modify directory with images in IMG_DIR
Then modify output in OUT_PATH
"""

## initialize model
class YoloToCSV():
    def __init__(self, model, img_path):
        """
        model is a model object from YoloModelLatest class.
        img_path is the complete path to the image
        """
        self.model = model
        self.img_path = img_path
        self.img = cv2.imread(img_path)

    def get_annotations(self):
        # pass through img processor. Image and cut size.
        img_size = 416
        outputs = self.model.pass_model(self.img)
        self.outputs = outputs
        return outputs

    def write_to_csv(self, out_path):
        """Writes outputs to CSV at out_path"""
        img_name = os.path.basename(self.img_path)
        outputs = self.get_annotations()
        df = self.pd_for_csv(outputs, img_name)
        df.to_csv(out_path, mode='a', header=True, index=None)
        print(f"Wrote {img_name} to csv!")

    def draw_on_im(self, out_path, text=None):
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
        cv2.imwrite(out_path, img)

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
        out_df = out_df.set_axis(['dataCells1','dataCells2','dataCells3','dataCells4','dataCells5','class'], axis=1)
        return out_df




if __name__ == "__main__":
    # declare source directory and out path
    """
    IMG_DIR is path to the folder with the images
    OUT_PATH is the path to the csv file you would like the output to go to
    i.e './output/sample.csv'
    """
    IMG_DIR = sys.argv[1]

    OUT_PATH = sys.argv[2]

    ## Declare settings for nn
    ## make sure to change these prarameters for your work enviroment
    settings = {'model_def': "cfg/yolov3-spp-1cls.cfg",
                'weights_path': "weights/416_1_4_full_best200ep.pt",
                'class_path': "cfg/classes.names",
                'img_size': 608,
                'iou_thres': 0.6,
                'no_gpu': True,
                'conf_thres': 0.3,
                'batch_size': 6,
                'augment': None,
                'classes': None}

    model = YoloModelLatest(settings)
    

    img_list = os.listdir(IMG_DIR)

    for im_name in img_list:
        img_path = os.path.join(IMG_DIR, im_name)
        # Create object -- generates detections
        ToCSV = YoloToCSV(model, img_path)
        ToCSV.write_to_csv(OUT_PATH)

    ## for the last img, draw bounding boxes and write image to out dir to confirm outputs are correct
    img_out_path = os.path.join(os.path.dirname(OUT_PATH), "sample.png")
    ToCSV.draw_on_im(img_out_path)


    ## test ##
    #test_im = "data/test_data/exp328_22.png"

    #test = YoloToCSV(model, test_im)
    #test.write_to_csv(OUT_PATH)

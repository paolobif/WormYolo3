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
        frame_obj = ImageProcessor(self.img, out_size=self.model.img_size) # can also hardcode 608
        input_dict = frame_obj.image_slices
        outputs = self.model.pass_model(input_dict)
        return outputs

    def write_to_csv(self, out_path):
        """ Writes outputs to CSV at out_path
        """
        img_name = os.path.basename(self.img_path)
        outputs = self.get_annotations()
        df = self.pd_for_csv(outputs, img_name)
        df.to_csv(out_path, mode='a', header=True, index=None)
        print(f"Wrote {img_name} to csv!")

    @staticmethod
    # creates pandas df for easy csv saving.
    def pd_for_csv(outputs, img_name = "name"):
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
#    if sys.argv[0] is not None:
#        IMG_DIR = sys.argv[0]
#    else:
    IMG_DIR = "./data/test_data"

    OUT_PATH = "./data/output/test.csv"


    ## Declare settings for nn
    settings = {'model_def': "cfg/yolov3-spp-1cls.cfg",
                'weights_path': "weights/last.pt",
                'class_path': "416_1_4_full/classes.names",
                'img_size': 608,
                'iou_thres': 0.6,
                'no_gpu': True,
                'conf_thres': 0.3,
                'batch_size': 6,
                'augment': None,
                'classes': None}

    model = YoloModelLatest(settings)

    ## test ##
    test_im = "data/test_data/exp328_21.png"

    test = YoloToCSV(model, test_im)
    test.write_to_csv(OUT_PATH)

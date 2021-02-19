from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
#from d_utils.video_utils import *

import os
import sys
import time
import datetime
import cv2
import tqdm
import statistics
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms.functional as TF


## settings is a dictionary with model parameters
class YoloModelLatest():
    def __init__(self, settings):
        print(settings)
        self.model_def = settings['model_def']
        self.weights_path = settings['weights_path']
        self.class_path = settings['class_path']
        self.img_size = settings['img_size']
        self.iou_thres = settings['iou_thres']
        self.no_gpu = settings['no_gpu']
        self.conf_thres = settings['conf_thres']
        self.batch_size = settings['batch_size']
        self.augment = settings['augment']
        self.classes = settings['classes']

        self.cfg = check_file(self.model_def)
        self.names = check_file(self.class_path)
        self.load_model()

    def load_model(self):
        #LOAD gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #DEFIN MODEL
        model = Darknet(self.cfg, self.img_size)

        # Load weights
        weights = self.weights_path
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'], strict=False)
        else:  # darknet format
            print("Error: Not valid weights type.")


        # SET UP MODEL FOR TESTING
        model.to(device).eval()

        # Half precision
        # only cuda supported
        # model.half()

        classes = load_classes(self.names)
        self.model = model
        self.device = device

        print("Model Succesfully Loaded")

    ### PASSES LIST OF FRAMES THROUGH MODEL AND RETURNS BOUNDING BOXES
    def pass_model(self, im, cut_size=416):
        """ Takes image array and cut_size as arg. Cut size is the natural dimension of the img cut.
        Returns outputs with list of information on each bounding box in the image. """
        #print(f"number of slices bring processed: {len(dataset_dict)}")

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        keys = []  # Stores image paths
        img_detections = []  # Stores detections for each image index
        #print("\nPerforming object detection:")
        prev_time = time.time()

        dataset = CustomLoadImages(im, cut_size=cut_size, img_size=self.img_size) # key, image
        input_img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)
        _ = self.model(input_img.float())

        for batch_i, (key, input_img) in enumerate(dataset):
                #print(key, input_img.shape)
                #config input
                #input_img = Variable(input_img.type(Tensor))
                img = torch.from_numpy(input_img).to(self.device)
                img = img.float()
                img /= 255.0

                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                #pass through model:
                with torch.no_grad():
                    detections = self.model(img, augment=self.augment)[0]
                    #detections = self.model(input_img, augment=self.augment)[0]
                    #detections = detections.float()
                    detections = non_max_suppression(detections, self.conf_thres, self.iou_thres,
                                                    multi_label=False, classes=self.classes, agnostic=True)#0.4

                #save img and detection
                keys.append(key)
                img_detections.extend(detections)

        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        print("\tInference Time: %s" % (inference_time))

        outputs = []
        for img_i, (key, detections) in enumerate(zip(keys, img_detections)):
            img_shape = (cut_size, cut_size)
            # if detections exist in image.
            if detections is not None:
                print(f"{img_i} Slice: {img_i}, Cord: {key} --- worms: {len(detections)}")
                ## rescales boxes from upscaled size down to original img_shape
                detections = rescale_boxes(detections, self.img_size, img_shape)
                #unique_labels = detections[:, -1].cpu().unique()
                #n_cls_preds = len(unique_labels)
                #change to: for output in detection:
                for x1, y1, x2, y2, conf, cls_conf in detections:
                    raw_output = (x1, y1, x2, y2, conf, cls_conf)
                    output = self.rescale_bboxes(key, raw_output)
                    outputs.append(output)
            else:
                print(f"{img_i} Image: {key} --- worms: 0")
        return(outputs)
        #return(outputs)

    @staticmethod
    def rescale_bboxes(key, output):
        """ Rescales the bounding boxes from the slices to the correct cords
        on the full image. Returns translated cords in output list """
        # output is [(key), detection]
        mapX, mapY = key
        x1, y1, x2, y2, conf, cls_conf = output

        ax1, ax2 = x1 + mapX, x2 + mapX
        ay1, ay2 = y1 + mapY, y2 + mapY

        return([ax1, ay1, ax2, ay2, conf, cls_conf])


## creates maping generator objectls
class MapGenerator():
    """ Class generates a map grid with the specified cuts
        Takes an imput of either image or xy tuple"""
    def __init__(self, xy_shape, cut_size):
        self.cut_size = cut_size
        if type(xy_shape) != tuple:
            #print("input is an np img array")
            self.xy_shape = (xy_shape.shape[1], xy_shape.shape[0])
            #self.x_shape = img_xy.shape[1]
        else:
            self.xy_shape = xy_shape
            #self.y_shape = img_xy[1]

        self.paired_grid = [] # redundant....

        self.map_grid = self.generate_complete_map_grid()

    def generate_complete_map_grid(self):
        """Four passes of map_grid need to be run in order
        to cover entire area. Returns list of points for bounds.
        format: list of [(x1y1, x2y2), (x1y1, x2y2)...]"""
        xy_shape = self.xy_shape
        cut_size = self.cut_size

        shiftx = int(self.cut_size/2)
        shifty = int(self.cut_size/2)
        # Processes each of the 4 cuts neccesary to cover all the areas.
        pass1 = self.map_ar(xy_shape, cut_size, (0,0))
        pass2 = self.map_ar(xy_shape, cut_size, (shiftx, shifty))
        pass3 = self.map_ar(xy_shape, cut_size, (shiftx, 0))
        pass4 = self.map_ar(xy_shape, cut_size, (0, shifty))
        self.map_grids = [pass1, pass2, pass3, pass4]
        #self.map_grids = [pass1]

        complete_map_grid = []
        for map_grid in self.map_grids:
            paired_corners = self.convert_grid_to_cornerxy(map_grid)
            self.paired_grid.append(paired_corners)
            complete_map_grid.extend(paired_corners)

        return complete_map_grid

    # shiftxy can be modified when calling map_ar to create different arrangements of rectangles
    @staticmethod
    def map_ar(xy_shape, cut_size, shiftxy=(0,0)):
        """creates grid map for slicing - adjust shift to cover needed areas -current method
        is shift=(cut_size/2, cut_size/2) and (0, cut_size/2) and (cut_size, 0/2).
        IMPORTANT - cut_size must be div by 2"""
        x_size = xy_shape[0]
        y_size = xy_shape[1]

        x_shift = shiftxy[0]
        y_shift = shiftxy[1]

        map_ar = []

        # declare ranges
        x_start_range = 0 + x_shift
        y_start_range = 0 + y_shift

        # determines the range of each cut.
        second_pass_shift = (int(cut_size/2), int(cut_size/2))
        x_max_range = x_size + cut_size if shiftxy == (0,0) or shiftxy == second_pass_shift else x_size
        y_max_range = y_size + cut_size if shiftxy == (0,0) or shiftxy == second_pass_shift else y_size

        for x in range(0+x_shift, x_max_range, cut_size):
            for y in range(0+y_shift, y_max_range, cut_size):
                map_ar.append([x,y])

        return map_ar

    @staticmethod
    def convert_grid_to_cornerxy(map_grid):
        """ Takes the series of points and then returns point (x1,y1), (x2,y2) for each rectangle"""
        only_xs = [n[0] for n in map_grid]
        y_slice_count = only_xs.count(only_xs[0])

        paired_corners = []
        ## pair index is the matching corner to point
        for i, point in enumerate(map_grid):
            pair_index = i+y_slice_count+1

            # if the pair index is not the last of each chunck or the last group do ...
            if (i+1) % y_slice_count != 0 and pair_index <= len(map_grid):
                #print(i, pair_index, f"division:{(i+1)%y_slice_count}")
                pair = map_grid[pair_index]
                paired_corners.append([tuple(point), tuple(pair)])
            else:
                pass

        return(paired_corners)


class CustomLoadImages(MapGenerator):
    def __init__(self, img, cut_size=416, img_size=608):
        """ Takes image array. Cut size which is the size of the slice. Img size is the
        size of the image that it is upscaled to, before entering model.

        The map array is generated dictating where the cuts will be made.
        Everytime _getitem__ is called, the key(xcord, ycord) of the cut and cut img is returned.
        """
        MapGenerator.__init__(self, img, cut_size)
        self.img = img
        self.img_size = img_size
        self.cut_size = cut_size

    def __getitem__(self, index):
        """ Processes the image. Adds padding and upscales.
        Returns the key and the individual slice.
        """
        x1y1, x2y2 = self.map_grid[index]
        x1, y1 = x1y1
        x2, y2 = x2y2
        img_crop = self.img[y1:y2, x1:x2]
        print(x1y1, x2y2, f"shape {img_crop.shape}")
        # add padding if the image is not sized correctly
        if img_crop.shape[:2] != (self.cut_size, self.cut_size):
            img_crop = self.add_padding_to_square_img(img_crop, self.cut_size)
        # Upscale the image to img_zise and convert format for model.
        img = letterbox(img_crop, new_shape=self.img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        # Returns point for upper left corner and the adjusted img.
        return x1y1, img

    def __len__(self):
        return len(self.map_grid)

    @staticmethod
    def add_padding_to_square_img(img, cut_size):
        """Adds padding to the right and bottom of image to match cutsizexcutsize"""
        y_size, x_size = img.shape[:2]
        y_pad_amount = cut_size - y_size
        x_pad_amount = cut_size - x_size

        pad_img = np.pad(img, [(0,y_pad_amount), (0,x_pad_amount), (0,0)])

        return pad_img

##-------------------------------------------##

class ImageProcessor():
    def __init__(self, img, out_size=416, stride=416, train=False):
        self.img = img

        self.out_size = out_size
        self.stride = stride
        self.train = train

        self.xdim = img.shape[1]
        self.ydim = img.shape[0]

        self.image_slices = defaultdict()
        # [(x,y)] = {'image': img array, 'cord': upper (x,Y) of slice, 'bbs':list of outputs form nn}
        self.img_bbs = []

        self.cut_images()

    def cut_images(self):
        if self.train == True:
            stride = self.stride
        else:
            stride = self.out_size

        out_size = self.out_size
        xdim, ydim = self.xdim, self.ydim
        cut_images = []

        y_map = 0
        for Y in range(0, ydim, stride):
            x_map = 0
            for X in range(0, xdim, stride):
                im_slice = self.img[Y:Y+out_size, X:X+out_size]
                # adds apdding to the image to make it the correct shape
                if im_slice.shape[0:2] != (out_size, out_size):
                    im_slice = self.add_padding_to_square_img(im_slice, out_size)

                # create dictionary with key values for each slice
                self.image_slices[(x_map, y_map)] = {'image': im_slice, 'cord': (X,Y), 'bbs':[]}
                x_map += 1
            y_map += 1

    def rescale_bboxes(self, outputs):
        # output is [(key), detection]
        image_slices = self.image_slices

        for key, detection in outputs:
            mapX, mapY = image_slices[key]['cord']
            x1, y1, x2, y2, conf, cls_conf = detection

            ax1, ax2 = x1 + mapX, x2 + mapX
            ay1, ay2 = y1 + mapY, y2 + mapY

            self.img_bbs.append([ax1, ay1, ax2, ay2, conf, cls_conf])

    @staticmethod
    def add_padding_to_square_img(img, cut_size):
        y_size, x_size = img.shape[:2]
        y_pad_amount = cut_size - y_size
        x_pad_amount = cut_size - x_size

        pad_img = np.pad(img, [(0,y_pad_amount), (0,x_pad_amount), (0,0)])

        return pad_img

## Following functions perform post NN processing funcitons
def nearest_neighbor(point, centroids):
    """ Find the nearest centroid from a point. Point is bounding box center (x, y)
    centroids is a list of (x, y) cords of other bb centroids.
    Returns centroid (x, y) that was nearest."""
    point = np.array(point)
    # Find the distance to each point and store in indexed dictionary.
    distances = []
    for i, centroid in enumerate(centroids):
        centroid = np.array(centroid)
        dist = np.linalg.norm(point-centroid)  # Calcualtes Euclidian distance between points
        distances.append(dist)
    # Find the min distance
    indx = np.argmin(distances)
    return centroids[indx]


def calc_iou(point1, point2):
    """ Calculates the intersection over union for two points
    Each point is formated (x1, y1, x2, y2).
    Returns iou value between 0-1."""
    xA = max(point1[0], point2[0])
    yA = max(point1[1], point2[1])
    xB = min(point1[0], point2[0])
    yB = min(point1[0], point2[0])
    # Calculate intersection area

def outputs_to_centroid_dict(outputs):
    """ Takes list of outputs and calculates the centroid and returns dictionary with:
    {centroid(x, y): list(output)}"""
    centroids = {}
    for output in outputs:
        x1, y1, x2, y2, conf, cls_conf = output
        w = (x2 - x1) / 2
        h = (y2 - y1) / 2
        center = ((x1 + w), (y2 + h))
        centroids[center] = output
    return centroids


def filter_outputs():
    """ Takes list of outputs and eliminates duplicates that occured due to mapping process.
    Does so by finding closest centroid and then calculating iou. If iou is greater than
    set threshold it will remove the bounding box from the list of outputs.
    """


def draw_from_output(img, outputs, col=(255,255,0), text=None):
    """ Img is cv2.imread(img) and outputs are (x1, y1, x2, y2, conf, cls_conf)
    Returns the image with all the boudning boxes drawn on the img """
    for output in outputs:
        output = [int(n) for n in output]
        x1, y1, x2, y2, conf, cls_conf = output
        cv2.rectangle(img, (x1,y1), (x2,y2), col, 2)
        if text is not None:
            cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

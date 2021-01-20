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
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            load_darknet_weights(model, weights)


        # SET UP MODEL FOR TESTING
        model.to(device).eval()

        # Half precision
        # only cuda supported
        model.half()

        classes = load_classes(self.names)
        self.model = model
        self.device = device

        print("Model Succesfully Loaded")


    ### PASSES LIST OF FRAMES THROUGH MODEL AND RETURNS BOUNDING BOXES
    def pass_model(self, dataset_dict):
        #print(f"number of slices bring processed: {len(dataset_dict)}")

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        keys = []  # Stores image paths
        img_detections = []  # Stores detections for each image index
        #print("\nPerforming object detection:")
        prev_time = time.time()

        dataset = CustomLoadImages(dataset_dict, img_size=self.img_size) # key, image
        input_img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)
        _ = self.model(input_img.half())

        for batch_i, (key, input_img) in enumerate(dataset):
                # ignores edge images where there are no worms
                if input_img.shape[1:3] != (608,608):
                    # adds padding to the image thats not the correct shape
                    input_img = self.add_padding_to_square_img(input_img, 608)


                #print(key, input_img.shape)
                #config input
                #input_img = Variable(input_img.type(Tensor))
                img = torch.from_numpy(input_img).to(self.device)
                img = img.half()
                img /= 255.0



                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                #pass through model:
                #with torch.no_grad():
                detections = self.model(img, augment=self.augment)[0]
                #detections = self.model(input_img, augment=self.augment)[0]
                detections = detections.float()
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

            #print(img_i, key)
            slice = dataset_dict[key]['image']
            cord = dataset_dict[key]['cord']

            #img = cv2.imread(path)

            # if detections exist in image.
            if detections is not None:
                print(f"{img_i} Slice: {key}, Cord: {cord} --- worms: {len(detections)}")

                detections = rescale_boxes(detections, self.img_size, slice.shape[:2])
                #unique_labels = detections[:, -1].cpu().unique()
                #n_cls_preds = len(unique_labels)

                #change to: for output in detection:
                for x1, y1, x2, y2, conf, cls_conf in detections:
                    raw_output = (x1, y1, x2, y2, conf, cls_conf)
                    output = rescale_bboxes(key, raw_output, dataset_dict)
                    outputs.append(output)

            else:
                print(f"{img_i} Image: {key}, Cord: {cord} --- worms: 0")

        return(outputs)

        @staticmethod
        def add_padding_to_square_img(img, cut_size):
            y_size, x_size = img.shape[:2]
            y_pad_amount = cut_size - y_size
            x_pad_amount = cut_size - x_size

            pad_img = np.pad(img, [(0,y_pad_amount), (0,x_pad_amount), (0,0)])

        return pad_img


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def rescale_bboxes(key, output, dataset_dict):
    # output is [(key), detection]
    mapX, mapY = dataset_dict[key]['cord']
    x1, y1, x2, y2, conf, cls_conf = output

    ax1, ax2 = x1 + mapX, x2 + mapX
    ay1, ay2 = y1 + mapY, y2 + mapY

    return([ax1, ay1, ax2, ay2, conf, cls_conf])


class ImageInLine(Dataset):
    def __init__(self, frames, img_size=352):
        self.frames = frames
        self.frame_names = list(frames.keys())
        self.img_size = img_size

    def __getitem__(self, index):
        # frames dictionary is structured (x,y): {'image': ..., "bb": ...}
        key = self.frame_names[index]
        img0 = self.frames[key]['image']
        # padded resize
        img = letterbox(image, new_shape=self.img_size)[0]
        #img = resize(img, self.img_size)

        # convert iamge
        img = img[:, :, ::-1].transpose(2,0,1) # conver bgr to rgb
        img = np.ascontiguousarray(img)

        #print(key, img.shape)
        #if isinstance(img, list):
            #img = np.asarray(img)

        #img = TF.to_tensor(img)
        # make sure image is square
        # img, _ = pad_to_square(img, 0)
        # resize images
        #img = resize(img, self.img_size)

        return key, img

    def __len__(self):
        return len(self.frames)

class CustomLoadImages:
    def __init__(self, image_slices, img_size=608):
        self.image_slices = image_slices

        self.keys = list(image_slices.keys())
        #self.positions = [i[0] for i in image_and_pos]
        #self.images = [i[1] for i in image_and_pos]
        self.img_size = img_size

    def __getitem__(self, index):
        keys = self.keys
        key = keys[index]
        image = self.image_slices[key]['image']

        img = letterbox(image, new_shape=self.img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return key, img

    def __len__(self):
        return len(self.keys)


##-------------------------------------------##

class ImageProcessor():
    def __init__(self, img, out_size=352, stride=352, train=False):
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


def dict_to_input_lst(slice_dict):
    images = []
    keys = list(slice_dict.keys())
    for key in keys:
        image = slice_dict[key]['image']
        images.append(image)

    out = list(zip(keys, images))
    return out

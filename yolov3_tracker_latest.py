import os
import cv2
import csv
import time
import argparse
import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict

from utils.tracker import CentroidTracker
from yolov3_core import *

# not being used...
def load_images_to_array(im_dir):
    #data_ = defaultdict()
    data__ = []
    image_names = os.listdir(im_dir)
    for name in tqdm.tqdm(image_names, desc="Fetching Images"):
        img0 = cv2.imread(f"{im_dir}/{name}")
        data__.append(img0)
    return data__, image_names


def draw_on_im(img, x1, y1, x2, y2, conf, col, text=None):
    center_x = (x2-x1)/2 + x1
    center_y = (y2-y1)/2 + y1

    cv2.rectangle(img, (x1,y1), (x2,y2), col, 2)
    if text is not None:
        cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)


# creates pandas df for easy csv saving.
def pd_for_csv(outputs, img_name = "name"):
    csv_outputs = []
    for output in outputs:
        x1, y1, x2, y2, *_ = output
        w = abs(x2-x1)
        h = abs(y2-y1)
        csv_outputs.append([img_name, "worm", x1.tolist(), y1.tolist(), w.tolist(), h.tolist()]) # ideally change to list earlier bc now outputs is a mix of tensors and lists....
    out_df = pd.DataFrame(csv_outputs)
    # change header to datacells for R-shiny processing
    out_df = out_df.set_axis(['dataCells1','dataCells2','dataCells3','dataCells4','dataCells5','dataCells6'], axis=1)
    return out_df



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="cfg/yolov3-spp-1cls.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="416_8_9/best.pt", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="cfg/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.6, help="iou threshold required to qualify as detected def 0.4")
    parser.add_argument("--conf_thres", type=float, default=0.3, help="object confidence threshold def 0.001")
    parser.add_argument("--no_gpu", default=True, help="cuda enabled gpu")
    parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension of first layer of yolo")
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')

    # select input and output location also output options
    parser.add_argument('--out_path', type=str, default='output/custom')
    parser.add_argument("--data_path", type=str, default='data/samples', help="path the video file or to the directory with stack of images")
    parser.add_argument("--video", action='store_true', help="save the labled images as a video")
    ### track ###
    parser.add_argument("--track", action='store_true', help="processes images in reverse to make time of death calls")
    ### track ###
    parser.add_argument("--csv", action='store_true', help="save the bounding box data into a csv in the out directory")
    parser.add_argument("--csv2", action='store_true', help="save the bounding box data into a csv. one for all images")
    parser.add_argument("--img", action='store_true', help="store as image")
    parser.add_argument("--retro", action='store_true', help="processes images in reverse to make time of death call")
    opt = parser.parse_args()

    # create settings dictionaryprint(input_img.shape)
    settings = {'model_def': opt.model_def,
                'weights_path': opt.weights_path,
                'class_path': opt.class_path,
                'img_size': opt.img_size,
                'iou_thres': opt.iou_thres,
                'no_gpu': opt.no_gpu,
                'conf_thres': opt.conf_thres,
                'batch_size': opt.batch_size,
                'augment': opt.augment,
                'classes': opt.classes}

    # number depends on which weights being used.
    SLICE_SIZE = 480 #480 is a temp fix to solve worms not recognized towards bottom. must add padding in future. Ideal is 416

    # determine input type
    INPUT_VIDEO = bool
    if ".avi" in opt.data_path:
        INPUT_VIDEO = True
        print("--Input being processed as a video--")
    else:
        INPUT_VIDEO = False
        print("--Input is stack of images--")

    def save_img(img, outputs, file_path):
        pass


    # fork video out vs images out as the output currently not setup fror images to video
    OUT_VIDEO = opt.video
    if OUT_VIDEO == True:
        if INPUT_VIDEO: out_video_path = f"{opt.out_path}/{os.path.basename(opt.data_path).strip('.avi')}_anotated.avi"
        if not INPUT_VIDEO: out_video_path = f"{opt.out_path}/{os.path.basename(opt.data_path)}_anotated.avi"

        if os.path.exists(out_video_path):
            n = 1
            while os.path.exists(out_video_path):
                out_video_path = f"{out_video_path.strip('.avi')}_{n}.avi"
                n += 1
        # set up video capture and write
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(out_video_path, fourcc, 10, (1920, 1080), True) # currently set to 10fps
        print(f"generating video at {out_video_path}")


    # load yolov3 model and start processing
    Yolo = YoloModelLatest(settings)
    # init tracker. in future can impliment other trackers
    tracker = CentroidTracker() if opt.track else "none"


    # start parsing and processing
    if INPUT_VIDEO == True:
        vid = cv2.VideoCapture(opt.data_path)
        total_frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        video_name = os.path.basename(opt.data_path).strip('.avi')

        while (1):
            ret, frame = vid.read()
            frame_count = vid.get(cv2.CAP_PROP_POS_FRAMES)

            frame_obj = ImageProcessor(frame, out_size=SLICE_SIZE)
            input_dict = frame_obj.image_slices
            outputs = Yolo.pass_model(input_dict)
            print(f"Frame {frame_count}/{total_frame_count}")

            if opt.csv2 == True:
                new_name = f"frame_{int(frame_count)}"
                df = pd_for_csv(outputs, img_name=f"{new_name}")
                df.to_csv(f"{opt.out_path}/{video_name}.csv", mode='a', header=False, index=None)

            if opt.img == True:
                for output in outputs:
                    x1, y1, x2, y2, conf, cls_conf = output
                    draw_on_im(frame, x1, y1, x2, y2, conf, (100,255,0), text="Worm")
                cv2.imwrite(f"{opt.out_path}/frame{frame_count}_anotated.png", frame)

            if opt.video == True:
                for output in outputs:
                    x1, y1, x2, y2, conf, cls_conf = output
                    draw_on_im(frame, x1, y1, x2, y2, conf, (100,255,0), text="Worm")
                ### tracking ###
                if opt.track == True:
                    tracker_input = np.asarray(outputs)[:,:4]
                    tracker_output = tracker.update(tracker_input)
                    if tracker_output == None:
                        pass
                    else:
                        for id in tracker_output:
                            print(id)
                            x, y = tracker_output[id]
                            cv2.putText(frame, str(id), (x+15, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,102,102), 2)
                writer.write(frame)

    elif INPUT_VIDEO == False:
        start_time = time.time()
        file_names = sorted(os.listdir(opt.data_path))
        head_name = os.path.basename(opt.data_path)

        if opt.retro: file_names.reverse()

        for i, file_name in enumerate(file_names):
            print(file_name)
            frame = cv2.imread(f"{opt.data_path}/{file_name}")
            frame_obj = ImageProcessor(frame, out_size=SLICE_SIZE)
            input_dict = frame_obj.image_slices
            outputs = Yolo.pass_model(input_dict)
            print(f"\t Image: {i}/{len(file_names)}")

            if opt.csv == True:
                raw_name, extension = file_name.split(".")
                new_name  = f"exp{head_name}_{i}"
                # if slow can use shutil to copy and rename instead of writing a new image...
                if not os.path.exists(f"{opt.out_path}/NN_pretrain_im"):
                    os.mkdir(f"{opt.out_path}/NN_pretrain_im")
                if not os.path.exists(f"{opt.out_path}/NN_pretrain_csv"):
                    os.mkdir(f"{opt.out_path}/NN_pretrain_csv")

                cv2.imwrite(f"{opt.out_path}/NN_pretrain_im/{new_name}.{extension}", frame)

                csv_df = pd_for_csv(outputs, img_name=f"{new_name}.{extension}")

                if opt.csv2 == True:
                    csv_df.to_csv(f"{opt.out_path}/NN_pretrain_csv/{head_name}_NN.csv", mode='a', header=False, index=None)
                else:
                    csv_df.to_csv(f"{opt.out_path}/NN_pretrain_csv/{new_name}_NN.csv", header=True, index=None)

            # this should be removed
            if opt.csv2 == True:
                new_name = f"frame_{frame_count}"
                df = pd_for_csv(outputs, img_name=f"{new_name}")
                df.to_csv(f"{video_name}.csv", mode='a', header=True, index=None)

            if opt.img == True:
                for output in outputs:
                    x1, y1, x2, y2, conf, cls_conf = output
                    draw_on_im(frame, x1, y1, x2, y2, conf, (100,255,0), text="Worm")
                cv2.imwrite(f"{opt.out_path}/{file_name}_anotated.png", frame)

            if opt.video == True:
                for output in outputs:
                    x1, y1, x2, y2, conf, cls_conf = output
                    draw_on_im(frame, x1, y1, x2, y2, conf, (100,255,0), text="Worm")
                ### tracking ###
                if opt.track == True:
                    tracker_input = np.asarray(outputs)[:,:4]
                    tracker_output = tracker.update(tracker_input)
                    if tracker_output == None:
                        pass
                    else:
                        for id in tracker_output:
                            print(id)
                            x, y = tracker_output[id]
                            cv2.putText(frame, str(id), (x+15, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,102,102), 2)
                ### tracking ###
                writer.write(frame)

        finish_time = time.time()
        process_time = datetime.timedelta(seconds=finish_time-start_time)
        print(f"{len(file_names)} Images with \nimg: {opt.img} \nvideo: {opt.video} \ncsv: {opt.csv} \n took: {process_time} long")

# print('hello world!')

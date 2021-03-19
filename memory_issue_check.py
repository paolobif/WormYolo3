from yolov3_core import *
import matplotlib.pyplot as plt
import cv2

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
## path to test image
dir = "C:/Users/benja/Downloads/390_aligned/390_aligned"
img_names = os.listdir(dir)

for name in img_names:
    img = cv2.imread(os.path.join(dir, name))
    outputs = model.pass_model(img)
    print("pre", len(outputs))
    outputs = filter_outputs(outputs, thresh=0.85)
    outputs = non_max_suppression_post(outputs, overlapThresh=0.3)

    
    print("post", len(outputs))
    draw_from_output(img, outputs, col=(255,255,0))
    cv2.imwrite(f"C:/Users/benja/Downloads/390_aligned/390_aligned/outputs/filtered_{name}", img)

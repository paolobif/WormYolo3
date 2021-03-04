

This repo contains modified Ultralytics inference and training code for YOLOv3 in PyTorch. The code works on Linux, MacOS and Windows. Credit to Joseph Redmon for YOLO  https://pjreddie.com/darknet/yolo/.


## Requirements

Python 3.7 or later with all `requirements.txt` dependencies installed, including `torch >= 1.5`. To install run:
```bash
$ pip install -U -r requirements.txt
```

## How to use
* download weights from google drive link in weights folder
* look at "simple_test.ipynb" to test if all dependencies are installed correctly and to see the simplest way to pass the model on images is
* note: classes.names is in cfg folder

* using csv and img options:
-- $ python3 yolov3_tracker_latest.py -h
the only arguments that you should have to set are: weights, out_path, data_path, video, csv, img
-- note: currently video doesn't work
* example
if you wanted to generate csv file from the images you'd run:
-- $ python3 yolov3_tracker_latest.py --weights %path_to_weights% --out_path %save_folder% --data_path %images_folder% --csv
* if you also wanted to generate anotated png files you would add --img to the end.


## Data Info
* approximately 1068 images 1080x1920
* 56,047 individual worms
* --csv tag saves 1 csv per image in the format: img_name, x1, y1, w, h
## Citation

 https://pjreddie.com/media/files/papers/YOLOv3.pdf

*test**
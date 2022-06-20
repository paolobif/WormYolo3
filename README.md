# Worm YOLOv3
An implementation of [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) to locate and track ***C. elegans*** from experiment videos generated by the [WormBot](https://github.com/JasonNPitt/wormbot). Tracked locations are then used to make **time of death calls** and other biologically relevant metrics.

## Example
[![test](https://img.youtube.com/vi/pzxg0H6FQl4/0.jpg)](
https://youtu.be/pzxg0H6FQl4)
[(*click for video...*)](https://youtu.be/pzxg0H6FQl4)

##**Automated Lifespan Curves**
![Results](https://drive.google.com/uc?export=view&id=17aDlpJQs5MTnbJ5adh5O5VaTPIiQ6zPY)

##  Pipeline
1. `Videos exported from the experiment.`
<!-- ![RawImage](https://drive.google.com/uc?export=view&id=12lVwhj4M3lJ-vphTHwiZpAzlrpbshwtB) -->
2. `Bounding box detections are made from the videos.`
<!-- 3. ![Processed](https://drive.google.com/uc?export=view&id=1yIYDmaFXVnej_rslTmtlnQ9hPwHkFc7T) -->
3. `Death calls are made using the detections.`


---

### [Docker](https://docs.docker.com/compose/install/)

```bash

$ docker-compose build
```
* To run:
```bash
$ docker-compose run

# Will start running the pipeline.
# By default points to ~/data/vids and will save in ~data/results
# If you wish to modify: change command path in docker-compose.yml
```

### Pip
* Cuda and Python Versioning.

```
    nvidia/cuda: 11.0
    python: 3.8.10
```

* Install Dependencies

```bash
$ pip install -U -r requirements.txt
$ pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

* Download Wieghts

```bash
$ bash weights/download_yolov3_weights.sh
```

* To run:

```bash
$ python3 vid_annotater_bulk.py -- include params

# Will run with defualt params on specified path.
# Check docs if you wish to modify.
```


---

## Google Colab

***WIP:***
*Currently working on a test environmen for others to quickly experiemt with our pipeline.*


---

## Weights Info

* Dataset of 1068 images 1080x1920
* 29,628 individual worms
* [downlaod latest weights](https://www.dropbox.com/sh/xx4kalzjxrkej26/AABzftltaYpoQiyNhkwQQOqCa?dl=1)

* #### Training
  * Pre-trained with weights from the COCO dataset found [here](https://pjreddie.com/media/files/yolov3-spp.weights).
  * Each image is gridded into several 416x416 slices of overlaying crops.
    * Sample Cropping:

      | Crop1 | Crop2 | Crop3 | Crop4 |
      | ----- | ----- | ----- | ----- |
      |  ![Crops](https://drive.google.com/uc?export=view&id=1bgnw-oaV3q2784TXzcWQg_DbM1zkNSft)| ![Crops](https://drive.google.com/uc?export=view&id=14U1OxpQSdBbYyXIlC6cEclB3XiTBWoBn)| ![Crops](https://drive.google.com/uc?export=view&id=18VkqCZxylZ0PBe3Lj8cHHZr_VMph9Aoq)| ![Crops](https://drive.google.com/uc?export=view&id=18VkqCZxylZ0PBe3Lj8cHHZr_VMph9Aoq)

  * Images are then upscalled to 608x608px.
  * Model configuration -> [~/cfg/yolov3-spp-1cls.cfg](~/cfg/yolov3-spp-1cls.cfg).




## Train Custom Weights
* Our custom [Data Loader](https://github.com/paolobif/DataLoader-Worm-Yolo3) for for properly formatting custom training data
* Using the directory created by [Data Loader](https://github.com/paolobif/DataLoader-Worm-Yolo3)
  * ```bash
    $ python3 train.py --h

    # Specify desired parameters for training.
    # Point arguments to the respective files generated by the data loader

    # Example:
    $ python3 train.py --epochs 100 --cfg cfg/yolov3-spp-1cls.cfg --data your/folder.data --img-size 608 --rect --single-cls
    ```









----
## Relavent Links

 https://pjreddie.com/media/files/papers/YOLOv3.pdf


<!-- ![worm](https://drive.google.com/uc?export=view&id=182g1x387z_wbBZYfqR3Ny56zVho3IV-C) -->
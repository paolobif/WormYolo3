import os
from pixellib.custom_train import instance_custom_training
from pixellib.semantic import semantic_segmentation
from pixellib.instance import custom_segmentation
import image_analysis as ia
import sys
import cv2
import numpy as np



"""
Requires Tenserflow and Pixellib
Handles Pixellib processing for training and annotation.

python segmentation.py mode in_path out_path (training_model)
mode = 0: Train new model
  in_path = folder of images and jsons to train from
  out_path = folder to store the new models
  (training_model) = 0 or 1. 0 Trains from dark model, 1 trains from light model. Does nothing if not Training
mode = 1:
"""

# The models are too large to be uploaded on Github and will have to be downloaded seperately.

# Pixellib Model for Light Images
LIGHT_MODEL = "C:/Users/cdkte/Downloads/worm_segmentation/New_Training_Data/Light_Models/mask_rcnn_model.006-0.178200.h5"

# Pixellib Model for Dark Images
DARK_MODEL = "C:/Users/cdkte/Downloads/worm_segmentation/New_Training_Data/Dark_Models/mask_rcnn_model.008-0.663363.h5"


def trainFromFolder(model, input, output):
  """
  Trains AI models from the input folder and puts the most successful models in the output folder.
  Note that for this to work, there needs to be both images and JSON files in a Train folder and Test folder in the input folder
  This format is easily made using labelme

  Args:
    model: The file path to the model to start training from
    input: The file path to the data to train from
    output: The file path to store new models in
  """
  train_maskrcnn = instance_custom_training()
  train_maskrcnn.modelConfig(network_backbone = "resnet101", num_classes= 1, batch_size = 2)
  train_maskrcnn.load_pretrained_model(model)
  train_maskrcnn.load_dataset(input)
  train_maskrcnn.train_model(num_epochs = 300, augmentation=True,  path_trained_models = output)



def annotateSingle(input_path, output_path):
  """
  Annotates a single image

  Args:
    input_path: The raw image file
    output_path: The file where the annotated image should be stored
  """
  segment_image = custom_segmentation()
  segment_image.inferConfig(num_classes= 1, class_names= ["BG", ""])
  img = cv2.imread(input_path)
  h, w, colors = img.shape
  grayscale_matrix = np.zeros((h, w))
  for y in range(h):
    for x in range(w):
      rgb_values = img[y,x]
      grayscale_matrix[y,x] = np.min(rgb_values)
  if np.mean(grayscale_matrix) < 255/2:
    model = DARK_MODEL
  else:
    model = LIGHT_MODEL

  segment_image.load_model(model)
  segment_image.segmentImage(input_path, show_bboxes=False, output_image_name=output_path,
  extract_segmented_objects= False, save_extracted_objects=False)

def annotateFolder(input_path, output_path):
  """
  Annotates an entire folder of worm images and stores it in a new folder
  Uses LIGHT_MODEL and DARK_MODEL based on the average shade of each image.

  Args:
    input_path: The folder location storing the images relative to the python file
    output_path: The folder location to store the annotated images
  """

  all_image = os.listdir(input_path)
  os.mkdir(output_path)
  segment_light = custom_segmentation()
  segment_light.inferConfig(num_classes= 1, class_names= ["BG", ""])
  segment_light.load_model(LIGHT_MODEL)

  segment_dark = custom_segmentation()
  segment_dark.inferConfig(num_classes= 1, class_names= ["BG", ""])
  segment_dark.load_model(DARK_MODEL)

  for item in all_image:
    img = cv2.imread(input_path + "/"+item)
    h, w, colors = img.shape
    grayscale_matrix = np.zeros((h, w))
    for y in range(h):
      for x in range(w):
        rgb_values = img[y,x]
        grayscale_matrix[y,x] = np.min(rgb_values)
    if np.mean(grayscale_matrix) > 255/2:
      print("Light")
      segment_light.segmentImage(input_path+"/"+item, show_bboxes=False, output_image_name=output_path+"/Annotated_"+item,
      extract_segmented_objects= False, save_extracted_objects=False)
    else:
      print("Dark")
      segment_dark.segmentImage(input_path+"/"+item, show_bboxes=False, output_image_name=output_path+"/Annotated_"+item,
      extract_segmented_objects= False, save_extracted_objects=False)

if __name__ == "__main__":
  """
  segment_image = custom_segmentation()
  segment_image.inferConfig(num_classes= 1, class_names= ["BG", ""])
  img = cv2.imread("C:/641/1.0/641_2_1.0_x1y1x2y2_306_205_365_251.png")
  h, w, colors = img.shape
  grayscale_matrix = np.zeros((h, w))
  for y in range(h):
    for x in range(w):
      rgb_values = img[y,x]
      grayscale_matrix[y,x] = np.min(rgb_values)
  if np.mean(grayscale_matrix) < 255/2:
    model = DARK_MODEL
  else:
    model = LIGHT_MODEL

  segment_image.load_model(model)
  segment_image.segmentImage("C:/641/1.0/641_2_1.0_x1y1x2y2_306_205_365_251.png", show_bboxes=False, output_image_name="EVEN_BIGGER.png",
  extract_segmented_objects= False, save_extracted_objects=False)
  """
  try:
    # Console Input
    mode = sys.argv[1]
    folder_path = sys.argv[2]
    out_path = sys.argv[3]
    if mode == "0":
      train_from = folder_path
      train_to = out_path
      if sys.argv[4] == "0":
        print("Training From Dark")
        start_model = DARK_MODEL
      else:
        print("Training From Light")
        start_model = LIGHT_MODEL
      trainFromFolder(start_model, train_from,train_to)
    elif mode == "1":
      annotateFolder(folder_path,out_path)
    elif mode == "2":
      annotateSingle(folder_path,out_path)
    else:
      raise Exception
  except:
    try:
      mode = sys.argv[1]
      raise Exception("Invalid Input. Correct input: python segmentation mode in_path out_path\n\
        Modes: 0 = Training Mode, 1 = Annotate Folder, 2 = Annotate Single Image")
    except:
      # Manual Input
      train_from = None
      train_to = None
      trainFromFolder(DARK_MODEL, train_from,train_to)


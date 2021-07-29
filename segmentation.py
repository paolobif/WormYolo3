import os
import pixellib
from pixellib.custom_train import instance_custom_training
from pixellib.semantic import semantic_segmentation
from pixellib.instance import custom_segmentation

"""
Requires Tenserflow and Pixellib
"""


def trainFromFolder(input,output):
  """
  Trains AI models from the input folder and puts the most successful models in the output folder.
  Note that for this to work, there needs to be both images and JSON files in a Train folder and Test folder in the input folder

  input: The file path to the data to train from
  output: The file path to store info in
  """
  train_maskrcnn = instance_custom_training()
  train_maskrcnn.modelConfig(network_backbone = "resnet101", num_classes= 1, batch_size = 4, gpu_count = 1)
  train_maskrcnn.load_pretrained_model("models/mask_rcnn_model.002-0.633895.h5")
  train_maskrcnn.load_dataset(input)
  train_maskrcnn.train_model(num_epochs = 300, augmentation=True,  path_trained_models = output)



def annotateSingle(input_path, model_path, output_path):
  """
  Annotates a single image
  input: The raw image source
  output: The file where the annotated image should be stored
  """
  segment_image = custom_segmentation()
  segment_image.inferConfig(num_classes= 1, class_names= ["BG", ""])
  segment_image.load_model(model_path)
  segment_image.segmentImage(input_path, show_bboxes=False, output_image_name=output_path,
  extract_segmented_objects= False, save_extracted_objects=False)

def annotateFolder(input_path, model_path, output_path):
  """
  input_path: The folder location storing the images relative to the python file
  model_path: The path to the h5 AI model
  output_path: The folder location to store the annotated images
  """


  all_image = os.listdir(input_path)
  os.mkdir(output_path)
  segment_image = custom_segmentation()
  segment_image.inferConfig(num_classes= 1, class_names= ["BG", ""])
  segment_image.load_model(model_path)

  for item in all_image:
    segment_image.segmentImage(input_path+"\\"+item, show_bboxes=False, output_image_name=output_path+"\\Annotated_"+item,
    extract_segmented_objects= False, save_extracted_objects=False)

def annotateVideo(model_path):
  """
  Annotates a video
  Here as an example of how to implement, not as a function
  """
  test_video = custom_segmentation()
  test_video.inferConfig(num_classes= 1, class_names= ["BG", ""])
  test_video.load_model(model_path)
  test_video.process_video("694.avi", show_bboxes = True,  output_video_name="694_anno.avi", frames_per_second=25)



if __name__ == "__main__":
  file_directory = __file__
  file_folder = __file__.split("\\")
  file_folder.remove(file_folder[-1])
  file_folder = "\\".join(file_folder)
  os.chdir(file_folder)
  #trainFromFolder("Training_Data","models")
  #annotateFolder("4967.0","models\\mask_rcnn_model.002-0.633895.h5","Annotated_4967")
  annotateSingle("4967.0\\344_470_4967.0_x1y1x2y2_905_834_966_854.png", "models\\mask_rcnn_model.002-0.633895.h5", "edited.png")
  #annotateVideo("models\\mask_rcnn_model.001-0.565890.h5")
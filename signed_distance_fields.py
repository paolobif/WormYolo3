import numpy as np
import cv2
from matplotlib import pyplot as plt
import skfmm # Needs to be pip installed scikit-fmm

def identify_worm(base_img_path,anno_img_path):
    base_img = cv2.imread(base_img_path,cv2.IMREAD_GRAYSCALE)
    base_img2 = cv2.imread(base_img_path)

    #plt.show()
    try:
      anno_img = cv2.imread(anno_img_path,cv2.IMREAD_GRAYSCALE)
      diff = base_img != anno_img
      diff = diff.astype(float)

      #diff -= 0.5
      sd = skfmm.distance(diff,dx=1)
      return diff

    except:
      # If no worm is in the bounding box
      return None







#diff = identify_worm("C:/Users/cdkte/Downloads/day7_16_1025.0_x1y1x2y2_679_647_745_703.png","C:/Users/cdkte/Downloads/Annotated_day7_16_1025.0_x1y1x2y2_679_647_745_703.png")
#diff = diff.astype(float)
#sd = skfmm.distance(diff,dx=1)
#plt.imshow(sd)
#plt.colorbar()
#plt.show()


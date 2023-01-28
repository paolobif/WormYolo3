import cv2
import numpy as np
from matplotlib import pyplot as plt

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = False
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

params.minThreshold = 10
params.maxThreshold = 200

params.filterByColor = True
params.blobColor = 255

detector = cv2.SimpleBlobDetector_create(params)

def dropSmall(anno):
  anno = (anno*255).astype(np.uint8)
  mask = np.ones(anno.shape, dtype="uint8")

  def circleCrop(array, key_point:cv2.KeyPoint):
    if (key_point.size > 15):
        return
    else:
        center = (int(key_point.pt[0]),int(key_point.pt[1]))
        circle_mask = cv2.circle(mask, center, int(np.ceil(key_point.size)), 0, -1)
        array*= circle_mask
        #plt.imshow(circle_mask)
        #plt.show()

        pass

  keypoints = detector.detect(anno)
  keypoints = list(keypoints)
  #print(keypoints)
  keypoints.sort(key=lambda x: x.size)

  [circleCrop(anno,keypoint) for keypoint in keypoints]

  return anno/255
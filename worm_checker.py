import convert_image as ci
import numpy as np
import skeleton_cleaner as sc

"""
All functions return whether the worm is acceptable: True for a valid worm, False for an invalid worm
"""

# Area of worm: Between length and max width * length
def check_area_bounds(area, length, width):
  """
  Checks the area of the worm is within reasonable bounds

  Args:
      area: Worm Area
      length: Worm Length
      width: Worm Width

  Returns:
     Boolean: If the worm is within reasonable bounds, returns True. Otherwise, returns False
  """
  if (length < area) and (area < 2*width*length):
    return True
  else:
    return False



def check_number(worm_dict):
  """
  Tests whether there is more than 1 worm.

  Args:
      worm_dict: Dictionary describing the different worms

  Returns:
      Boolean: If there is 1 worm, returns True. Otherwise returns False.
  """
  if len(worm_dict) != 1:
    return False
  else:
    return True


# TODO: Connectivity



AREA_CONSTANT = 0.7
def check_relative_area(worm_matrix, area):
  """
  Checks whether the area is in reasonable bounds relative to the size of the image

  Args:
    worm_matrix: The matrix describing which pixels of the image are part of the worm
    area: The area of the worm

  Returns:
    Boolean: If the area is acceptable, returns True
  """
  width, height = worm_matrix.shape
  if area > AREA_CONSTANT * width * height:
    return False
  else:
    return True


"""
Could possibly work to detect forks (two worms that are being read as the same)
Requires testing every triplet of 'corners' but would probably catch forks and
the random blobs that sometimes show up

Would require testing to see how well this works and how long it takes to run

C:/Users/cdkte/Downloads/timelapse/data3/369/432/369_432_1_x1y1x2y2_962_864_1019_883.png

In the segmentation of pixels - If distinct endpoints, detect head and tail
See if confidence intervals are outputted and if certain threshold links back.

Different methods for different types of analysis, superimpose segmentations over video. highlight bad worms with red box
highlight good worms with blue box would be easier - try to get annotated worm on if possible
"""
CORNER_DISTANCE_CONSTANT = 1.5
def check_corners(worm_matrix, length):
  height, width = worm_matrix.shape
  pixel_dense = {}
  worm_min = np.inf
  pop_list = []
  for y in range(height):
    for x in range(width):
      if worm_matrix[y,x]:
        worm_near = ci.wormNearPixel(worm_matrix,y,x,search_range = 1)
        if worm_near < worm_min:
          pixel_dense[(y,x)] = worm_near
          worm_min = worm_near
          for item in pixel_dense:
            if pixel_dense[item] > worm_min:
              pop_list.append(item)
  for item in pop_list:
    pixel_dense.pop(item)
    while item in pop_list:
      pop_list.remove(item)
  distance_dict = {}
  for pix1 in pixel_dense:
    for pix2 in pixel_dense:
      for pix3 in pixel_dense:
        if pix1!=pix2 and pix1!=pix3 and pix2!=pix3:
          dist_sum = ci.pointDistance(pix1,pix2)+ci.pointDistance(pix1,pix3)+ci.pointDistance(pix2,pix3)
          if dist_sum > length*CORNER_DISTANCE_CONSTANT:
            return False

  return True

SKEL_LIMIT = 5
def check_skel_list(skel_list):
  """
  Determines if there is a reasonable number of points in the Skeleton List

  Args:
    skel_list: The list of points that make up the worms skeleton

  Returns:
    Boolean True if skeleton list is acceptable
  """
  if len(skel_list) < SKEL_LIMIT:
    return False
  else:
    return True

SHADE_CONSTANT = 30
def check_shade(shade):
  """
  Checks whether the image is bright enough for there to be a recognizable worm

  Args:
    shade: The average shade of the worm

  Returns:
    Boolean True if shade is acceptable
  """
  if shade < SHADE_CONSTANT:
    return False
  else:
    return True

WIDTH_HIGH_CONSTANT = 20
WIDTH_LOW_CONSTANT = 2
def check_width(width):
  """
  Checks whether the width is within reasonable bounds

  Args:
    width: The middle width of the worm

  Returns:
    Boolean True if width is acceptable
  """
  if width > WIDTH_HIGH_CONSTANT or width < 2:
    return False
  else:
    return True

AREA_CONSTANT = 100
def check_area(area):
  """
  Checks whether the area is within reasonable bounds

  Args:
    area: The total area of the worm

  Returns:
    Boolean True if area is acceptable
  """
  if area < AREA_CONSTANT:
    return False
  else:
    return True

# This would be the worm winding all the way around and then some.
CML_ANGLE_CONSTANT = 3 * np.pi
def check_cml_angle(cml_angle):
  """
  Checks whether the cumulative angle is within reasonable bounds

  Args:
    cml_angle: The cumulative angle of the worm's skeleton

  Returns:
    Boolean True if cml_angle is acceptable
  """
  if cml_angle < 0 or cml_angle > CML_ANGLE_CONSTANT:
    return False
  else:
    return True


LENGTH_CONSTANT = 300
def check_length(length):
  """
  Checks whether the length is within reasonable bounds

  Args:
    length: The length of the worm's skeleton

  Returns:
    Boolean true if length is acceptable
  """

  if length <= 0 or length > LENGTH_CONSTANT:
    return False
  else:
    return True

# How far to check in either direction
BRANCH_CONSTANT = 7
MIN_ANGLE = np.pi / 6
"""
Checks if the angle on any given side is too much.
"""
def check_for_branching(worm_matrix):
  """
  Checks whether the three most seperated corners are reasonably close.

  Args:
    worm_matrix: The matrix describing which pixels are part of the worm

  Returns:
    Boolean False if it detects branching
  """
  outline = ci.makeWormOutline(worm_matrix)
  w, h = outline.shape
  outline_array = []

  # Find a starting point
  stop = False
  for x in range(w):
    for y in range(h):
      if outline[x][y]:
        point = (x, y)
        stop = True
        break
    if stop:
      break
  start = point
  stop = False
  for x1 in range(x-1,x+1):
    for y1 in range(y-1,y+1):
      if outline[x1][y1] and (x1,y1)!=point:
        prev = point
        outline_array.append(prev)
        point = (x1, y1)
        if point == start:
          stop = True
          break
    if stop:
      break

  for i in range(BRANCH_CONSTANT,len(outline_array)-BRANCH_CONSTANT):
    if sc.getAngle(outline_array[i-BRANCH_CONSTANT],outline_array[i],outline_array[i+BRANCH_CONSTANT]) < MIN_ANGLE:
      return False
  return True

def check_worm(worm_dict,worm_matrix,outnumpy,skel_list):
  """
  Checks whether the worm is a reasonable sample. If it is not, throws an error.

  Args:
    worm_dict: The dictcionary describing all worms
    worm_matrix: The matrix describing the processed worm
    outnumpy: The list of values collected from the worm
    skel_list: The list of points that make up the estimated skeleton of the worm

  Returns:
    Nothing - Raises an Exception if the worm is invalid in some form
  """
  # Note that although the matrix, outnumpy, and skel_list could be derived from worm_dict,
  # we pass all these to the function to save processing time

  area = float(outnumpy[6])
  shade = float(outnumpy[7])
  length = float(outnumpy[9])
  mid_width = float(outnumpy[11])

  if not check_area_bounds(area,length,mid_width):
    raise Exception("Worm Area does not match max width and length:",area,length,mid_width)

  if not check_number(worm_dict):
    raise Exception("More than an acceptable number of worms")
  if not check_relative_area(worm_matrix, area):
    raise Exception("Worm takes up too much of image")
  if not check_corners(worm_matrix, length):
    raise Exception("Three furthest points too far apart")
  if not check_shade(shade):
    raise Exception("Worm too dark!")

  if not check_area(area):
    raise Exception("Area not in reasonable bounds:", area)

  if not check_cml_angle(float(outnumpy[8])):
    raise Exception("Cumulative Angle not in reasonable bounds:", outnumpy[8])

  if not check_length(length):
    raise Exception("Length not in reasonable bounds:", length)

  if not check_width(mid_width):
    raise Exception("Worm width not within reasonable bounds:", mid_width)

  if not check_skel_list(skel_list):
    raise Exception("Worm has an unreasonable starting point")
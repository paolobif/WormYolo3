from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

def xyxy2xywh(xyxy):
    x1, y1, x2, y2 = xyxy
    w = (x2-x1)/2
    h = (y2-y1)/2
    center_X = x1 + w
    center_Y = y1 + h

    return(center_X, center_Y, w, h)

class CentroidTracker():
    def __init__(self, maxDis=3):
        self.nextWormID = 0
        self.worms = OrderedDict()
        self.dissapeared = OrderedDict()

        self.maxDis = maxDis

    def register(self, centroid):
        centerX, centerY = centroid

        # 1 = alive 0 = dead
        self.worms[self.nextWormID] = (centerX, centerY)
        self.dissapeared[self.nextWormID] = 0
        self.nextWormID += 1
        # initializing countdown to see if object dissapears.

    def deregister(self, wormID):
        # remove object from dict.
        del self.worms[wormID]
        del self.dissapeared[wormID]

    def update(self, rects):
        # rect is tuple x1, y1, x2, y2
        # check to see if there is a bounding box
        if len(rects) == 0:
            for wormID in list(self.dissapeared.keys()):
                # add to the maxDis counter
                self.dissapeared[wormID] = self.dissapeared + 1


                if self.dissapeared[wormID] > self.maxDis:
                    self.deregister(wormID)

            # ends loop early if rect is empty
            return (self.worms)

        # init array size len of input x (x,y)-of each centroid
        inputCentroids = np.zeros((len(rects),2), dtype="int")

        for (i, rect) in enumerate(rects):
            # convert to centerx and center y
            center_X, center_Y, w, h = xyxy2xywh(rect)
            inputCentroids[i] = (center_X, center_Y)

        # init first round of objects to be Tracked
        if len(self.worms) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        # if its alraedy been init... get euclid min and update dict
        else:
            wormIDs = list(self.worms.keys())
            wormCentroids = list(self.worms.values())
            #wormCentroids = [worm_values[0] for value in worm_values]

            # get euclid distance between all existing and input points
            eu_distance = dist.cdist((wormCentroids), inputCentroids)
            # sorted indecies of min values
            rows = eu_distance.min(axis=1).argsort()
            # return sorted values
            cols = eu_distance.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row,col) in zip(rows, cols):
                # check if index has already been checked.
                if row in usedRows or col in usedCols:
                    continue

                # match the worms and update the position
                wormID = wormIDs[row]
                self.worms[wormID] = inputCentroids[col]
                self.dissapeared[wormID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, eu_distance.shape[0])).difference(usedRows)
            unusedCols = set(range(0, eu_distance.shape[1])).difference(usedCols)

            # mark dissapeared worms if there are less worms inputed than in prev frame
            if eu_distance.shape[0] >= eu_distance.shape[1]:
                for row in unusedRows:
                    wormID = wormIDs[row]
                    self.dissapeared[wormID] += 1

                    # check if worm has fully exited frame
                    if self.dissapeared[wormID] > self.maxDis:
                        self.deregister(wormID)

            # if there are more worms inputed then must register new worms
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

            #print("used:", usedCols)
            #print("unused:",unusedCols)
            return(self.worms)

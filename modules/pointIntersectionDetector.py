import numpy as np
import cv2


class PointIntersectionDetector():

    ''' 
        The following class implements the algorithm to detect the bottom 20% bounding box intersection with 
        the ROI and issue alert based on it
    '''

    def __init__(self, roiImgs, width, height):
        self.roiImgs = roiImgs
        self.width = width
        self.height = height
        
    def _intersectionDetector(self, boxes):
        alerts = []

        if len(boxes) > 0 :
            # calculate the center of x 
            cx = (boxes[:,0] + boxes[:,2])//2
            rad = ((boxes[:,3] - boxes[:,1])*0.10).astype(int)
            # adjust the center of the circles
            cy = boxes[:,3] - rad
            cent = np.hstack((cx.reshape(-1,1), cy.reshape(-1,1))).astype(int)
            
            
            
            # check if the bounding boxes intersects with the ROI
            for c, r in zip(cent, rad):
                x,y = c
                img = np.zeros((self.height, self.width))
                cv2.circle(img, (x,y), r, 255, -1)
                alert = np.logical_and(self.roiImgs, img)
                alerts.append(alert.any())
        
        return alerts  
        
    def process(self, frames):

        # process all the imput frames and return the results
        
        frameAlerts = [self._intersectionDetector(boxes) for boxes in frames]
        
        return frameAlerts
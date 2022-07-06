import numpy as np
import cv2

class PerAreaIntersectionDetector():

    '''
        The following class implements duncan's algorithm to detect if a bounding box intersects with the ROi
        based on the area of ROI and bunding box intersection 
    '''
    
    def __init__(self, roiImgs, width, height):
        self.roiImgs = roiImgs
        self.width = width
        self.height = height
    
    def _getIntersectionPercentage(self, roiImg, box, threshold):
        
        # get the bounding box and check if the roi and box intersect
        x,y,w,h = box
        intersctionPerc = 0
        img = np.zeros((self.height, self.width))
        cv2.rectangle(img, (x,y), (w,h), 1, -1)
        intr = np.logical_and(img, roiImg)
        
        # calculate the interesection percentage if the there is any intersection
        if intr.any():
            roiArea = roiImg.sum()
            boxArea = (w-x)*(h-y)
            interArea = intr.sum()
            
            if boxArea < roiArea:
                intersctionPerc = interArea/boxArea
            else:
                intersctionPerc = interArea/roiArea
        

        return intersctionPerc >= threshold
    
    def process(self, frames, threshold=0.5):
        
        # process all the imput frames and return the results

        frameAlerts = []
        for boxes in frames:
            alerts = []
            for box in boxes:
                alert = np.where(np.array([self._getIntersectionPercentage(roi, box, threshold) for roi in self.roiImgs]) > 0)[0]
                alert = alert[0] if len(alert) > 0 else 0
                alerts.append(alert) 
                #perc[perc > threshold] = True
            frameAlerts.append(alerts)    
            
        return frameAlerts  
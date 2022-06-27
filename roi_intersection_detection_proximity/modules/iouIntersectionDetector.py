import numpy as np
import cv2

class IOUIntersectionDetector():
    
    '''
        This class implements the IOU based intersection between the bounding box and ROI 
        ** works well if the ROI and bounding box has comparable areas **

    '''
    def __init__(self, roiImgs, width, height):
        self.roiImgs = roiImgs
        self.width = width
        self.height = height
        
        
    def _iouItersection(self, box, threshold):
        
        # calculate the IOU of bounding boxe over the ROI's
        x,y,w,h = box
        img = np.zeros((self.height, self.width))
        cv2.rectangle(img, (x,y), (w,h), 1, -1)
        boxArea = (w-x)*(h-y)
        roiArea = np.array([roiImg.sum() for roiImg in self.roiImgs])
        interArea = np.array([np.logical_and(roiImg, img).sum() for roiImg in self.roiImgs])
        
        iou = interArea/(roiArea + boxArea)
        
        # ceck if the iou is above the threshold 
        return (iou > threshold).any()
        
        
    def process(self, frames, threshold):
        # process all the imput frames and return the results
        frameAlerts = []
        for boxes in frames:
            alert = [self._iouItersection(box, threshold) for box in boxes]
            frameAlerts.append(alert)
            
        
        return frameAlerts
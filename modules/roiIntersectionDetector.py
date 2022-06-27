import numpy as np
import cv2
from .iouIntersectionDetector import IOUIntersectionDetector
from .perAreaIntersectionDetector import PerAreaIntersectionDetector
from .pointIntersectionDetector import PointIntersectionDetector


class ROIDetector():

    '''

        This is the main class to integrate all the different intersection detectors together and process the results

    '''
    
    def __init__(self, config):
        
        self.roi = config["roi"]
        self.width = config['width'] 
        self.height = config['height']
        self.roiImgs = self._drawPloygon()
        self.trigger = self._getTrigger(config['trigger'])
    
    # retrun the different intersection detection algorithm based on the user choice
    def _getTrigger(self, trigger):
        
        if trigger == 'iou':
            return IOUIntersectionDetector(self.roiImgs, self.width, self.height)
        elif trigger == 'area':
            return PerAreaIntersectionDetector(self.roiImgs, self.width, self.height)
        elif trigger == 'point':
            return PointIntersectionDetector(self.roiImgs, self.width, self.height)
        else :
            raise Exception('invalid trigger for roi interaction detection')
    
    # create the roi image based on the roi information 
    def _drawPloygon(self):
        
        img = np.zeros((self.height, self.width))
        imgarr = []
        for cnt in self.roi:
            i = cv2.drawContours(img.copy(), [cnt], -1, color=1, thickness=cv2.FILLED)
            imgarr.append(i)
            
        return np.array(imgarr)    


    def process(self, params):
        alert = self.trigger.process(*params)
        return alert
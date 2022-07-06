from unicodedata import category
import cv2
from .objectDetector import YoloDetector
import numpy as np
import os

class VideoReader():
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.yolo = YoloDetector(cfg['weight'], cfg['config'], cfg['class_names'], cfg['detector_threshold'])

        
    def process_video(self, video):
        
        cap = cv2.VideoCapture(video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        bounding_boxes, categories, confidences = [], [], []
        # counter = 0
        # if not os.path.exists('temp'):
        #     os.makedirs('temp')
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read() 
            if ret == True:

                boxes, class_name, confd = self.yolo.detect_boxes(frame)
                boxes, class_name = np.array(boxes, dtype=int), np.array(class_name)
                if(len(boxes) > 0):
                    boxes[:,2] = boxes[:,0] + boxes[:,2]
                    boxes[:,3] = boxes[:,1] + boxes[:,3]
                temp_box, cat, temp_cof = [], [], []
                for box, c, cfd in zip(boxes, class_name, confd):
                    if (c in self.cfg['target_class_roi']) or (c in self.cfg['target_class_nm']):
                        temp_box.append(box)
                        cat.append(c) 
                        temp_cof.append(cfd)
                         
                bounding_boxes.append(np.array(temp_box, dtype=int))
                #bounding_boxes_nm.append(np.array(temp_box_nm, dtype=int))
                categories.append(np.array(cat))
                confidences.append(np.array(temp_cof))
                #image_name = 'temp/' + str(counter) + '.jpg'
                #cv2.imwrite(image_name, frame)
                #frames.append(image_name)
                #counter += 1

            else: 
                break
            
        cap.release()       
        cv2.destroyAllWindows()   
        
        return bounding_boxes, categories, confidences, fps
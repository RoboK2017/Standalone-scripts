import numpy as np 
from scipy.spatial import distance_matrix
import cv2

class ProximityDetector():
    def __init__(self, rois:list, width:int, height:int, threshold:float):
        self.roi = rois
        self.width = width
        self.height = height
        self.threshold = threshold
        self.roi_imgs = self._drawPloygon()

    def _drawPloygon(self):
        
        img = np.zeros((self.height, self.width))
        imgarr = []
        for cnt in self.roi:
            i = cv2.drawContours(img.copy(), [cnt], -1, color=1, thickness=cv2.FILLED)
            imgarr.append(i)
            
        return np.array(imgarr)     

    def _getIntersectionPercentage(self, box:np.array, roiImg:np.array):
        
        # get the bounding box and check if the roi and box intersect
        #roiImg = self.roi_imgs
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
        

        return intersctionPerc >= self.threshold

    def _cal_iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / min(boxAArea, boxBArea)
        # return the intersection over union value
        return iou
        
    def _testProximity(self, cents:np.array, rads:np.array, 
            classes:np.array, target_pair:dict, boxes:np.array, 
            iou_thresh:float, outlier_priority:int):

        dis = distance_matrix(cents, cents)
        np.fill_diagonal(dis, 100000)
        idx = np.where(dis < rads)
        res = np.zeros(len(boxes), dtype=int)
        nm_roi = np.zeros(len(boxes), dtype=int)
        for x,y in zip(idx[0],idx[1]):
            k = classes[x]+','+classes[y]
            if k in target_pair.keys(): 
                i1 = np.where(np.array([self._getIntersectionPercentage(boxes[x], r) for r in self.roi_imgs ]) > 0)[0]
                i2 = np.where(np.array([self._getIntersectionPercentage(boxes[y], r) for r in self.roi_imgs ]) > 0)[0]

                i1 = i1[0] if len(i1) > 0 else 0
                i2 = i2[0] if len(i2) > 0 else 0
                nm_roi[x], nm_roi[y] = i1, i2

                if (i1 or i2):

                    if [classes[x], classes[y]] in [["person", "forklift"], ["forklift", 'person']]:
                        iou = self._cal_iou(boxes[x], boxes[y])
                        #print('person forklift', iou)
                        if iou > iou_thresh :
                            res[x], res[y] = outlier_priority, outlier_priority
                    else:
                        if res[x] < target_pair[k]: 
                            res[x] = target_pair[k]
                            
                        if res[y] < target_pair[k]: 
                            res[y] = target_pair[k]    
                
                #print('person in forklift')
                # x1,y1,w1,h1 = boxes[x]
                # x2,y2,w2,h2 = boxes[y]

                # if (x1 < x2 and y1 < y2 and w1 > w2 and h1 > h2) :

                #     res[x], res[y] = 1, 1
                # elif (x1 > x2 and y1 > y2 and w1 < w2 and h1 < h2) :
                #     res[x], res[y] = 1, 1
                
        return res, nm_roi
        
    def process(self, frames:list, class_arr:list, target_pair:dict, alpha:float, iou_thresh:float, outlier_priority:int):
        result, nm_roi, miss_type = [], [], 0
        for boxes, classes in zip(frames, class_arr):
            if len(boxes) > 0:
                cx = (boxes[:,0] + boxes[:,2])//2
                r = abs((boxes[:,0] - boxes[:,2])//2)
                cy = boxes[:,3] - r
                centroid = np.hstack((cx.reshape(-1,1), cy.reshape(-1,1)))
                r = r.reshape(-1,1) + r + alpha*r
                res, roi_flag = self._testProximity(centroid, r, classes, target_pair, boxes, iou_thresh, outlier_priority)

                #class_filter = np.isin(classes, self.nm_classes)
                result.append(res)
                nm_roi.append(roi_flag)
               # print(res)
                if max(res) > miss_type:
                    miss_type = max(res)


            else:
                result.append([])
                nm_roi.append([])

        return result, nm_roi, miss_type   
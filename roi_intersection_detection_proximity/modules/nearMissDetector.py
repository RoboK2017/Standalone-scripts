import numpy as np 
from scipy.spatial import distance_matrix

class ProximityDetector():
    def __init__(self):
        pass
        
    def _testProximity(self, cents:np.array, rads:np.array, classes:np.array, target_pair:dict, boxes:np.array):
        dis = distance_matrix(cents, cents)
        np.fill_diagonal(dis, 100000)
        idx = np.where(dis < rads)
        res = np.zeros(len(boxes), dtype=int)
        for x,y in zip(idx[0],idx[1]):
            k = classes[x]+','+classes[y]
            if k in target_pair.keys():  
                if res[x] < target_pair[k]: 
                    res[x] = target_pair[k]
                    
                if res[y] < target_pair[k]: 
                    res[y] = target_pair[k]    
                
            if [classes[x], classes[y]] in [["person", "forklift"], ["forklift", 'person']]:
                x1,y1,w1,h1 = boxes[x]
                x2,y2,w2,h2 = boxes[y]
                if (x1 < x2 and y1 < y2 and w1 > w2 and h1 > h2) :
                    res[x], res[y] = 0, 0
                elif (x1 > x2 and y1 > y2 and w1 < w2 and h1 < h2) :
                    res[x], res[y] = 0, 0
                
        return res
        
    def process(self, frames:list, class_arr:list, target_pair:dict, alpha:float):
        result = []
        for boxes, classes in zip(frames, class_arr):
            if len(boxes) > 0:
                cx = (boxes[:,0] + boxes[:,2])//2
                r = abs((boxes[:,0] - boxes[:,2])//2)
                cy = boxes[:,3] - r
                centroid = np.hstack((cx.reshape(-1,1), cy.reshape(-1,1)))
                r = r.reshape(-1,1) + r + alpha*r
                res = self._testProximity(centroid, r, classes, target_pair, boxes)


                result.append(res)

            else:
                result.append([0])

        return result   
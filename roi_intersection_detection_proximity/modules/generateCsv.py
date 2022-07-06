import numpy as np 
import random, string
import pandas as pd

class CSVGenerator():
    
    def __init__(self, bounding_boxes:list, categories:list, 
                pred_confd:list, roi_result:list, fps:int, 
                rev_map:dict, width:int, height:int):

        self.rev_map = rev_map
        self.bounding_boxes, self.categories, self.roi_result = bounding_boxes, categories, roi_result
        self.nm_results = [[]]*len(categories)
        self.final_folder = ['']*len(categories)
        self.seq_id = ['']*len(categories)
        self.inp_fps = fps
        self.pred_confd = pred_confd
        self.nm_roi = [[]]*len(categories)
        self.width = width
        self.height = height
    
    def update(self, nm_result:list, idx:list, final_fol:str, nm_roi:list):
        seq = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=10))
        for i, nm, nm_r in zip(idx, nm_result, nm_roi):
            self.nm_results[i] = nm
            self.final_folder[i] = final_fol
            self.seq_id[i] = seq 
            self.nm_roi[i] = nm_r

        return seq    
        #print(len(self.nm_results))


    def _createOutputArray(self):
        output = []
        hh, mm, ss, ms = 0, 0, 0, 0
        
        for i in range(len(self.categories)):
            
            boxes, classes, roi_results = self.bounding_boxes[i], self.categories[i], self.roi_result[i]
            nm_results, folder, seq_id = self.nm_results[i], self.final_folder[i], self.seq_id[i]
            nm_roi, pred_conf = self.nm_roi[i], self.pred_confd[i]
            
            ms += 1000//self.inp_fps
            ss += ms//1000 
            mm += ss//60
            hh += mm//60
            
            ms = ms%1000
            ss = ss%60
            mm = mm%60
            
            timestamp = '{}:{}:{}::{}'.format(hh,mm,ss,ms)

            if len(roi_results) != len(nm_results):
                nm_results = [0]*len(roi_results)
                nm_roi = [0]*len(roi_results)
                
            
            for b, c, roi, nm, nm_r, confd in zip(boxes, classes, roi_results, nm_results, nm_roi, pred_conf):
                
                nm = self.rev_map[nm] if nm > 0 else ''
                nm_r = 'nm_{}'.format(nm_r) if nm_r > 0 else ''
                roi = 'roi_{}'.format(roi) if roi > 0 else ''

                output.append([i, timestamp, self.width, self.height, b[0], b[1], b[2], b[3], c, confd, roi, nm_r, nm, seq_id, folder])
                

           # print(roi_results)    
            if len(boxes) == 0:
                output.append([i, timestamp, self.width, self.height, '', '', '', '', '', '', '','','', seq_id, folder])
        
        return output
        
    def generateCsv(self, filename:str):
        
        output = self._createOutputArray()
        cols=['frame_no', 'timestamp', 'frame_width', 'frame_height', 'x_min', 'y_min', 'x_max', 'y_max', 
              'category', 'confidence', 'roi_result', 'nm_roi', 'near_miss', 'sequence_id', 'output_folder']
        df = pd.DataFrame(output, columns=cols)
        
        df.to_csv(filename, index=False)
        
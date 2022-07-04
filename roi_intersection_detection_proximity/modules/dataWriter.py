import numpy as np
from .nearMissDetector import ProximityDetector
import cv2
import os
import random, string
#from moviepy.editor import VideoFileClip

class DataWriter():
    
    def __init__(self, config):
        self.width = config['width']
        self.height = config['height']
        self.roi = config['roi']
        self.base_path = config['output_folder']
        self.rev_map = config['nm_rev_map']
        self.near_miss_pair = config['target_pair_nm']
        self.alpha = config['proximity_alpha']
        self.outlier_priority = config['outlier_priority']
        self.iou_threshold = config['iou_threshold']
        self.roi_nm = config['roi_nm']
        self.near_miss_detector = ProximityDetector(self.roi_nm, self.width, self.height, config['nm_roi_threshold'])
        self.output_fps = config['output_fps']
        self.target_class_roi = config['target_class_roi']
    
    def _filterResult(self, result:list, cat:list, target_class:np.array):
        result_filtered = []
        for res, c in zip(result,cat):
            temp = []
            if len(c) > 0 :
                x = np.isin(c, target_class)
                temp = np.bitwise_and(res,x)

            result_filtered.append(temp)

        return result_filtered


    def _divideFrames(self, re, buffer, video_file):

        res = np.array([sum(r) for r in re], dtype=int)
        res[res > 0] = 1  
        i = 1
        while( i < len(res)):

            if res[i-1] == 0 and res[i] == 1:
                res[max(i-buffer,0):i] = 1

            elif res[i-1] == 1 and res[i] == 0:
                res[i:min(i+buffer, len(res))] = 1
                i += buffer

            i += 1

        idx = np.where(res)[0]
        frames = self._grabFrames(video_file, idx)
        idx_arr, frame_arr = [], []

        l = 0
        for i in range(1, len(idx)):
            if idx[i-1] != idx[i] - 1 :
                idx_arr.append(idx[l:i])
                frame_arr.append(frames[l:i])
                l = i
        idx_arr.append(idx[l:])
        frame_arr.append(frames[l:])
        
        return idx_arr, frame_arr

    def _gen_video(self, filename, imgs, frames, cat_col, codex='mp4v', fps=10,):

        out = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*codex), fps, (self.width, self.height))

        for boxes, img, cols in zip(frames, imgs, cat_col):
            #img = cv2.imread(img)
            img = cv2.drawContours(img, self.roi, -1, color=(255,0,0), thickness=2)
            #print(len(boxes), len(cols))
            for i in range(len(boxes)):
                x, y, w, h = boxes[i]

                col = (0,255,0)
                if cols[i]:
                    col = (0,0,255)
                cv2.rectangle(img, (x,y), (w,h), col , 2)

            out.write(img)


        out.release()
        cv2.destroyAllWindows()  

    def _grabFrames(self, video_file, idx):

        cap = cv2.VideoCapture(video_file)
        frames = []
        counter = 0
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read() 
            if ret == True:
                if counter in idx:
                    frames.append(frame)
                counter += 1

            else: 
                break
            
        cap.release()       
        cv2.destroyAllWindows()   

        return frames


    def writeData(self, video_path:str, bounding_boxes:list, result_roi:list, categories:list, buffer=5):
        filename = os.path.basename(video_path)
        result_roi_filtered = self._filterResult(result_roi, categories, self.target_class_roi)
        indx_arr, frame_arr = self._divideFrames(result_roi_filtered, buffer, video_path)
       
        for i in range(len(indx_arr)):
            #f = filename + '_' + str(i) + '.mp4'
            idx = indx_arr[i]
            #img_arr = self._grabFrames(video_file)
            boxes = [bounding_boxes[i] for i in idx]
            cat = [categories[i] for i in idx]
            #print(self.alpha)
            result_nm, miss_type = self.near_miss_detector.process(boxes, cat, self.near_miss_pair, self.alpha, self.iou_threshold, self.outlier_priority)
            miss_type = 0
            
            # if len(result_nm) > 0 :
            #     miss_type = np.max(max(result_nm, key=max))

            base_path = os.path.join(self.base_path, self.rev_map[miss_type])
            f = os.path.join(base_path , filename[:-4] + '_' + str(i) + '.mp4')
            img_arr = frame_arr[i]
            #box_arr = [frames[i] for i in idx]

            cat_col = [result_roi[i] for i in idx]
       
            if miss_type > 0 :
                #f = os.path.join(self.near_miss_path , filename[:-4] + '_' + str(i) + '.mp4')
                #box_arr = boxes_nm
                cat_col = result_nm
            
            #print(img_arr)
            #print(self.rev_map[miss_type])
            self._gen_video(f, img_arr, boxes, cat_col, fps=self.output_fps)

            ### create output for csv
            # for i in idx:
            #     temp_arr = [self.rev_map[mt] for mt in result_nm[i]]
            #     nm_result_arr[i] = temp_arr
            
            # videoClip = VideoFileClip(f)
            # videoClip.write_gif(f[:-4]+ ".gif")

        
       
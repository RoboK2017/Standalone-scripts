from modules.configReader import ConfigReader
from modules.videoReader import VideoReader
from modules.roiIntersectionDetector import ROIDetector
from modules.dataWriter import DataWriter
import argparse
import os
import time



parser = argparse.ArgumentParser()

parser.add_argument('--video', type=str,
    help='Input video path', required=True)

parser.add_argument('--config', type=str,
    default='config/test.json', help='Config path')    



class ROIService():

    def __init__(self, config:str):

        self.config = ConfigReader(config)
        self.reader = VideoReader(self.config)


    def process(self, video, fps=10):

        print("Detecting objects in video...")
        #st_time = time.time()
        bounding_boxes, bounding_boxes_nm, categories = self.reader.process_video(video)
        #et_time = time.time()
        #print(et_time - st_time)
        print("detecting ROI interaction...")
        #print(len(frames))
        detector = ROIDetector(self.config)
        
        result = detector.process([bounding_boxes])
        print('writing results into output folder...')
        writer = DataWriter(self.config)
        video_name = os.path.basename(video)
       # output_path = os.path.join(self.config['output_folder'] , video_name[:-4])
        buffer = self.config['buffer'] * fps
        writer.writeData(video_name, video, bounding_boxes, result, bounding_boxes_nm, categories, buffer)

        

        
if __name__ == "__main__":
    # need to add argparse for taking input
    #ser = ROIService('config/test.json')
    #ser.process('test.avi')
    args = parser.parse_args()    
    #print(args.video, args.config)
    ser = ROIService(args.config)
    for f in os.listdir(args.video):
        if '.avi' in f or '.mp4' in f:
            ser.process(os.path.join(args.video, f))
        
            
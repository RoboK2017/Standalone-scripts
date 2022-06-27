# ROI Interaction Detector + near miss proximity detection

## Introduction 

The following service takes a video input and detects object in it and detects if at any point the object
intersects with the defined region of intrest (ROI) and finally outputs the instances in the video where the intersection 
with the ROI happened. additionally a new proximity sub module is added to flag the roi output as a potential near miss.

## Dependencies 

The following service requires 
<ul>
<li>Python >= 3.5</li>
<li>numpy</li>
<li>opencv > 4.0 with DNN support</li>
<li>scipy</li>
</ul>


## Usage

`python main.py --video path to input video folder --config path to config file`

## Config

The service requires a valid json config file to process the video. The keys in the config file are 

<ul>
<li>width: The width if the video frame</li>
<li>height: The height of the video frame</li>
<li>trigger: Defines the type of ROI detector to use --options [point, area, iou]</li>
<li>weight: path to darknet weight file </li>
<li>config: path to darknet model config file </li>
<li>class_names: path to darknet model class names</li>
<li>roi: defines the ROI for interaction detector, can accomodate multiple polygons in the standard format [poly1, poly2, .... ,polyn] where
 a polygon is defined by [[x1,y1], [x2,y2], [x3,y3], ..... , [xn,yn]] </li>
<li>buffer: defines how many seconds to keep before and after the intersection detection in the output clip</li> 
<li>output_folder: path to output folder</li>
<li>target_class_roi: selects the class which the object detector will filter</li>
<li>target_pair_nm: defines the near miss pair for the detection followed by the priority value</li>
<li>proximity_alpha: define an alpha variable which adjusts the distance between the target pairs to classify as a near miss, 
 take values between 0-1</li>
<li>
</ul>

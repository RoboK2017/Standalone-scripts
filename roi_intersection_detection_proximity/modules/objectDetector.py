import numpy as np
import cv2
import time

class YoloDetector:

    def __init__(self, weights_file, config_file, class_names_filepath, threshold:float):

        #self._yolo_weights_file = "bus_camera/models/yolov4-tiny/yolov4-tiny.weights"
        #self._yolo_config = "bus_camera/models/yolov4-tiny/yolov4-tiny.cfg"
        self._yolo_weights_file = weights_file
        self._yolo_config = config_file
        self.threshold = threshold
        self.ver = cv2.__version__
        self._initialise_object_detector(self._yolo_config, self._yolo_weights_file)

        self._class_names = open(class_names_filepath).read().strip().split('\n')

    def detect_boxes(self, image):
        """
        For a given image, return the number of passengers
        :param image:
        :return:
        """

        # construct a blob from the image
        
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (608, 608), swapRB=True, crop=False)

        # feedforward
        #print(blob.shape)
        self._net.setInput(blob)
        
        #st_tm = time.time()
        outputs = self._net.forward(self._ln)
        # analyse the result
        boxes = []
        confidences = []
        classIDs = []
        h, w = image.shape[:2]

        for output in outputs:
            scores = output[:,5:]
            cids = np.argmax(scores, axis=1)
            confidence = scores[range(len(cids)), cids]
            pos = np.where(confidence > self.threshold)[0]
            if len(pos) > 0:

                box = (output[pos, : 4] * np.array([w, h, w, h]))
                box[:,0] = box[:,0] - (box[:,2]/2)
                box[:,1] = box[:,1] - (box[:,3]/2)
                box = box.astype(int)

                confidence = confidence[pos].astype(float)
                cids = cids[pos]
                
                boxes.extend(box)
                confidences.extend(confidence)
                classIDs.extend(cids)

        #boxes = [list(b) for b in boxes]
        # if (len(boxes) > 1):        
        #     classIDs = np.concatenate(classIDs)
        #     confidences = np.concatenate(confidences)
        #ed_tm = time.time()

        #print(ed_tm - st_tm)
       # boxes = list(boxes)
       #
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # get the bounding boxes from indices
        bounding_boxes = []
        predicted_labels = []
        for idx1 in indices:
            if self.ver >= '4.5.5':
                idx2 = idx1 #[0]
            else :
                idx2 = idx1[0]    
            label = self._class_names[classIDs[idx2]]
            predicted_labels.append(label)
            box = boxes[idx2]
            bounding_boxes.append(box)

        return bounding_boxes, predicted_labels

    def _initialise_object_detector(self, yolo_config, yolo_weights):
        self._net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
        #self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # determine the output layer
        self._ln = self._net.getLayerNames()
        if self.ver >= '4.5.5':
            self._ln = [self._ln[i - 1] for i in self._net.getUnconnectedOutLayers()]
        else : 
            self._ln = [self._ln[i[0] - 1] for i in self._net.getUnconnectedOutLayers()]   
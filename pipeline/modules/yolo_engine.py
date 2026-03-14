import cv2 as cv
import numpy as np
from onnxruntime import InferenceSession

class YoloDetector:
    def __init__(self, model_path, input_shape=(640, 640), conf_thres=0.5, iou_thres=0.4):
        # Initialize the ONNX Runtime session with GPU support if available
        self.session = InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_shape = input_shape
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def preprocess(self, img):
        """
        Prepares the image for the model (Letterbox + Blob).
        """
        h, w = img.shape[:2]
        
        # scaling factor (for letterbox)
        scale = min(self.output_shape[0]/h, self.output_shape[1]/w)
        
        # new constraints (before Padding)
        new_unpad = (int(round(w * scale)), int(round(h * scale)))
        
        # Resize
        resized = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
        
        # Padding calculations (Grey areas)
        dw = (self.output_shape[1] - new_unpad[0]) / 2
        dh = (self.output_shape[0] - new_unpad[1]) / 2
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # Addition the grey frame (114, 114, 114)
        img_padded = cv.copyMakeBorder(resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=(114, 114, 114))
        
        blob = cv.dnn.blobFromImage(img_padded, 1/255.0, self.output_shape, swapRB=True, crop=False)
        blob = blob.astype(np.float16)
        
        return blob, (dw, dh), scale

    '''
    Make detection on the input frame and return the bounding box of the detected object.
    - Input: frame (BGR format)
    - Returns: [x, y, w, h] or None if no valid detection is found.
    '''
    def detect(self, frame):
        detections = self.detect_all(frame)
        if detections:
            return detections[0]['bbox']
        return None

    '''
    Make detection on the input frame and return all valid bounding boxes with confidence scores.
    - Input: frame (BGR format)
    - Returns: list of dicts [{'bbox': [x, y, w, h], 'confidence': float}, ...] or empty list.
    '''
    def detect_all(self, frame):
        original_h, original_w = frame.shape[:2]
        
        # 1. Preprocess
        blob, (dw, dh), scale = self.preprocess(frame)
        
        # 2. Inference
        outputs = self.session.run(None, {self.input_name: blob})
        
        # 3. ACCELERATED POST-PROCESS WITH NUMPY
        data = outputs[0].squeeze(axis=0) 
        
        scores = data[4, :]
        
        valid_indices = np.where(scores > self.conf_thres)[0]
        
        if len(valid_indices) == 0:
            return []

        valid_data = data[:, valid_indices]
        valid_scores = scores[valid_indices]
        center_xs = valid_data[0, :]
        center_ys = valid_data[1, :]
        ws = valid_data[2, :]
        hs = valid_data[3, :]
        
        lefts = ((center_xs - ws/2) - dw) / scale
        tops = ((center_ys - hs/2) - dh) / scale
        widths = ws / scale
        heights = hs / scale
        
        boxes = np.stack((lefts, tops, widths, heights), axis=1).astype(int).tolist()
        confidences = valid_scores.tolist()
        
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.conf_thres, self.iou_thres)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                
                is_big = w > (original_w * 0.05) or h > (original_h * 0.05)
                in_center_x = (original_w * 0.15) < x and (x + w) < (original_w * 0.85)
                in_center_y = (original_h * 0.10) < y and (y + h) < (original_h * 0.90)
                
                if is_big and in_center_x and in_center_y:
                    results.append({'bbox': [x, y, w, h], 'confidence': confidences[i]})
        return results
import sys
import cv2 as cv
import numpy as np
from config import AVTRACK_ENGINE_PATH,AVTRACK_ROOT


# =============================================================================
# Load of Preferred Tracker (AVTrack, ORTrack, MixFormerV2, VitTracker vb.)
# =============================================================================

# Change the dependency paths according to the current tracker you want to use.

#AVTrack
import params.tracker.av_track_params as avtrack_params
sys.path.append(AVTRACK_ROOT)
from modules.custom.avtrack_adapter import AVTrackTracker

#ORTrack
# import params.tracker.or_track_params as ortrack_params
# sys.path.append(ORTRACK_ROOT)
# from lib.test.tracker.ortrack import ORTrack

#MixFormerV2
# import params.tracker.mixformer_params as mixformer_params
# sys.path.append(MIXFORMER_ROOT)
# from lib.test.tracker.mixformer2_vit import MixFormer  

#VitTracker
# from .builtin.vittracker import VitTracker 

from params.tracker_types import TRACKERS

class TrackerAdapter:

    def __init__(self, tracker_model = TRACKERS.AVTrack):
        """
        Loads the preferred tracker model.
        - tracker_model: Enum value from TRACKERS (e.g., TRACKERS.AVTrack, TRACKERS.ORTrack, etc.)
        """
        #To use the other Trackers, change the comment lines
        if (tracker_model == TRACKERS.AVTrack):
            self.tracker = AVTrackTracker(
                config_name='deit_tiny_patch16_224',
                # onnx_path='/home/furkan/Desktop/CS/altek/pipeline/models/avtrack.onnx',  # ORT modu
                engine_path=AVTRACK_ENGINE_PATH,  # ORT modu
                device='cuda'
            )
            # pass
        elif (tracker_model == TRACKERS.ORTrack):
            pass
            # self.tracker = ORTrack(ortrack_params.params, dataset_name=ortrack_params.dataset_name)
        elif (tracker_model == TRACKERS.MixFormerV2):
            pass
            # self.tracker = MixFormer(mixformer_params.params,dataset_name="MixFormer-Live")
        else:
            pass
            # self.tracker = VitTracker()  # VitTracker sınıfını kullan        
        self.is_initialized = False
        self.tracker_model = tracker_model
        print(f"Tracker is ready, model has been loaded: {tracker_model.name}")

    def initialize(self, frame, bbox):
        """
        Starts tracking with the given bounding box coming from YOLO.
        - bbox format: [x, y, w, h]
        - frame: The current video frame (numpy array)
        """
        if self.tracker is None:
            print("WARNING: Tracker is not initialized. Cannot start tracking.")
            return
        
        bbox = [int(x) for x in bbox]
        # Change the bbox parameters according to the tracker models
        if (self.tracker_model != TRACKERS.VitTracker):
            self.tracker.initialize(frame, {'init_bbox': bbox})
        else:
            self.tracker.initialize(frame, bbox)
        self.is_initialized = True

    def update(self, frame):
        """
        Updates the tracker with the current frame and returns the new bounding box.
        - frame: The current video frame (numpy array)
        Return: (success: bool, bbox: list) -> [x, y, w, h]
        """
        if not self.is_initialized or self.tracker is None:
            return False, None
        outputs = self.tracker.track(frame)
        
        if (outputs is None):
            return False, None
        if 'target_bbox' in outputs:
            bbox = outputs['target_bbox']
            score = outputs.get('best_score', 1.0) # default value is 1.0
            
            # Thresholding the score to determine if frame successfuly tracked or not.
            if score > 0.2: # Threshold is %40, you can change it.
                final_bbox = [int(v) for v in bbox]
                return True, final_bbox,score
            else:
                return False, None,score
        return False, None, 0

    def clear_initialization(self):
        """
        Clears the tracker's initialization state, allowing it to be re-initialized with a new bounding box.
        """
        self.is_initialized = False
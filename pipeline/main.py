import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2 as cv # type: ignore
import sys
import time

# Import the modules
from modules.yolo_engine import YoloDetector
from modules.tracker_adapter import TrackerAdapter
from main.pipeline import DronePipeline
from params.tracker_types import TRACKERS
from config import YOLO_MODEL_PATH, DEFAULT_VIDEO

def main():
    # --- 1. Settings ---
    # Define the model and video paths in your computer.
    MODEL_PATH = YOLO_MODEL_PATH
    VIDEO_PATH = DEFAULT_VIDEO
    tracking_method = TRACKERS.AVTrack

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file does not exist. -> {MODEL_PATH}")
        return
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video file does not exist. -> {VIDEO_PATH}")
        return

    # --- 2. SYSTEM INITIALIZATION ---
    print("System is initializing...")
    
    print("1. YOLO detector is being prepared...")
    yolo_engine = YoloDetector(model_path=MODEL_PATH, conf_thres=0.5)
    
    print(f"2. {tracking_method} Tracker is being prepared...")
    tracker_engine = TrackerAdapter(tracker_model=tracking_method)  # MixFormerV2 kullanılıyor
    
    print("3. Pipeline is initalizing...")
    pipeline = DronePipeline(yolo_engine, tracker_engine)

    # --- 3. MAIN LOOP ---
    cap = cv.VideoCapture(VIDEO_PATH)
    
    
    print("System is ready! Starting...")
    prev_time = 0
    while True:
        cv.namedWindow("TEKNOFEST 2026 - TARGET SYSTEM", cv.WINDOW_NORMAL)
        ret, frame = cap.read()
        if not ret: break
        fh,fw = frame.shape[0], frame.shape[1]
        processed_frame = pipeline.run_step(frame)

        curr_time = time.time()
        time_diff = curr_time - prev_time
        
        # To prevent zero division error
        if time_diff > 0: fps = 1 / time_diff
        else: fps = 0
            
        prev_time = curr_time
        fps_text = f"FPS: {int(fps)}"
        
        cv.putText(processed_frame, fps_text, (1050, 680), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        # ------------------------------------------------
        x1, y1 = int(fw * 0.25), int(fh * 0.1)
        x2, y2 = int(fw * 0.75), int(fh * 0.9)
        cv.rectangle(processed_frame, (x1, y1), (x2, y2), (255,255,0), 2)
        cv.putText(processed_frame, "TARGET AREA (AV)",(x1 + 4, y1 + 16), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0) , 1)
        cv.imshow("TEKNOFEST 2026 - TARGET SYSTEM", processed_frame)        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    

    print("\n\nTOTAL LOCK COUNT: ", pipeline.getTotalLockCount());
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
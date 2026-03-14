import time
import cv2 as cv
from enum import Enum
from pipeline.modules.target_selector import select_target, SelectionStrategy

AV_X_MIN     = 0.25   # Target Engagement Zone horizontal left boundary
AV_X_MAX     = 0.75   # Target Engagement Zone horizontal right boundary
AV_Y_MIN     = 0.10   # Target Engagement Zone vertical top boundary
AV_Y_MAX     = 0.90   # Target Engagement Zone vertical bottom boundary
MIN_COVERAGE = 0.06   # 5% specification limit; using 6% as a safety margin

# Colors (BGR)
COLOR_AV      = (0, 220, 255)   # Target Engagement Zone
COLOR_LOCK    = (0, 0, 255)     # Lock rectangle — red #FF0000
COLOR_LOCKING = (0, 165, 255)   # Lock in progress — orange
COLOR_SEARCH  = (200, 200, 200) # Search mode — gray

# Enum for state management (improves readability)
class SystemState(Enum):
    SEARCHING = 1   # Scan the environment with YOLO
    TRACKING = 2    # Target acquired, continue tracking

class DronePipeline:
    def __init__(self, yolo_engine, tracker_engine, selection_strategy=SelectionStrategy.CLOSEST_TO_CENTER):
        self.yolo = yolo_engine
        self.tracker = tracker_engine
        self.selection_strategy = selection_strategy

        # Initial state
        self.state = SystemState.SEARCHING
        
        # Timers
        self.lock_start_time = None  # Lock duration timer
        
        # Target data
        self.bbox = None # [x, y, w, h]
        self.last_known_bbox = None  # Last tracked bbox for re-acquisition


    def run_step(self, frame):
        output_frame = frame
        fh, fw = frame.shape[:2]
        if self.lock_start_time is not None:
            elapsed = time.time() - self.lock_start_time
            if elapsed >= 4.0:  # If 4 seconds have elapsed
                print("LOCK ENGAGEMENT COMPLETE. TARGET NEUTRALIZED.")
                self._draw_status(frame, self.bbox, COLOR_AV, "STRIKE!")
                self.lock_start_time = None  # Lock sequence completed, reset timer
        
        # --- STATE 1: SEARCH MODE ---
        if self.state == SystemState.SEARCHING:
            self.execute_searching_mode(output_frame, frame,fh,fw)
        # --- STATE 2: TRACKING MODE ---
        elif self.state == SystemState.TRACKING:
            self.execute_tracking_mode(output_frame, frame,fh,fw)
        return output_frame

    def execute_searching_mode(self, output,frame,fh,fw):
        detections = self.yolo.detect_all(frame)
        if detections:
            selected_bbox = select_target(
                detections, fw, fh,
                strategy=self.selection_strategy,
                last_bbox=self.last_known_bbox
            )
            if selected_bbox is not None:
                x, y, w, h = selected_bbox
                cx = x + w / 2.0
                cy = y + h / 2.0
                if self.check_center_of_box(cx, cy, fw, fh) and self.check_coverage(w, h, fw, fh):
                    print("TARGET ACQUIRED. INITIATING TRACKING...")
                    self.tracker.initialize(frame, selected_bbox)
                    self.state = SystemState.TRACKING
                    self.bbox = selected_bbox
                    self.last_known_bbox = selected_bbox
                    self.lock_start_time = time.time()
                    self._draw_status(output, self.bbox, (0, 0, 255), "DETECTED")
            # Draw other detections as non-selected
            for det in detections:
                if selected_bbox is None or det['bbox'] != selected_bbox:
                    bx, by, bw, bh = [int(v) for v in det['bbox']]
                    cv.rectangle(output, (bx, by), (bx + bw, by + bh), COLOR_SEARCH, 1)
        else:
            cv.putText(output, "SEARCHING (YOLO)...", (50, 50), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)

    def execute_tracking_mode(self,output,frame,fh,fw):
        success, new_bbox = self.tracker.update(frame)
        if not success:
            print("[PIPELINE] TRACKER LOST TARGET -> SEARCHING")
            self.last_known_bbox = self.bbox
            self.lock_start_time = None
            self.state = SystemState.SEARCHING
            self.bbox  = None
            cv.putText(output, "STATE: TRACKING (LOST)",
                        (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
            return output

        self.bbox = new_bbox
        x, y, w, h = [int(v) for v in self.bbox]
        cx = x + w / 2.0
        cy = y + h / 2.0

        check_center = self.check_center_of_box(cx, cy, fw, fh)
        coverage_ok = self.check_coverage(w, h, fw, fh)
        lock_valid  = check_center and coverage_ok

        # Lock timing management
        if lock_valid:
            if self.lock_start_time is None:
                self.lock_start_time = time.time()
                print(f"[PIPELINE] LOCK ENGAGEMENT STARTED: {self.lock_start_time} MS")
            elapsed = time.time() - self.lock_start_time
            if elapsed >= 4.0:
                print(">>> LOCK ENGAGEMENT COMPLETE. TARGET NEUTRALIZED. <<<")
                self._draw_status(frame, self.bbox, COLOR_AV, "STRIKE!")

                # Optional: return to search mode after mission completion
                # self._reset_to_search() 
                self.lock_start_time = None 
        else:
            # Conditions no longer met — reset timer
            if self.lock_start_time is not None:
                print("[PIPELINE] LOCK CONDITIONS NO LONGER MET -> LOCK TIMER RESET")
            self.lock_start_time = None
        # Rendering
        is_locking = self.lock_start_time is not None
        color = COLOR_LOCKING if is_locking else COLOR_LOCK
        cv.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv.circle(output, (int(cx), int(cy)), 4, color, -1)
        cv.putText(output, f"H:{w/fw*100:.1f}%  V:{h/fh*100:.1f}%",
                    (x, y - 8), cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv.putText(output, "STATE: TRACKING",
                    (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        return output
    def check_coverage(self, w, h, fw, fh):
        return (w / fw >= MIN_COVERAGE) or (h / fh >= MIN_COVERAGE)

    def check_center_of_box(self, cx, cy, fw, fh) -> bool:
        return (
            AV_X_MIN <= cx / fw <= AV_X_MAX and
            AV_Y_MIN <= cy / fh <= AV_Y_MAX
        )
    def _draw_status(self, img, bbox, color, text):
        """Helper function: Draws the bounding box and status text."""
        x, y, w, h = [int(v) for v in bbox]
        # Bounding box
        cv.rectangle(img, (x, y), (x + w, y + h), color, 3)
        # Text background (for readability)
        (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv.rectangle(img, (x, y - 25), (x + tw, y), color, -1)
        # Text
        cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _draw_av(self, img, fw, fh):
        x1, y1 = int(fw * AV_X_MIN), int(fh * AV_Y_MIN)
        x2, y2 = int(fw * AV_X_MAX), int(fh * AV_Y_MAX)
        cv.rectangle(img, (x1, y1), (x2, y2), COLOR_AV, 2)
        cv.putText(img, "TARGET ENGAGEMENT ZONE (AV)",
                (x1 + 4, y1 + 16), cv.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_AV, 1)


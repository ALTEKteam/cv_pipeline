"""
Central configuration for the ALTEK Computer Vision Pipeline.

All paths are relative to the project root directory.
Copy this file as `config_local.py` and adjust paths if your setup differs.
"""

import os

# Project root directory (auto-detected)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMPLEMENTATIONS_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "tracking_implementations")

# ============================================================
# Model paths
# ============================================================
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOCK_VIDEO_DIR = os.path.join(PROJECT_ROOT, "locks")

YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "best_new_fp16.onnx")
AVTRACK_CHECKPOINT = os.path.join(MODELS_DIR, "AVTrack_model.pth")
AVTRACK_ONNX_PATH = os.path.join(MODELS_DIR, "avtrack.onnx")
AVTRACK_ENGINE_PATH = os.path.join(MODELS_DIR, "avtrack_3060_fp16.engine")
ORTRACK_CHECKPOINT = os.path.join(MODELS_DIR, "ORTrack_model.pth")
MIXFORMER_CHECKPOINT = os.path.join(MODELS_DIR, "mixformerv2_small.pth")
VITTRACKER_ONNX_PATH = os.path.join(MODELS_DIR, "vitTracker.onnx")

# ============================================================
# Video paths
# ============================================================
VIDEOS_DIR = os.path.join(PROJECT_ROOT, "videos")
DEFAULT_VIDEO = os.path.join(VIDEOS_DIR, "hoosier_fpv_700_feet.mp4")

# ============================================================
# Tracker implementation paths
# ============================================================
TRACKERS_DIR = os.path.join(PROJECT_ROOT, "tracking_implementations")

AVTRACK_ROOT = os.path.join(IMPLEMENTATIONS_DIR, "AVTrack")
ORTRACK_ROOT = os.path.join(IMPLEMENTATIONS_DIR, "ORTrack")
MIXFORMER_ROOT = os.path.join(IMPLEMENTATIONS_DIR, "MixFormerV2")
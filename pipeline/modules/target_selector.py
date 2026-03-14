import math
from enum import Enum


class SelectionStrategy(Enum):
    CLOSEST_TO_CENTER = 1
    HIGHEST_CONFIDENCE = 2
    WEIGHTED = 3


def select_target(detections, frame_w, frame_h, strategy=SelectionStrategy.CLOSEST_TO_CENTER, last_bbox=None):
    """
    Selects the best tracking target from multiple detections.

    Args:
        detections: list of dicts [{'bbox': [x, y, w, h], 'confidence': float}, ...]
        frame_w: frame width in pixels
        frame_h: frame height in pixels
        strategy: SelectionStrategy enum value
        last_bbox: previous tracked bbox [x, y, w, h] or None (used for re-acquisition)

    Returns:
        [x, y, w, h] of the selected target, or None if detections is empty.
    """
    if not detections:
        return None

    if len(detections) == 1:
        return detections[0]['bbox']

    if last_bbox is not None:
        return _select_nearest_to_last(detections, last_bbox)

    if strategy == SelectionStrategy.CLOSEST_TO_CENTER:
        return _select_closest_to_center(detections, frame_w, frame_h)
    elif strategy == SelectionStrategy.HIGHEST_CONFIDENCE:
        return _select_highest_confidence(detections)
    elif strategy == SelectionStrategy.WEIGHTED:
        return _select_weighted(detections, frame_w, frame_h)

    return detections[0]['bbox']


def _bbox_center(bbox):
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def _select_closest_to_center(detections, frame_w, frame_h):
    """Select the detection whose center is closest to the frame center."""
    cx_frame = frame_w / 2.0
    cy_frame = frame_h / 2.0

    best = None
    best_dist = float('inf')
    for det in detections:
        cx, cy = _bbox_center(det['bbox'])
        dx = (cx - cx_frame) / frame_w
        dy = (cy - cy_frame) / frame_h
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < best_dist:
            best_dist = dist
            best = det['bbox']
    return best


def _select_highest_confidence(detections):
    """Select the detection with the highest confidence score."""
    best = max(detections, key=lambda d: d['confidence'])
    return best['bbox']


def _select_weighted(detections, frame_w, frame_h, center_weight=0.6, confidence_weight=0.4):
    """Select based on a weighted combination of proximity-to-center and confidence."""
    cx_frame = frame_w / 2.0
    cy_frame = frame_h / 2.0

    max_dist = math.sqrt(0.25 + 0.25)  # max normalized distance from center

    best = None
    best_score = -1.0
    for det in detections:
        cx, cy = _bbox_center(det['bbox'])
        dx = (cx - cx_frame) / frame_w
        dy = (cy - cy_frame) / frame_h
        dist = math.sqrt(dx * dx + dy * dy)
        proximity_score = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0
        score = center_weight * proximity_score + confidence_weight * det['confidence']
        if score > best_score:
            best_score = score
            best = det['bbox']
    return best


def _select_nearest_to_last(detections, last_bbox):
    """Select the detection closest to the last known target position (re-acquisition)."""
    last_cx, last_cy = _bbox_center(last_bbox)

    best = None
    best_dist = float('inf')
    for det in detections:
        cx, cy = _bbox_center(det['bbox'])
        dist = math.sqrt((cx - last_cx) ** 2 + (cy - last_cy) ** 2)
        if dist < best_dist:
            best_dist = dist
            best = det['bbox']
    return best

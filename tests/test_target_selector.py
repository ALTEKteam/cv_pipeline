import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.modules.target_selector import (
    select_target,
    SelectionStrategy,
    _bbox_center,
    _select_closest_to_center,
    _select_highest_confidence,
    _select_weighted,
    _select_nearest_to_last,
)


class TestBboxCenter(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(_bbox_center([100, 200, 50, 80]), (125.0, 240.0))

    def test_zero_size(self):
        self.assertEqual(_bbox_center([10, 20, 0, 0]), (10.0, 20.0))


class TestSelectTargetEmpty(unittest.TestCase):
    def test_empty_detections(self):
        self.assertIsNone(select_target([], 1920, 1080))

    def test_single_detection(self):
        det = [{'bbox': [100, 200, 50, 50], 'confidence': 0.9}]
        self.assertEqual(select_target(det, 1920, 1080), [100, 200, 50, 50])


class TestClosestToCenter(unittest.TestCase):
    def test_selects_center_detection(self):
        detections = [
            {'bbox': [10, 10, 50, 50], 'confidence': 0.95},     # top-left
            {'bbox': [935, 515, 50, 50], 'confidence': 0.70},   # near center of 1920x1080
            {'bbox': [1800, 1000, 50, 50], 'confidence': 0.99}, # bottom-right
        ]
        result = select_target(detections, 1920, 1080, strategy=SelectionStrategy.CLOSEST_TO_CENTER)
        self.assertEqual(result, [935, 515, 50, 50])

    def test_direct_function(self):
        detections = [
            {'bbox': [0, 0, 100, 100], 'confidence': 0.8},
            {'bbox': [450, 250, 100, 100], 'confidence': 0.6},
        ]
        result = _select_closest_to_center(detections, 1000, 600)
        self.assertEqual(result, [450, 250, 100, 100])


class TestHighestConfidence(unittest.TestCase):
    def test_selects_highest(self):
        detections = [
            {'bbox': [100, 100, 50, 50], 'confidence': 0.60},
            {'bbox': [200, 200, 50, 50], 'confidence': 0.95},
            {'bbox': [300, 300, 50, 50], 'confidence': 0.80},
        ]
        result = select_target(detections, 1920, 1080, strategy=SelectionStrategy.HIGHEST_CONFIDENCE)
        self.assertEqual(result, [200, 200, 50, 50])

    def test_direct_function(self):
        detections = [
            {'bbox': [10, 10, 10, 10], 'confidence': 0.3},
            {'bbox': [20, 20, 10, 10], 'confidence': 0.9},
        ]
        result = _select_highest_confidence(detections)
        self.assertEqual(result, [20, 20, 10, 10])


class TestWeightedStrategy(unittest.TestCase):
    def test_weighted_prefers_center_with_decent_conf(self):
        detections = [
            {'bbox': [935, 515, 50, 50], 'confidence': 0.60},   # center, lower conf
            {'bbox': [10, 10, 50, 50], 'confidence': 0.99},     # far corner, high conf
        ]
        result = select_target(detections, 1920, 1080, strategy=SelectionStrategy.WEIGHTED)
        self.assertEqual(result, [935, 515, 50, 50])

    def test_direct_function(self):
        detections = [
            {'bbox': [500, 300, 100, 100], 'confidence': 0.5},
            {'bbox': [0, 0, 100, 100], 'confidence': 0.99},
        ]
        result = _select_weighted(detections, 1100, 700)
        self.assertEqual(result, [500, 300, 100, 100])


class TestNearestToLast(unittest.TestCase):
    def test_reacquire_nearest(self):
        detections = [
            {'bbox': [100, 100, 50, 50], 'confidence': 0.90},
            {'bbox': [800, 800, 50, 50], 'confidence': 0.95},
        ]
        last_bbox = [110, 110, 50, 50]
        result = select_target(detections, 1920, 1080, last_bbox=last_bbox)
        self.assertEqual(result, [100, 100, 50, 50])

    def test_reacquire_overrides_strategy(self):
        detections = [
            {'bbox': [100, 100, 50, 50], 'confidence': 0.50},
            {'bbox': [960, 540, 50, 50], 'confidence': 0.99},
        ]
        last_bbox = [90, 90, 50, 50]
        # Even with HIGHEST_CONFIDENCE strategy, last_bbox should override
        result = select_target(
            detections, 1920, 1080,
            strategy=SelectionStrategy.HIGHEST_CONFIDENCE,
            last_bbox=last_bbox,
        )
        self.assertEqual(result, [100, 100, 50, 50])

    def test_direct_function(self):
        detections = [
            {'bbox': [500, 500, 40, 40], 'confidence': 0.7},
            {'bbox': [100, 100, 40, 40], 'confidence': 0.9},
        ]
        result = _select_nearest_to_last(detections, [510, 510, 40, 40])
        self.assertEqual(result, [500, 500, 40, 40])


if __name__ == '__main__':
    unittest.main()

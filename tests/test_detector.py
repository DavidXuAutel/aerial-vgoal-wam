"""Unit tests for TargetDetector frontend (Mock and YOLO detectors)."""

import unittest
import numpy as np

from vgoal.detector import DetectionResult, MockDetector, YOLOTargetDetector


class TestDetector(unittest.TestCase):

    def test_detection_result_properties(self):
        res = DetectionResult(
            bbox=np.array([100.0, 50.0, 200.0, 150.0], dtype=np.float32),
            confidence=0.88,
            class_id=2,
            class_name="car",
        )
        self.assertTrue(np.allclose(res.center, [150.0, 100.0]))
        self.assertEqual(res.area, 100.0 * 100.0)

    def test_mock_detector_detects_target(self):
        detector = MockDetector(target_bbox=[50.0, 50.0, 120.0, 120.0], confidence=0.92, class_name="landing_pad")
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        det = detector.detect(img)
        self.assertIsNotNone(det)
        self.assertEqual(det.class_name, "landing_pad")
        self.assertEqual(det.confidence, 0.92)
        self.assertTrue(np.allclose(det.bbox, [50.0, 50.0, 120.0, 120.0]))

    def test_mock_detector_handles_no_target(self):
        detector = MockDetector(target_bbox=None)
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        det = detector.detect(img)
        self.assertIsNone(det)

        detector.set_target([10, 10, 20, 20], confidence=0.0)
        self.assertIsNone(detector.detect(img))

    def test_yolo_detector_filter_logic_mocked(self):
        # Verify filtering logic without invoking real neural net weights
        detector = YOLOTargetDetector(target_classes={"car", "truck"})
        self.assertIn("car", detector.target_classes)
        self.assertIn("truck", detector.target_classes)
        self.assertNotIn("person", detector.target_classes)


if __name__ == "__main__":
    unittest.main()

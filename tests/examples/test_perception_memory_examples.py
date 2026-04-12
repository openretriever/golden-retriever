from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.advanced.memory_examples.common import BeliefTracker, DetectionDropout, SelectBeliefTarget
from examples.advanced.perception_examples.common import ColorDetector, ColorSegmenter, PointToLabel, SyntheticColorCamera


class PerceptionMemoryExampleTests(unittest.TestCase):
    def _next_frame(self, camera: SyntheticColorCamera):
        return camera.step(None)

    def test_detection_and_segmentation_share_the_same_scene(self) -> None:
        camera = SyntheticColorCamera(dt=0.1)
        camera.init()
        frame = self._next_frame(camera)

        detector = ColorDetector()
        segmenter = ColorSegmenter()

        detections = detector.step(frame)
        segmentation = segmenter.step(frame)

        labels_from_detections = {det.label for det in detections.detections}
        labels_from_segmentation = set(segmentation.labels)

        self.assertEqual(frame.frame_id, 1)
        self.assertEqual(labels_from_detections, {"red", "blue"})
        self.assertEqual(labels_from_segmentation, {"red", "blue"})
        self.assertGreater(segmentation.pixel_counts["red"], segmentation.pixel_counts["blue"])

    def test_pointing_selects_requested_label(self) -> None:
        camera = SyntheticColorCamera(dt=0.1)
        camera.init()
        detector = ColorDetector()
        pointer = PointToLabel(target_label="blue")

        point = pointer.step(detector.step(self._next_frame(camera)))

        self.assertEqual(point.label, "blue")
        self.assertIsNotNone(point.x_norm)
        self.assertIsNotNone(point.y_norm)
        self.assertGreaterEqual(point.x_norm, 0.0)
        self.assertLessEqual(point.x_norm, 1.0)
        self.assertGreaterEqual(point.y_norm, 0.0)
        self.assertLessEqual(point.y_norm, 1.0)

    def test_belief_tracker_holds_target_through_dropout(self) -> None:
        camera = SyntheticColorCamera(dt=0.1)
        camera.init()
        detector = ColorDetector()
        dropout = DetectionDropout(target_label="red", every_n=2)
        belief = BeliefTracker(hold_steps=1)
        belief.init()

        batch1 = detector.step(self._next_frame(camera))
        scene1 = belief.step(dropout.step(batch1))
        self.assertEqual({obj.label for obj in scene1.objects}, {"red", "blue"})

        batch2 = detector.step(self._next_frame(camera))
        scene2 = belief.step(dropout.step(batch2))
        red = next(obj for obj in scene2.objects if obj.label == "red")
        self.assertEqual(red.missing_steps, 1)
        self.assertGreater(red.confidence, 0.0)

    def test_pointing_memory_uses_belief_state(self) -> None:
        camera = SyntheticColorCamera(dt=0.1)
        camera.init()
        detector = ColorDetector()
        dropout = DetectionDropout(target_label="red", every_n=2)
        belief = BeliefTracker(hold_steps=1)
        belief.init()
        selector = SelectBeliefTarget(target_label="red")

        scene1 = belief.step(dropout.step(detector.step(self._next_frame(camera))))
        point1 = selector.step(scene1)
        scene2 = belief.step(dropout.step(detector.step(self._next_frame(camera))))
        point2 = selector.step(scene2)

        self.assertEqual(point1.label, "red")
        self.assertEqual(point2.label, "red")
        self.assertIsNotNone(point2.confidence)
        self.assertGreater(point2.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()

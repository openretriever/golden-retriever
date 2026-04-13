from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retriever.types.language import Caption, GroundedPhrase, PlanText, ReferringExpression
from retriever.types.perception import BBox2D, Detection2D, DetectionBatch

from examples.advanced.language_examples.common import CaptionPlanner, DetectionGrounder


class LanguageExampleTests(unittest.TestCase):
    def test_caption_planner_emits_primitive_plan_text(self) -> None:
        planner = CaptionPlanner()
        plan = planner.step(Caption(text='pick the red cube and place it at the goal'))

        self.assertIsInstance(plan, PlanText)
        self.assertGreaterEqual(len(plan.steps), 3)
        self.assertEqual(plan.steps[0].index, 0)
        self.assertEqual(plan.steps[-1].action_label, 'place')

    def test_detection_grounder_resolves_referent_label(self) -> None:
        grounder = DetectionGrounder()
        detections = DetectionBatch(
            detections=(
                Detection2D(label='red', bbox=BBox2D(x=1.0, y=2.0, width=3.0, height=4.0), confidence=0.9),
                Detection2D(label='blue', bbox=BBox2D(x=5.0, y=6.0, width=2.0, height=2.0), confidence=0.8),
            ),
            frame_index=3,
        )
        phrase = grounder.step((ReferringExpression(text='the blue object'), detections))

        self.assertIsInstance(phrase, GroundedPhrase)
        self.assertEqual(phrase.referent_label, 'blue')
        self.assertEqual(phrase.frame_index, 3)


if __name__ == '__main__':
    unittest.main()

"""Minimal advanced language example: ground a referring expression using detections.

Run:
  pixi run -e golden-local demo-language-grounded-reference
  pixi run -e golden-local python -m examples.advanced.language_examples.grounded_reference --steps 6 --dt 0.1
"""

from __future__ import annotations

import argparse

from retriever.flow import Latest, Pipeline, Rate, Trigger

from examples.advanced.language_examples.common import (
    DetectionGrounder,
    GroundedPhrasePrinter,
    ReferringExpressionSource,
)
from examples.advanced.perception_examples.common import ColorDetector, SyntheticColorCamera


def build_pipeline(*, dt: float) -> Pipeline:
    hz = 1.0 / max(dt, 1e-6)
    pipe = Pipeline('advanced_language_grounding')
    with pipe:
        camera = SyntheticColorCamera(dt=dt) @ Rate(hz=hz)
        detector = ColorDetector() @ Trigger('image')
        expression = ReferringExpressionSource() @ Rate(hz=hz)
        grounder = DetectionGrounder() @ Trigger('frame_index')
        printer = GroundedPhrasePrinter() @ Trigger('text')
        pipe.connect(camera, detector, sync=Latest())
        pipe.connect(expression, grounder, sync=Latest())
        pipe.connect(detector, grounder, sync=Latest())
        pipe.connect(grounder, printer, sync=Latest())
    return pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Ground referring expressions against deterministic detections.')
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--dt', type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(dt=args.dt)
    try:
        for _ in range(args.steps):
            pipe.step(dt=args.dt)
    finally:
        pipe.close_stepper()


if __name__ == '__main__':
    main()

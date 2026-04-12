"""Minimal advanced perception example: detection from a deterministic scene.

Run:
  pixi run demo-perception-detection-flow
  pixi run python -m examples.advanced.perception_examples.detection_flow --steps 12 --dt 0.1
"""

from __future__ import annotations

import argparse

from retriever.flow import Latest, Pipeline, Rate, Trigger

from examples.advanced.perception_examples.common import (
    ColorDetector,
    DetectionPrinter,
    SyntheticColorCamera,
)


def build_pipeline(*, dt: float) -> Pipeline:
    hz = 1.0 / max(dt, 1e-6)
    pipe = Pipeline("advanced_perception_detection")
    with pipe:
        camera = SyntheticColorCamera(dt=dt) @ Rate(hz=hz)
        detector = ColorDetector() @ Trigger("image")
        printer = DetectionPrinter() @ Trigger("frame_id")
        pipe.connect(camera, detector, sync=Latest())
        pipe.connect(detector, printer, sync=Latest())
    return pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detection flow over a deterministic synthetic scene.")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(dt=args.dt)
    try:
        for _ in range(args.steps):
            pipe.step(dt=args.dt)
    finally:
        pipe.close_stepper()


if __name__ == "__main__":
    main()

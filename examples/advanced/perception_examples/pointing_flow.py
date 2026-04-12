"""Minimal advanced perception example: convert detections into a target point.

Run:
  pixi run demo-perception-pointing-flow
  pixi run python -m examples.advanced.perception_examples.pointing_flow --target blue --steps 12 --dt 0.1
"""

from __future__ import annotations

import argparse

from retriever.flow import Latest, Pipeline, Rate, Trigger

from examples.advanced.perception_examples.common import (
    ColorDetector,
    PointPrinter,
    PointToLabel,
    SyntheticColorCamera,
)


def build_pipeline(*, dt: float, target_label: str) -> Pipeline:
    hz = 1.0 / max(dt, 1e-6)
    pipe = Pipeline("advanced_perception_pointing")
    with pipe:
        camera = SyntheticColorCamera(dt=dt) @ Rate(hz=hz)
        detector = ColorDetector() @ Trigger("image")
        pointer = PointToLabel(target_label=target_label) @ Trigger("frame_id")
        printer = PointPrinter() @ Trigger("frame_id")
        pipe.connect(camera, detector, sync=Latest())
        pipe.connect(detector, pointer, sync=Latest())
        pipe.connect(pointer, printer, sync=Latest())
    return pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pointing flow over synthetic detections.")
    parser.add_argument("--target", choices=["red", "blue"], default="red")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(dt=args.dt, target_label=args.target)
    try:
        for _ in range(args.steps):
            pipe.step(dt=args.dt)
    finally:
        pipe.close_stepper()


if __name__ == "__main__":
    main()

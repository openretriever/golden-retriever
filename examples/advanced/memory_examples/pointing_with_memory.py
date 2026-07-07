"""Minimal advanced memory example: keep pointing stable through intermittent detections.

Run:
  pixi run -e golden-retriever demo-memory-pointing-flow
  pixi run python -m examples.advanced.memory_examples.pointing_with_memory --target red --steps 12 --dt 0.1
"""

from __future__ import annotations

import argparse

from retriever.flow import Latest, Pipeline, Rate, Trigger

from examples.advanced.memory_examples.common import BeliefTracker, DetectionDropout, SelectBeliefTarget
from examples.advanced.perception_examples.common import ColorDetector, PointPrinter, SyntheticColorCamera


def build_pipeline(*, dt: float, target_label: str) -> Pipeline:
    hz = 1.0 / max(dt, 1e-6)
    pipe = Pipeline("advanced_memory_pointing")
    with pipe:
        camera = SyntheticColorCamera(dt=dt) @ Rate(hz=hz)
        detector = ColorDetector() @ Trigger("image")
        dropout = DetectionDropout(target_label=target_label, every_n=3) @ Trigger("frame_id")
        belief = BeliefTracker(hold_steps=2) @ Trigger("frame_id")
        selector = SelectBeliefTarget(target_label=target_label) @ Trigger("frame_id")
        printer = PointPrinter() @ Trigger("frame_id")
        pipe.connect(camera, detector, sync=Latest())
        pipe.connect(detector, dropout, sync=Latest())
        pipe.connect(dropout, belief, sync=Latest())
        pipe.connect(belief, selector, sync=Latest())
        pipe.connect(selector, printer, sync=Latest())
    return pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pointing flow with remembered detections.")
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

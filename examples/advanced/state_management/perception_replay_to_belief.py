"""
Replay a recorded perception stream into a stateful belief updater.

Run:
  pixi run demo-perception-replay-to-belief
  pixi run python examples/advanced/state_management/perception_replay_to_belief.py --mode combined --steps 12
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from retriever.flow import Flow, Pipeline, Rate, Trigger, Latest, io

from examples.advanced.perception_debug.synthetic_color_stepper import (
    SyntheticCamera,
    ColorDetector,
    DetectionOut,
)
from examples.advanced.perception_debug.replay_utils import load_synthetic_frame_buffer_from_mcap


@io
@dataclass
class BeliefOut:
    frame_id: int | None = None
    color: str | None = None
    x_norm: float | None = None
    confidence: float | None = None


class BeliefTracker(Flow[DetectionOut, BeliefOut]):
    def init(self) -> None:
        self.x_norm = 0.5
        self.confidence = 0.0
        self.color = 'unknown'

    def reset(self) -> None:
        self.x_norm = 0.5
        self.confidence = 0.0
        self.color = 'unknown'

    def step(self, det: DetectionOut) -> BeliefOut:
        if det.frame_id is None or det.dominant_color is None or det.centroid_x is None:
            return BeliefOut()
        self.color = det.dominant_color
        self.x_norm = 0.75 * self.x_norm + 0.25 * (float(det.centroid_x) / 63.0)
        self.confidence = min(1.0, 0.7 * self.confidence + 0.3 * float(det.confidence or 0.0))
        return BeliefOut(
            frame_id=det.frame_id,
            color=self.color,
            x_norm=self.x_norm,
            confidence=self.confidence,
        )


class BeliefPrinter(Flow[BeliefOut, None]):
    def step(self, belief: BeliefOut) -> None:
        if belief.frame_id is None or belief.color is None or belief.x_norm is None:
            return None
        print(
            f"[frame={belief.frame_id:02d}] belief_color={belief.color:<4} x_norm={belief.x_norm:.2f} confidence={float(belief.confidence or 0.0):.2f}"
        )
        return None


def build_pipeline(*, dt: float) -> tuple[Pipeline, object]:
    hz = 1.0 / max(dt, 1e-6)
    pipe = Pipeline('perception_replay_to_belief')
    with pipe:
        camera = SyntheticCamera() @ Rate(hz=hz)
        detector = ColorDetector() @ Trigger('image')
        belief = BeliefTracker() @ Trigger('dominant_color')
        printer = BeliefPrinter() @ Trigger('x_norm')
        pipe.connect(camera, detector, sync=Latest())
        pipe.connect(detector, belief, sync=Latest())
        pipe.connect(belief, printer, sync=Latest())
    return pipe, camera


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Replay a recorded perception stream into a belief updater.')
    p.add_argument('--mode', choices=['combined', 'record', 'replay'], default='combined')
    p.add_argument('--recording', type=Path, default=Path('logs/perception_replay_to_belief.mcap'))
    p.add_argument('--steps', type=int, default=12)
    p.add_argument('--dt', type=float, default=0.1)
    return p.parse_args()


def run_record(path: Path, *, steps: int, dt: float) -> None:
    pipe, _camera = build_pipeline(dt=dt)
    try:
        pipe.record(path, steps=steps, dt=dt)
    finally:
        pipe.close_stepper()
    print(f'[record] wrote {path} ({steps} steps)')


def run_replay(path: Path, *, steps: int, dt: float) -> None:
    if not path.exists():
        raise FileNotFoundError(f'Recording not found: {path}')
    pipe, camera = build_pipeline(dt=dt)
    camera_buffer = load_synthetic_frame_buffer_from_mcap(path)
    pipe.replay(camera, buffer=camera_buffer)
    try:
        for _ in range(steps):
            pipe.step(dt=dt)
    finally:
        pipe.close_stepper()
    print(f'[replay] ran {steps} steps from {path}')


def main() -> None:
    args = parse_args()
    if args.mode in ('combined', 'record'):
        run_record(args.recording, steps=args.steps, dt=args.dt)
    if args.mode in ('combined', 'replay'):
        run_replay(args.recording, steps=args.steps, dt=args.dt)


if __name__ == '__main__':
    main()

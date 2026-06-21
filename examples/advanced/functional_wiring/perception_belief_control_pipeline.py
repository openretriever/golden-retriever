"""
Compose a perception stage and control stage into one runnable pipeline.

Run:
  pixi run demo-perception-belief-control
  pixi run python examples/advanced/functional_wiring/perception_belief_control_pipeline.py --steps 20 --dt 0.1
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from retriever.flow import Flow, Pipeline, Rate, Trigger, Latest, io


@io
@dataclass
class CameraFrame:
    image: np.ndarray | None = None
    frame_id: int | None = None
    t_sim: float | None = None


@io
@dataclass
class Observation:
    frame_id: int | None = None
    color: str | None = None
    x_norm: float | None = None
    confidence: float | None = None


@io
@dataclass
class Belief:
    frame_id: int | None = None
    color: str | None = None
    x_norm: float | None = None
    confidence: float | None = None


@io
@dataclass
class ControlAction:
    frame_id: int | None = None
    action: str | None = None
    reason: str | None = None


class SyntheticTargetSource(Flow[None, CameraFrame]):
    def __init__(self, *, width: int = 80, height: int = 48):
        super().__init__()
        self.width = int(width)
        self.height = int(height)

    def init(self) -> None:
        self.frame_id = 0
        self.t_sim = 0.0

    def reset(self) -> None:
        self.frame_id = 0
        self.t_sim = 0.0

    def step(self, _) -> CameraFrame:
        self.frame_id += 1
        self.t_sim += 0.1
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[..., 1] = 16

        square = 12
        x_norm = 0.15 + 0.7 * (0.5 + 0.5 * math.sin(self.frame_id * 0.22))
        x_center = int(x_norm * (self.width - 1))
        y_center = self.height // 2
        x0 = max(0, x_center - square // 2)
        x1 = min(self.width, x0 + square)
        y0 = max(0, y_center - square // 2)
        y1 = min(self.height, y0 + square)

        if (self.frame_id // 6) % 2 == 0:
            image[y0:y1, x0:x1, 0] = 255
            image[y0:y1, x0:x1, 1] = 40
            image[y0:y1, x0:x1, 2] = 40
        else:
            image[y0:y1, x0:x1, 0] = 40
            image[y0:y1, x0:x1, 1] = 40
            image[y0:y1, x0:x1, 2] = 255

        return CameraFrame(image=image, frame_id=self.frame_id, t_sim=self.t_sim)


class TargetDetector(Flow[CameraFrame, Observation]):
    def step(self, frame: CameraFrame) -> Observation:
        if frame.image is None or frame.frame_id is None:
            return Observation()
        image = frame.image
        red_mask = (image[..., 0] > 180) & (image[..., 1] < 100) & (image[..., 2] < 100)
        blue_mask = (image[..., 2] > 180) & (image[..., 0] < 100) & (image[..., 1] < 100)
        red_pixels = int(red_mask.sum())
        blue_pixels = int(blue_mask.sum())
        if red_pixels == 0 and blue_pixels == 0:
            return Observation(frame_id=frame.frame_id)
        if red_pixels >= blue_pixels:
            color = 'red'
            coords = np.argwhere(red_mask)
            score = red_pixels / max(red_pixels + blue_pixels, 1)
        else:
            color = 'blue'
            coords = np.argwhere(blue_mask)
            score = blue_pixels / max(red_pixels + blue_pixels, 1)
        x_norm = float(coords[:, 1].mean()) / max(image.shape[1] - 1, 1)
        return Observation(frame_id=frame.frame_id, color=color, x_norm=x_norm, confidence=float(score))


class BeliefTracker(Flow[Observation, Belief]):
    def init(self) -> None:
        self.color = 'unknown'
        self.x_norm = 0.5
        self.confidence = 0.0

    def reset(self) -> None:
        self.color = 'unknown'
        self.x_norm = 0.5
        self.confidence = 0.0

    def step(self, obs: Observation) -> Belief:
        if obs.frame_id is None or obs.color is None or obs.x_norm is None:
            return Belief()
        self.color = obs.color
        self.x_norm = 0.75 * self.x_norm + 0.25 * float(obs.x_norm)
        self.confidence = min(1.0, 0.7 * self.confidence + 0.3 * float(obs.confidence or 0.0))
        return Belief(
            frame_id=obs.frame_id,
            color=self.color,
            x_norm=self.x_norm,
            confidence=self.confidence,
        )


class Controller(Flow[Belief, ControlAction]):
    def __init__(self, *, target_x: float = 0.5):
        super().__init__()
        self.target_x = float(target_x)

    def step(self, belief: Belief) -> ControlAction:
        if belief.frame_id is None or belief.x_norm is None or belief.color is None:
            return ControlAction()
        error = float(belief.x_norm) - self.target_x
        if abs(error) < 0.08:
            action = 'hold'
        elif error < 0.0:
            action = 'move_right'
        else:
            action = 'move_left'
        reason = f"track {belief.color} target at x={belief.x_norm:.2f}"
        return ControlAction(frame_id=belief.frame_id, action=action, reason=reason)


class ActionPrinter(Flow[ControlAction, None]):
    def step(self, action: ControlAction) -> None:
        if action.frame_id is None or action.action is None or action.reason is None:
            return None
        print(f"[frame={action.frame_id:02d}] action={action.action:<10} reason={action.reason}")
        return None


def attach_perception_stage(pipe: Pipeline, source: object) -> BeliefTracker:
    detector = TargetDetector() @ Trigger('image')
    belief = BeliefTracker() @ Trigger('color')
    pipe.connect(source, detector, sync=Latest())
    pipe.connect(detector, belief, sync=Latest())
    return belief


def attach_control_stage(pipe: Pipeline, belief: object) -> None:
    controller = Controller() @ Trigger('x_norm')
    printer = ActionPrinter() @ Trigger('action')
    pipe.connect(belief, controller, sync=Latest())
    pipe.connect(controller, printer, sync=Latest())


def build_pipeline(*, dt: float) -> Pipeline:
    hz = 1.0 / max(dt, 1e-6)
    pipe = Pipeline('perception_belief_control_pipeline')
    with pipe:
        source = SyntheticTargetSource() @ Rate(hz=hz)
        belief = attach_perception_stage(pipe, source)
        attach_control_stage(pipe, belief)
    return pipe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Compose a perception stage and control stage.')
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--dt', type=float, default=0.1)
    return p.parse_args()


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

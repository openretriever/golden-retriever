"""
Internal-memory belief updater demo.

Run:
  pixi run demo-belief-updater-internal
  pixi run python examples/advanced/state_management/belief_updater_internal.py --steps 12 --dt 0.1
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from retriever.flow import Flow, Pipeline, Rate, Trigger, Latest, io


@io
@dataclass
class SensorOut:
    t_sim: float | None = None
    reading: float | None = None


@io
@dataclass
class BeliefOut:
    t_sim: float | None = None
    estimate: float | None = None
    confidence: float | None = None
    innovation: float | None = None


class SensorSim(Flow[None, SensorOut]):
    def __init__(self, *, dt: float):
        super().__init__()
        self.dt = float(dt)

    def init_config(self) -> dict:
        return {'dt': self.dt}

    def init(self) -> None:
        self.step_idx = 0
        self.t_sim = 0.0

    def reset(self) -> None:
        self.step_idx = 0
        self.t_sim = 0.0

    def step(self, _) -> SensorOut:
        self.step_idx += 1
        self.t_sim += self.dt
        reading = 0.7 + 0.25 * math.sin(self.step_idx * 0.45)
        return SensorOut(t_sim=self.t_sim, reading=reading)


class BeliefUpdaterInternal(Flow[SensorOut, BeliefOut]):
    def __init__(self, *, alpha: float = 0.35):
        super().__init__()
        self.alpha = float(alpha)

    def init_config(self) -> dict:
        return {'alpha': self.alpha}

    def init(self) -> None:
        self.estimate = 0.0
        self.confidence = 0.25

    def reset(self) -> None:
        self.estimate = 0.0
        self.confidence = 0.25

    def step(self, sensor: SensorOut) -> BeliefOut:
        if sensor.reading is None or sensor.t_sim is None:
            return BeliefOut()
        reading = float(sensor.reading)
        innovation = reading - self.estimate
        self.estimate = self.estimate + self.alpha * innovation
        self.confidence = min(1.0, self.confidence + 0.06)
        return BeliefOut(
            t_sim=float(sensor.t_sim),
            estimate=self.estimate,
            confidence=self.confidence,
            innovation=innovation,
        )


class Printer(Flow[BeliefOut, None]):
    def step(self, belief: BeliefOut) -> None:
        if belief.t_sim is None or belief.estimate is None or belief.confidence is None:
            return None
        print(
            f"[t={belief.t_sim:4.1f}s] estimate={belief.estimate:+.3f} "
            f"innovation={float(belief.innovation or 0.0):+.3f} confidence={belief.confidence:.2f}"
        )
        return None


def build_pipeline(*, dt: float) -> Pipeline:
    hz = 1.0 / max(dt, 1e-6)
    pipe = Pipeline('belief_updater_internal')
    with pipe:
        sensor = SensorSim(dt=dt) @ Rate(hz=hz)
        updater = BeliefUpdaterInternal(alpha=0.35) @ Trigger('reading')
        printer = Printer() @ Trigger('estimate')
        pipe.connect(sensor, updater, sync=Latest())
        pipe.connect(updater, printer, sync=Latest())
    return pipe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Internal-memory belief updater demo.')
    p.add_argument('--steps', type=int, default=12)
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

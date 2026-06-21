"""
Stateful flow reset demo.

Run:
  pixi run demo-stateful-reset
  pixi run python examples/advanced/state_management/stateful_flow_reset.py --steps 5 --dt 0.1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from retriever.flow import Flow, Pipeline, Rate, Trigger, Latest, io


@io
@dataclass
class ScalarOut:
    value: float | None = None


@io
@dataclass
class AverageOut:
    count: int | None = None
    total: float | None = None
    average: float | None = None


class SignalSource(Flow[None, ScalarOut]):
    def __init__(self, pattern: tuple[float, ...] = (1.0, 0.5, -0.5, 1.5)):
        super().__init__()
        self.pattern = tuple(float(v) for v in pattern)

    def init(self) -> None:
        self.idx = 0

    def reset(self) -> None:
        self.idx = 0

    def step(self, _) -> ScalarOut:
        value = self.pattern[self.idx % len(self.pattern)]
        self.idx += 1
        return ScalarOut(value=value)


class RunningAverage(Flow[ScalarOut, AverageOut]):
    def init(self) -> None:
        self.count = 0
        self.total = 0.0

    def reset(self) -> None:
        self.count = 0
        self.total = 0.0

    def step(self, signal: ScalarOut) -> AverageOut:
        if signal.value is None:
            return AverageOut()
        self.count += 1
        self.total += float(signal.value)
        return AverageOut(count=self.count, total=self.total, average=self.total / self.count)


class Printer(Flow[AverageOut, None]):
    def step(self, summary: AverageOut) -> None:
        if summary.count is None or summary.average is None or summary.total is None:
            return None
        print(f"[count={summary.count}] total={summary.total:+.2f} average={summary.average:+.2f}")
        return None


def build_pipeline(*, dt: float) -> Pipeline:
    hz = 1.0 / max(dt, 1e-6)
    pipe = Pipeline('stateful_flow_reset')
    with pipe:
        source = SignalSource() @ Rate(hz=hz)
        averager = RunningAverage() @ Trigger('value')
        printer = Printer() @ Trigger('average')
        pipe.connect(source, averager, sync=Latest())
        pipe.connect(averager, printer, sync=Latest())
    return pipe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Stateful flow reset demo.')
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--dt', type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(dt=args.dt)
    try:
        print('=== run #1 ===')
        for _ in range(args.steps):
            pipe.step(dt=args.dt)

        print('\n=== reset ===')
        pipe.reset()

        print('\n=== run #2 ===')
        for _ in range(args.steps):
            pipe.step(dt=args.dt)
    finally:
        pipe.close_stepper()


if __name__ == '__main__':
    main()

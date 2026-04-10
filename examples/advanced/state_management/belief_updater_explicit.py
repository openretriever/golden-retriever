"""
Explicit-state belief updater (state passed through the pipeline).

Run:
  pixi run demo-belief-updater-explicit
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from retriever.flow import Flow, Pipeline, Rate, Latest, io


@dataclass(frozen=True)
class BeliefState:
    estimate: float
    confidence: float


@io
class SensorOut:
    t_sim: float | None = None
    reading: float | None = None


@io
class StateIn:
    state: BeliefState | None = None


@io
class StateOut:
    state: BeliefState | None = None


@io
class UpdateIn:
    t_sim: float | None = None
    reading: float | None = None
    state: BeliefState | None = None


@io
class UpdateOut:
    t_sim: float | None = None
    estimate: float | None = None
    confidence: float | None = None
    state: BeliefState | None = None


class SensorSim(Flow[None, SensorOut]):
    def __init__(self, *, dt: float):
        super().__init__()
        self.dt = float(dt)

    def init_config(self) -> dict:
        return {"dt": self.dt}

    def init(self) -> None:
        self.step_idx = 0
        self.t_sim = 0.0

    def reset(self) -> None:
        self.step_idx = 0
        self.t_sim = 0.0

    def step(self, _):  # type: ignore[override]
        self.step_idx += 1
        self.t_sim += self.dt
        reading = 0.8 + 0.2 * ((self.step_idx % 6) - 3)
        return SensorOut(t_sim=self.t_sim, reading=reading)


class StateHolder(Flow[StateIn, StateOut]):
    """Holds the current BeliefState, updated via explicit state messages."""

    def __init__(self, *, initial: BeliefState):
        super().__init__()
        self.initial = initial

    def init_config(self) -> dict:
        return {"initial": {"estimate": self.initial.estimate, "confidence": self.initial.confidence}}

    @classmethod
    def from_init_config(cls, config: dict) -> "StateHolder":
        init = config.get("initial", {})
        return cls(initial=BeliefState(estimate=init.get("estimate", 0.0), confidence=init.get("confidence", 0.3)))

    def init(self) -> None:
        self.state = self.initial

    def reset(self) -> None:
        self.state = self.initial

    def step(self, input: StateIn) -> StateOut:
        if input.state is not None:
            self.state = input.state
        return StateOut(state=self.state)


class BeliefUpdater(Flow[UpdateIn, UpdateOut]):
    """Stateless updater: uses incoming state + sensor to produce next state."""

    def __init__(self, *, alpha: float):
        super().__init__()
        self.alpha = float(alpha)

    def init_config(self) -> dict:
        return {"alpha": self.alpha}

    def step(self, input: UpdateIn) -> UpdateOut:
        if input.t_sim is None or input.reading is None or input.state is None:
            return UpdateOut()

        reading = float(input.reading)
        prior = input.state
        estimate = self.alpha * reading + (1.0 - self.alpha) * prior.estimate
        confidence = min(1.0, prior.confidence + 0.05)
        next_state = BeliefState(estimate=estimate, confidence=confidence)

        return UpdateOut(
            t_sim=float(input.t_sim),
            estimate=estimate,
            confidence=confidence,
            state=next_state,
        )


class Printer(Flow[UpdateOut, None]):
    def step(self, input: UpdateOut) -> None:
        if input.t_sim is None or input.estimate is None or input.confidence is None:
            return None
        print(
            f"[t={input.t_sim:4.1f}s] estimate={input.estimate:+.3f} "
            f"confidence={input.confidence:.2f}"
        )
        return None


def build_pipeline(*, dt: float) -> Pipeline:
    hz = 1.0 / max(dt, 1e-6)
    pipe = Pipeline("belief_updater_explicit")

    with pipe:
        sensor = SensorSim(dt=dt) @ Rate(hz=hz)
        holder = StateHolder(initial=BeliefState(estimate=0.0, confidence=0.3)) @ Rate(hz=hz)
        updater = BeliefUpdater(alpha=0.4) @ Rate(hz=hz)
        printer = Printer() @ Rate(hz=hz)

        pipe.connect(
            sensor,
            updater,
            sync=Latest(),
            map={"t_sim": "t_sim", "reading": "reading"},
        )
        pipe.connect(holder, updater, sync=Latest(), map={"state": "state"})

        pipe.connect(updater, holder, sync=Latest(), map={"state": "state"})
        pipe.connect(updater, printer, sync=Latest())

    return pipe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explicit-state belief updater demo.")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--dt", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(dt=args.dt)
    for _ in range(args.steps):
        pipe.step(dt=args.dt)
    pipe.close_stepper()


if __name__ == "__main__":
    main()

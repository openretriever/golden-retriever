"""
Effectful localization with a deterministic sensor simulator.

Run:
  pixi run python examples/advanced/state_management/localization_eff.py --steps 30 --dt 0.1
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace

from retriever.flow import Flow, Pipeline, Rate, flow_io
from retriever.types import Eff


@flow_io
@dataclass
class SensorOut:
    dx: float
    dy: float
    dtheta: float
    gps_x: float
    gps_y: float
    gps_ok: bool
    t: float


@flow_io
@dataclass
class PoseOut:
    x: float
    y: float
    theta: float
    uncertainty: float
    has_gps: bool
    t: float


@dataclass(frozen=True)
class PoseBelief:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    uncertainty: float = 0.2
    t: float = 0.0


def predict(dx: float, dy: float, dtheta: float, dt: float, noise: float) -> Eff[PoseBelief, None]:
    def op(state: PoseBelief) -> tuple[None, PoseBelief]:
        new_state = PoseBelief(
            x=state.x + dx,
            y=state.y + dy,
            theta=state.theta + dtheta,
            uncertainty=min(1.0, state.uncertainty + noise * (abs(dx) + abs(dy) + abs(dtheta))),
            t=state.t + dt,
        )
        return None, new_state

    return Eff(op)


def correct(
    gps_x: float | None,
    gps_y: float | None,
    gps_ok: bool,
    gps_trust: float,
) -> Eff[PoseBelief, bool]:
    def op(state: PoseBelief) -> tuple[bool, PoseBelief]:
        if not gps_ok or gps_x is None or gps_y is None:
            return False, state

        alpha = max(0.0, min(1.0, gps_trust))
        new_state = replace(
            state,
            x=(1.0 - alpha) * state.x + alpha * gps_x,
            y=(1.0 - alpha) * state.y + alpha * gps_y,
            uncertainty=max(0.05, state.uncertainty * (1.0 - 0.5 * alpha)),
        )
        return True, new_state

    return Eff(op)


class SensorSim(Flow[None, SensorOut]):
    def __init__(self, dt: float = 0.1, gps_interval: int = 5):
        super().__init__()
        self.dt = dt
        self.gps_interval = max(1, gps_interval)

    def init_config(self) -> dict:
        return {"dt": self.dt, "gps_interval": self.gps_interval}

    def init(self) -> None:
        self.step = 0
        self.t = 0.0
        self.true_x = 0.0
        self.true_y = 0.0
        self.true_theta = 0.0

    def step(self, _):  # type: ignore[override]
        self.step += 1
        self.t += self.dt

        speed = 0.4
        turn = 0.2

        self.true_theta += turn * self.dt
        true_dx = speed * math.cos(self.true_theta) * self.dt
        true_dy = speed * math.sin(self.true_theta) * self.dt
        self.true_x += true_dx
        self.true_y += true_dy

        dx = true_dx * (1.01 + 0.01 * math.sin(self.step * 0.3))
        dy = true_dy * (0.99 + 0.01 * math.cos(self.step * 0.2))
        dtheta = turn * self.dt * 1.02

        gps_ok = (self.step % self.gps_interval) == 0
        gps_x = None
        gps_y = None
        if gps_ok:
            gps_x = self.true_x + 0.05 * math.sin(self.step * 0.4)
            gps_y = self.true_y + 0.05 * math.cos(self.step * 0.5)

        return SensorOut(
            dx=dx,
            dy=dy,
            dtheta=dtheta,
            gps_x=gps_x,
            gps_y=gps_y,
            gps_ok=gps_ok,
            t=self.t,
        )


class LocalizationFlow(Flow[SensorOut, PoseOut]):
    def __init__(self, process_noise: float = 0.08, gps_trust: float = 0.6):
        super().__init__()
        self.process_noise = process_noise
        self.gps_trust = gps_trust

    def init_config(self) -> dict:
        return {"process_noise": self.process_noise, "gps_trust": self.gps_trust}

    def init(self) -> None:
        self.state = PoseBelief()

    def reset(self) -> None:
        self.state = PoseBelief()

    def step(self, input: SensorOut) -> PoseOut:
        missing = input.dx is None or input.dy is None or input.dtheta is None or input.t is None
        if missing:
            return PoseOut()

        dt = max(1e-6, float(input.t) - self.state.t)

        program = predict(
            dx=float(input.dx),
            dy=float(input.dy),
            dtheta=float(input.dtheta),
            dt=dt,
            noise=self.process_noise,
        ) >> (lambda _: correct(input.gps_x, input.gps_y, bool(input.gps_ok), self.gps_trust))

        has_gps, new_state = program.run(self.state)
        self.state = new_state

        return PoseOut(
            x=new_state.x,
            y=new_state.y,
            theta=new_state.theta,
            uncertainty=new_state.uncertainty,
            has_gps=has_gps,
            t=float(input.t),
        )


class Printer(Flow[PoseOut, None]):
    def __init__(self, print_every: int = 5):
        super().__init__()
        self.print_every = max(1, print_every)

    def init_config(self) -> dict:
        return {"print_every": self.print_every}

    def init(self) -> None:
        self.k = 0

    def step(self, input: PoseOut) -> None:
        if input.x is None or input.y is None or input.t is None:
            return None
        self.k += 1
        if self.k % self.print_every != 0:
            return None
        source = "gps" if input.has_gps else "odom"
        print(
            f"[t={input.t:4.1f}s] x={input.x:+.2f} y={input.y:+.2f} "
            f"theta={input.theta:+.2f} unc={input.uncertainty:.2f} src={source}"
        )
        return None


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    hz = 1.0 / max(args.dt, 1e-6)
    clock = Rate(hz=hz)
    pipe = Pipeline("localization_eff")

    with pipe:
        sensors = SensorSim(dt=args.dt, gps_interval=args.gps_interval) @ clock
        localizer = LocalizationFlow(
            process_noise=args.process_noise, gps_trust=args.gps_trust
        ) @ clock
        printer = Printer(print_every=args.print_every) @ clock

        sensors >> localizer >> printer

    return pipe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Effectful localization demo.")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--gps-interval", type=int, default=5)
    p.add_argument("--gps-trust", type=float, default=0.6)
    p.add_argument("--process-noise", type=float, default=0.08)
    p.add_argument("--print-every", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(args)

    for _ in range(args.steps):
        pipe.step(dt=args.dt)

    pipe.close_stepper()


if __name__ == "__main__":
    main()

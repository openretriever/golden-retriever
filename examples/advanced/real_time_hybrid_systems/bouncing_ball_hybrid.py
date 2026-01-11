"""
Hybrid bouncing ball with mode switching + explicit deadline monitoring.

Run:
  pixi run python examples/advanced/real_time_hybrid_systems/bouncing_ball_hybrid.py --duration 30
  pixi run python examples/advanced/real_time_hybrid_systems/bouncing_ball_hybrid.py --deadline-ms 6 --work-ms 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import retriever
from retriever.flow import Flow, Pipeline, Rate, Trigger, Latest, io

sys.path.append(str(Path(__file__).parent))

from flows import SimClock, DeadlineMonitor, sleep_ms


@io
class ModeInput:
    t_sim: Optional[float] = None
    t_wall: Optional[float] = None
    dt: Optional[float] = None
    deadline_missed: Optional[bool] = None
    miss_count: Optional[int] = None


@io
class RestitutionMode:
    t_sim: Optional[float] = None
    t_wall: Optional[float] = None
    mode: Optional[str] = None
    restitution: Optional[float] = None
    reason: Optional[str] = None


@io
class BallInput:
    t_sim: Optional[float] = None
    t_wall: Optional[float] = None
    dt: Optional[float] = None
    restitution: Optional[float] = None
    mode: Optional[str] = None


@io
class BallState:
    t_sim: Optional[float] = None
    t_wall: Optional[float] = None
    dt: Optional[float] = None
    y: Optional[float] = None
    v: Optional[float] = None
    mode: Optional[str] = None
    restitution: Optional[float] = None
    impact: Optional[bool] = None
    impact_count: Optional[int] = None
    energy: Optional[float] = None


class RestitutionModeFlow(Flow[ModeInput, RestitutionMode]):
    def __init__(
        self, *, elastic: float, damped: float, switch_period: float, miss_limit: int
    ):
        super().__init__()
        self.elastic = float(elastic)
        self.damped = float(damped)
        self.switch_period = float(switch_period)
        self.miss_limit = int(miss_limit)

    def init_config(self) -> dict:
        return {
            "elastic": self.elastic,
            "damped": self.damped,
            "switch_period": self.switch_period,
            "miss_limit": self.miss_limit,
        }

    def init(self) -> None:
        self._mode = "elastic"
        self._last_switch = 0.0

    def step(self, input_data: ModeInput) -> RestitutionMode:
        if input_data.t_sim is None:
            return RestitutionMode()

        t_sim = float(input_data.t_sim)
        reason = "time"
        if (
            input_data.miss_count is not None
            and input_data.miss_count >= self.miss_limit
        ):
            self._mode = "damped"
            reason = "deadline"
        elif t_sim - self._last_switch >= self.switch_period:
            self._mode = "damped" if self._mode == "elastic" else "elastic"
            self._last_switch = t_sim
            reason = "time"

        restitution = self.elastic if self._mode == "elastic" else self.damped
        return RestitutionMode(
            t_sim=t_sim,
            t_wall=input_data.t_wall,
            mode=self._mode,
            restitution=restitution,
            reason=reason,
        )


class BouncingBallSim(Flow[BallInput, BallState]):
    def __init__(self, *, gravity: float, work_ms: float):
        super().__init__()
        self.gravity = float(gravity)
        self.work_ms = float(work_ms)

    def init_config(self) -> dict:
        return {"gravity": self.gravity, "work_ms": self.work_ms}

    def init(self) -> None:
        self.y = 1.0
        self.v = 0.0
        self.impact_count = 0

    def step(self, input_data: BallInput) -> BallState:
        if input_data.t_sim is None or input_data.dt is None:
            return BallState()

        sleep_ms(self.work_ms)

        dt = float(input_data.dt)
        t_sim = float(input_data.t_sim)
        t_wall = float(input_data.t_wall) if input_data.t_wall is not None else None
        restitution = (
            float(input_data.restitution)
            if input_data.restitution is not None
            else 0.85
        )
        mode = input_data.mode or "elastic"

        self.v += -self.gravity * dt
        self.y += self.v * dt

        impact = False
        if self.y <= 0.0 and self.v < 0.0:
            self.y = 0.0
            self.v = -restitution * self.v
            self.impact_count += 1
            impact = True

        energy = 0.5 * self.v * self.v + self.gravity * self.y
        return BallState(
            t_sim=t_sim,
            t_wall=t_wall,
            dt=dt,
            y=self.y,
            v=self.v,
            mode=mode,
            restitution=restitution,
            impact=impact,
            impact_count=self.impact_count,
            energy=energy,
        )


class BallVizFlow(Flow[BallState, None]):
    def __init__(
        self,
        *,
        print_every: int,
        log_rerun: bool,
        namespace: str,
        trail_len: int,
        ground_half_width: float,
        ball_radius: float,
        invert_viz: bool,
    ):
        super().__init__()
        self.print_every = int(print_every)
        self.log_rerun = bool(log_rerun)
        self.namespace = str(namespace)
        self.trail_len = int(trail_len)
        self.ground_half_width = float(ground_half_width)
        self.ball_radius = float(ball_radius)
        self.invert_viz = bool(invert_viz)

    def init_config(self) -> dict:
        return {
            "print_every": self.print_every,
            "log_rerun": self.log_rerun,
            "namespace": self.namespace,
            "trail_len": self.trail_len,
            "ground_half_width": self.ground_half_width,
            "ball_radius": self.ball_radius,
            "invert_viz": self.invert_viz,
        }

    def init(self) -> None:
        self.step_idx = 0
        self.trail: list[list[float]] = []
        self._blueprint_sent = False
        self._legend_logged = False

    def step(self, input_data: BallState) -> None:
        if input_data.t_sim is None or input_data.y is None or input_data.v is None:
            return None

        viz_sign = -1.0 if self.invert_viz else 1.0
        pos = [0.0, viz_sign * float(input_data.y)]
        self.trail.append(pos)
        if len(self.trail) > self.trail_len:
            self.trail = self.trail[-self.trail_len :]

        self.step_idx += 1
        if self.print_every > 0 and self.step_idx % self.print_every == 0:
            print(
                f"[{self.namespace}] t={input_data.t_sim:6.2f} y={input_data.y:6.3f} "
                f"v={input_data.v:6.3f} mode={input_data.mode} impacts={input_data.impact_count}",
                flush=True,
            )

        if not self.log_rerun:
            return None

        self._maybe_send_blueprint()
        self._maybe_log_legend()

        self._log_rerun(input_data, pos)
        return None

    def _log_rerun(self, input_data: BallState, pos: list[float]) -> None:
        try:
            import rerun as rr
            from rerun.archetypes import Scalars
        except Exception:
            return

        if hasattr(rr, "set_time_seconds"):
            rr.set_time_seconds("sim_time", float(input_data.t_sim))
        else:
            rr.set_time("sim_time", timestamp=float(input_data.t_sim))

        ground = [[-self.ground_half_width, 0.0], [self.ground_half_width, 0.0]]
        rr.log(f"{self.namespace}/ground", rr.LineStrips2D([ground]))
        rr.log(f"{self.namespace}/ball", rr.Points2D([pos], radii=self.ball_radius))
        if len(self.trail) > 1:
            rr.log(f"{self.namespace}/trail", rr.LineStrips2D([self.trail]))
        rr.log(f"{self.namespace}/height", Scalars(float(input_data.y)))
        rr.log(f"{self.namespace}/energy", Scalars(float(input_data.energy or 0.0)))
        rr.log(
            f"{self.namespace}/restitution",
            Scalars(float(input_data.restitution or 0.0)),
        )
        rr.log(f"{self.namespace}/mode", rr.TextLog(input_data.mode or ""))
        if input_data.impact:
            rr.log(
                f"{self.namespace}/impact",
                rr.TextLog(f"impact {input_data.impact_count}"),
            )
        return

    def _maybe_send_blueprint(self) -> None:
        if self._blueprint_sent:
            return
        try:
            import rerun as rr
            import rerun.blueprint as rrb
        except Exception:
            return

        layout = rrb.Horizontal(
            rrb.Spatial2DView(origin=f"/{self.namespace}", name="Bounce View"),
            rrb.Vertical(
                rrb.TimeSeriesView(origin=f"/{self.namespace}", name="Energy & Restitution"),
                rrb.TimeSeriesView(origin="/timing/bounce", name="Timing"),
                rrb.TextDocumentView(
                    origin=f"/{self.namespace}",
                    contents=f"/{self.namespace}/legend",
                    name="Legend",
                ),
            ),
            column_shares=[0.55, 0.45],
        )
        rr.send_blueprint(
            rrb.Blueprint(layout, auto_layout=False, auto_views=False),
            make_active=True,
            make_default=True,
        )
        self._blueprint_sent = True

    def _maybe_log_legend(self) -> None:
        if self._legend_logged:
            return
        try:
            import rerun as rr
        except Exception:
            return

        legend = "\n".join(
            [
                "# Bouncing ball",
                "- Y axis = height (screen-friendly by default)",
                "- Ground line at y=0",
                "- Trail shows recent bounces",
                "- Energy + restitution curves show dissipation",
            ]
        )
        media_type = "text/markdown"
        if hasattr(rr, "MediaType"):
            media_type = getattr(rr.MediaType, "MARKDOWN", media_type)
        rr.log(f"{self.namespace}/legend", rr.TextDocument(legend, media_type=media_type))
        self._legend_logged = True


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    retriever.init(default_sync=Latest())
    pipe = Pipeline("bouncing_ball_hybrid")

    time_map = {"t_sim": "t_sim", "t_wall": "t_wall", "dt": "dt"}

    with pipe:
        clock = SimClock(dt=1.0 / args.hz, use_wall=not args.fixed_dt) @ Rate(
            hz=args.hz, on_lag=args.on_lag
        )
        mode = RestitutionModeFlow(
            elastic=args.restitution_elastic,
            damped=args.restitution_damped,
            switch_period=args.mode_period,
            miss_limit=args.miss_limit,
        ) @ Rate(hz=args.mode_hz, on_lag=args.on_lag)
        sim = BouncingBallSim(gravity=args.gravity, work_ms=args.work_ms) @ Trigger(
            "t_sim"
        )
        deadline = DeadlineMonitor(
            deadline_s=args.deadline_ms / 1000.0,
            label="sim_loop",
            print_every=args.print_every,
            log_rerun=not args.no_rerun,
            namespace="timing/bounce",
        ) @ Trigger("t_wall")
        viz = BallVizFlow(
            print_every=args.print_every,
            log_rerun=not args.no_rerun,
            namespace="hybrid/bounce",
            trail_len=args.trail_len,
            ground_half_width=args.ground_width * 0.5,
            ball_radius=args.ball_radius,
            invert_viz=not args.no_invert_viz,
        ) @ Trigger("t_sim")

        pipe.connect(clock, sim, map=time_map, sync=Latest())
        pipe.connect(
            mode, sim, map={"restitution": "restitution", "mode": "mode"}, sync=Latest()
        )

        pipe.connect(clock, mode, map=time_map, sync=Latest())
        pipe.connect(
            deadline,
            mode,
            map={"missed": "deadline_missed", "miss_count": "miss_count"},
            sync=Latest(),
        )

        pipe.connect(
            sim, deadline, map={"t_sim": "t_sim", "t_wall": "t_wall"}, sync=Latest()
        )
        pipe.connect(sim, viz, sync=Latest())

    return pipe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid bouncing ball with deadlines.")
    p.add_argument(
        "--backend", default="dora", choices=["multiprocessing", "dora", "in-process"]
    )
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--hz", type=float, default=120.0)
    p.add_argument("--mode-hz", type=float, default=1.0)
    p.add_argument("--mode-period", type=float, default=8.0)
    p.add_argument("--miss-limit", type=int, default=12)
    p.add_argument("--gravity", type=float, default=9.81)
    p.add_argument("--restitution-elastic", type=float, default=0.98)
    p.add_argument("--restitution-damped", type=float, default=0.92)
    p.add_argument("--deadline-ms", type=float, default=10.0)
    p.add_argument(
        "--work-ms", type=float, default=0.0, help="Simulate compute time per step."
    )
    p.add_argument(
        "--fixed-dt", action="store_true", help="Use fixed dt instead of wall clock."
    )
    p.add_argument(
        "--on-lag", default="catch_up", choices=["warn", "drop", "catch_up", "error"]
    )
    p.add_argument("--print-every", type=int, default=60)
    p.add_argument("--trail-len", type=int, default=180)
    p.add_argument("--ground-width", type=float, default=1.0)
    p.add_argument("--ball-radius", type=float, default=0.06)
    p.add_argument(
        "--no-invert-viz",
        action="store_true",
        help="Disable Y-axis inversion in 2D visualization.",
    )
    p.add_argument(
        "--no-rerun", action="store_true", help="Disable Rerun visualization."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(args)

    print(
        f"[bouncing_ball] backend={args.backend} duration={args.duration}s "
        f"rerun={'off' if args.no_rerun else 'on'}"
    )
    pipe.visualize(open_browser=True)

    visualize = None if args.no_rerun else "rerun"
    pipe.run(
        backend=args.backend, duration=args.duration, visualize=visualize, blocking=True
    )


if __name__ == "__main__":
    main()

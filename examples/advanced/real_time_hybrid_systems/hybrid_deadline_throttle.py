"""
Hybrid thrust control with mode switching + explicit deadlines.

Run:
  pixi run python examples/advanced/real_time_hybrid_systems/hybrid_deadline_throttle.py --duration 30
  pixi run python examples/advanced/real_time_hybrid_systems/hybrid_deadline_throttle.py --work-ms 6 --deadline-ms 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import retriever
from retriever.flow import Flow, Pipeline, Rate, Trigger, Latest, io

sys.path.append(str(Path(__file__).parent))

from flows import SimClock, DeadlineMonitor, clamp, sleep_ms


@io
class ModeInput:
    t_sim: Optional[float] = None
    t_wall: Optional[float] = None
    dt: Optional[float] = None
    alt: Optional[float] = None
    vel: Optional[float] = None
    miss_count: Optional[int] = None
    deadline_missed: Optional[bool] = None


@io
class ModeState:
    t_sim: Optional[float] = None
    t_wall: Optional[float] = None
    mode: Optional[str] = None
    target_v: Optional[float] = None
    reason: Optional[str] = None


@io
class ControlInput:
    t_sim: Optional[float] = None
    t_wall: Optional[float] = None
    dt: Optional[float] = None
    alt: Optional[float] = None
    vel: Optional[float] = None
    mode: Optional[str] = None
    target_v: Optional[float] = None


@io
class ThrottleCommand:
    t_sim: Optional[float] = None
    t_wall: Optional[float] = None
    throttle: Optional[float] = None
    target_v: Optional[float] = None
    mode: Optional[str] = None


@io
class RocketInput:
    t_sim: Optional[float] = None
    t_wall: Optional[float] = None
    dt: Optional[float] = None
    throttle: Optional[float] = None
    mode: Optional[str] = None


@io
class RocketState:
    t_sim: Optional[float] = None
    t_wall: Optional[float] = None
    dt: Optional[float] = None
    alt: Optional[float] = None
    vel: Optional[float] = None
    accel: Optional[float] = None
    throttle: Optional[float] = None
    mode: Optional[str] = None


class ModeManager(Flow[ModeInput, ModeState]):
    def __init__(
        self,
        *,
        target_alt: float,
        band: float,
        climb_rate: float,
        descend_rate: float,
        miss_limit: int,
    ):
        super().__init__()
        self.target_alt = float(target_alt)
        self.band = float(band)
        self.climb_rate = float(climb_rate)
        self.descend_rate = float(descend_rate)
        self.miss_limit = int(miss_limit)

    def init_config(self) -> dict:
        return {
            "target_alt": self.target_alt,
            "band": self.band,
            "climb_rate": self.climb_rate,
            "descend_rate": self.descend_rate,
            "miss_limit": self.miss_limit,
        }

    def init(self) -> None:
        self.mode = "climb"

    def step(self, input_data: ModeInput) -> ModeState:
        if input_data.t_sim is None:
            return ModeState()

        alt = float(input_data.alt) if input_data.alt is not None else 0.0
        miss_count = int(input_data.miss_count or 0)

        reason = "altitude"
        if miss_count >= self.miss_limit:
            self.mode = "safe"
            reason = "deadline"
        else:
            if alt < self.target_alt - self.band:
                self.mode = "climb"
            elif alt > self.target_alt + self.band:
                self.mode = "descend"
            else:
                self.mode = "hold"

        if self.mode == "climb":
            target_v = self.climb_rate
        elif self.mode == "descend":
            target_v = -self.descend_rate
        else:
            target_v = 0.0

        return ModeState(
            t_sim=float(input_data.t_sim),
            t_wall=input_data.t_wall,
            mode=self.mode,
            target_v=target_v,
            reason=reason,
        )


class ThrottleController(Flow[ControlInput, ThrottleCommand]):
    def __init__(
        self,
        *,
        max_thrust: float,
        mass: float,
        kp: float,
        work_ms: float,
    ):
        super().__init__()
        self.max_thrust = float(max_thrust)
        self.mass = float(mass)
        self.kp = float(kp)
        self.work_ms = float(work_ms)
        self.g = 9.81

    def init_config(self) -> dict:
        return {
            "max_thrust": self.max_thrust,
            "mass": self.mass,
            "kp": self.kp,
            "work_ms": self.work_ms,
        }

    def step(self, input_data: ControlInput) -> ThrottleCommand:
        if input_data.t_sim is None:
            return ThrottleCommand()

        sleep_ms(self.work_ms)

        vel = float(input_data.vel) if input_data.vel is not None else 0.0
        target_v = (
            float(input_data.target_v) if input_data.target_v is not None else 0.0
        )
        mode = input_data.mode or "hold"

        hover = self.g * self.mass / self.max_thrust
        throttle = hover + self.kp * (target_v - vel)
        throttle = clamp(throttle, 0.0, 1.0)

        if mode == "safe":
            throttle = 0.0

        return ThrottleCommand(
            t_sim=float(input_data.t_sim),
            t_wall=input_data.t_wall,
            throttle=throttle,
            target_v=target_v,
            mode=mode,
        )


class RocketSim(Flow[RocketInput, RocketState]):
    def __init__(self, *, mass: float, max_thrust: float, drag: float):
        super().__init__()
        self.mass = float(mass)
        self.max_thrust = float(max_thrust)
        self.drag = float(drag)
        self.g = 9.81

    def init_config(self) -> dict:
        return {"mass": self.mass, "max_thrust": self.max_thrust, "drag": self.drag}

    def init(self) -> None:
        self.alt = 0.0
        self.vel = 0.0

    def step(self, input_data: RocketInput) -> RocketState:
        if input_data.t_sim is None or input_data.dt is None:
            return RocketState()

        dt = float(input_data.dt)
        t_sim = float(input_data.t_sim)
        throttle = (
            float(input_data.throttle) if input_data.throttle is not None else 0.0
        )
        mode = input_data.mode or "hold"

        thrust = throttle * self.max_thrust
        accel = thrust / self.mass - self.g - self.drag * self.vel * abs(self.vel)

        self.vel += accel * dt
        self.alt += self.vel * dt

        if self.alt <= 0.0 and self.vel < 0.0:
            self.alt = 0.0
            self.vel = 0.0

        return RocketState(
            t_sim=t_sim,
            t_wall=input_data.t_wall,
            dt=dt,
            alt=self.alt,
            vel=self.vel,
            accel=accel,
            throttle=throttle,
            mode=mode,
        )


class RocketVizFlow(Flow[RocketState, None]):
    def __init__(
        self,
        *,
        print_every: int,
        log_rerun: bool,
        namespace: str,
        trail_len: int,
        ground_half_width: float,
        marker_radius: float,
        target_alt: float,
        band: float,
        invert_viz: bool,
        profile_scale: float,
    ):
        super().__init__()
        self.print_every = int(print_every)
        self.log_rerun = bool(log_rerun)
        self.namespace = str(namespace)
        self.trail_len = int(trail_len)
        self.ground_half_width = float(ground_half_width)
        self.marker_radius = float(marker_radius)
        self.target_alt = float(target_alt)
        self.band = float(band)
        self.invert_viz = bool(invert_viz)
        self.profile_scale = float(profile_scale)

    def init_config(self) -> dict:
        return {
            "print_every": self.print_every,
            "log_rerun": self.log_rerun,
            "namespace": self.namespace,
            "trail_len": self.trail_len,
            "ground_half_width": self.ground_half_width,
            "marker_radius": self.marker_radius,
            "target_alt": self.target_alt,
            "band": self.band,
            "invert_viz": self.invert_viz,
            "profile_scale": self.profile_scale,
        }

    def init(self) -> None:
        self.step_idx = 0
        self.trail: list[list[float]] = []
        self.mode_marks: list[list[float]] = []
        self.last_mode: str | None = None
        self._blueprint_sent = False
        self._legend_logged = False

    def step(self, input_data: RocketState) -> None:
        if input_data.t_sim is None or input_data.alt is None or input_data.vel is None:
            return None

        self.step_idx += 1
        if self.print_every > 0 and self.step_idx % self.print_every == 0:
            print(
                f"[{self.namespace}] t={input_data.t_sim:6.2f} alt={input_data.alt:6.2f} "
                f"vel={input_data.vel:6.2f} mode={input_data.mode} throttle={input_data.throttle:5.2f}",
                flush=True,
            )

        viz_sign = -1.0 if self.invert_viz else 1.0
        x_pos = float(input_data.t_sim) * self.profile_scale
        pos = [x_pos, viz_sign * float(input_data.alt)]
        self.trail.append(pos)
        if len(self.trail) > self.trail_len:
            self.trail = self.trail[-self.trail_len :]

        if not self.log_rerun:
            return None

        self._maybe_send_blueprint()
        self._maybe_log_legend()

        try:
            import rerun as rr
            from rerun.archetypes import Scalars
        except Exception:
            return None

        if hasattr(rr, "set_time_seconds"):
            rr.set_time_seconds("sim_time", float(input_data.t_sim))
        else:
            rr.set_time("sim_time", timestamp=float(input_data.t_sim))

        min_x = max(self.ground_half_width * 2.0, 0.5)
        x_end = max(x_pos, min_x)
        ground = [[0.0, 0.0], [x_end, 0.0]]
        rr.log(f"{self.namespace}/ground", rr.LineStrips2D([ground]))

        target = [
            [0.0, viz_sign * self.target_alt],
            [x_end, viz_sign * self.target_alt],
        ]
        rr.log(f"{self.namespace}/target_alt", rr.LineStrips2D([target]))
        if self.band > 0.0:
            low = viz_sign * (self.target_alt - self.band)
            high = viz_sign * (self.target_alt + self.band)
            band_low = [[0.0, low], [x_end, low]]
            band_high = [[0.0, high], [x_end, high]]
            rr.log(f"{self.namespace}/band_low", rr.LineStrips2D([band_low]))
            rr.log(f"{self.namespace}/band_high", rr.LineStrips2D([band_high]))

        stem = [[x_pos, 0.0], pos]
        rr.log(f"{self.namespace}/stem", rr.LineStrips2D([stem]))
        rr.log(f"{self.namespace}/rocket", rr.Points2D([pos], radii=self.marker_radius))
        if len(self.trail) > 1:
            rr.log(f"{self.namespace}/trail", rr.LineStrips2D([self.trail]))
        if input_data.mode:
            if input_data.mode != self.last_mode:
                self.last_mode = str(input_data.mode)
                self.mode_marks.append(pos)
                rr.log(
                    f"{self.namespace}/mode_event",
                    rr.TextLog(f"{input_data.mode} @ t={input_data.t_sim:0.1f}s"),
                )
            if self.mode_marks:
                rr.log(
                    f"{self.namespace}/mode_marks",
                    rr.Points2D(self.mode_marks, radii=self.marker_radius * 0.6),
                )

        rr.log(f"{self.namespace}/altitude", Scalars(float(input_data.alt)))
        rr.log(f"{self.namespace}/velocity", Scalars(float(input_data.vel)))
        rr.log(f"{self.namespace}/throttle", Scalars(float(input_data.throttle or 0.0)))
        rr.log(f"{self.namespace}/mode", rr.TextLog(input_data.mode or ""))
        return None

    def _maybe_send_blueprint(self) -> None:
        if self._blueprint_sent:
            return
        try:
            import rerun as rr
            import rerun.blueprint as rrb
        except Exception:
            return

        layout = rrb.Horizontal(
            rrb.Spatial2DView(origin=f"/{self.namespace}", name="Altitude Profile"),
            rrb.Vertical(
                rrb.TimeSeriesView(origin=f"/{self.namespace}", name="Throttle Signals"),
                rrb.TimeSeriesView(origin="/timing/throttle", name="Timing"),
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
                "# Throttle profile",
                "- X axis = time (scaled by `--profile-scale`)",
                "- Y axis = altitude",
                "- Lines = target altitude band",
                "- Dot + stem = current vehicle state",
                "- Small dots = mode transitions (climb/hold/descend/safe)",
            ]
        )
        media_type = "text/markdown"
        if hasattr(rr, "MediaType"):
            media_type = getattr(rr.MediaType, "MARKDOWN", media_type)
        rr.log(f"{self.namespace}/legend", rr.TextDocument(legend, media_type=media_type))
        self._legend_logged = True


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    retriever.init(default_sync=Latest())
    pipe = Pipeline("hybrid_deadline_throttle")

    time_map = {"t_sim": "t_sim", "t_wall": "t_wall", "dt": "dt"}

    with pipe:
        clock = SimClock(dt=1.0 / args.hz, use_wall=not args.fixed_dt) @ Rate(
            hz=args.hz, on_lag=args.on_lag
        )
        mode = ModeManager(
            target_alt=args.target_alt,
            band=args.band,
            climb_rate=args.climb_rate,
            descend_rate=args.descend_rate,
            miss_limit=args.miss_limit,
        ) @ Rate(hz=args.mode_hz, on_lag=args.on_lag)
        controller = ThrottleController(
            max_thrust=args.max_thrust,
            mass=args.mass,
            kp=args.kp,
            work_ms=args.work_ms,
        ) @ Rate(hz=args.ctrl_hz, on_lag=args.on_lag)
        sim = RocketSim(
            mass=args.mass,
            max_thrust=args.max_thrust,
            drag=args.drag,
        ) @ Trigger("t_sim")
        deadline = DeadlineMonitor(
            deadline_s=args.deadline_ms / 1000.0,
            label="ctrl_loop",
            print_every=args.print_every,
            log_rerun=not args.no_rerun,
            namespace="timing/throttle",
        ) @ Trigger("t_wall")
        viz = RocketVizFlow(
            print_every=args.print_every,
            log_rerun=not args.no_rerun,
            namespace="hybrid/throttle",
            trail_len=args.trail_len,
            ground_half_width=args.ground_width * 0.5,
            marker_radius=args.marker_radius,
            target_alt=args.target_alt,
            band=args.band,
            invert_viz=not args.no_invert_viz,
            profile_scale=args.profile_scale,
        ) @ Trigger("t_sim")

        pipe.connect(clock, sim, map=time_map, sync=Latest())
        pipe.connect(
            controller, sim, map={"throttle": "throttle", "mode": "mode"}, sync=Latest()
        )

        pipe.connect(clock, controller, map=time_map, sync=Latest())
        pipe.connect(sim, controller, map={"alt": "alt", "vel": "vel"}, sync=Latest())
        pipe.connect(
            mode,
            controller,
            map={"mode": "mode", "target_v": "target_v"},
            sync=Latest(),
        )

        pipe.connect(clock, mode, map=time_map, sync=Latest())
        pipe.connect(sim, mode, map={"alt": "alt", "vel": "vel"}, sync=Latest())
        pipe.connect(
            deadline,
            mode,
            map={"miss_count": "miss_count", "missed": "deadline_missed"},
            sync=Latest(),
        )

        pipe.connect(
            controller,
            deadline,
            map={"t_sim": "t_sim", "t_wall": "t_wall"},
            sync=Latest(),
        )
        pipe.connect(sim, viz, sync=Latest())

    return pipe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hybrid thrust control with explicit deadlines."
    )
    p.add_argument(
        "--backend", default="dora", choices=["multiprocessing", "dora", "in-process"]
    )
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--hz", type=float, default=120.0)
    p.add_argument("--ctrl-hz", type=float, default=40.0)
    p.add_argument("--mode-hz", type=float, default=2.0)
    p.add_argument("--target-alt", type=float, default=5.0)
    p.add_argument("--band", type=float, default=0.4)
    p.add_argument("--climb-rate", type=float, default=1.2)
    p.add_argument("--descend-rate", type=float, default=1.0)
    p.add_argument("--miss-limit", type=int, default=3)
    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--max-thrust", type=float, default=20.0)
    p.add_argument("--drag", type=float, default=0.2)
    p.add_argument("--kp", type=float, default=0.6)
    p.add_argument("--deadline-ms", type=float, default=20.0)
    p.add_argument(
        "--work-ms",
        type=float,
        default=0.0,
        help="Simulate compute time per control step.",
    )
    p.add_argument(
        "--fixed-dt", action="store_true", help="Use fixed dt instead of wall clock."
    )
    p.add_argument(
        "--on-lag", default="catch_up", choices=["warn", "drop", "catch_up", "error"]
    )
    p.add_argument("--print-every", type=int, default=40)
    p.add_argument("--trail-len", type=int, default=240)
    p.add_argument("--ground-width", type=float, default=1.2)
    p.add_argument("--marker-radius", type=float, default=0.08)
    p.add_argument(
        "--profile-scale",
        type=float,
        default=0.2,
        help="Scale sim_time to X for the altitude profile view.",
    )
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
        f"[hybrid_deadline_throttle] backend={args.backend} duration={args.duration}s "
        f"rerun={'off' if args.no_rerun else 'on'}"
    )
    pipe.visualize(open_browser=True)

    visualize = None if args.no_rerun else "rerun"
    pipe.run(
        backend=args.backend, duration=args.duration, visualize=visualize, blocking=True
    )


if __name__ == "__main__":
    main()

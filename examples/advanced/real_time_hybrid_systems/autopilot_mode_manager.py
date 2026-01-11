"""
Autopilot mode manager with explicit deadlines (aero-style hybrid system).

Run:
  pixi run python examples/advanced/real_time_hybrid_systems/autopilot_mode_manager.py --duration 30
  pixi run python examples/advanced/real_time_hybrid_systems/autopilot_mode_manager.py --work-ms 5 --deadline-ms 15
"""

from __future__ import annotations

import argparse
import math
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
    vs: Optional[float] = None
    miss_count: Optional[int] = None
    deadline_missed: Optional[bool] = None


@io
class AutopilotMode:
    t_sim: Optional[float] = None
    t_wall: Optional[float] = None
    mode: Optional[str] = None
    target_vs: Optional[float] = None
    target_pitch: Optional[float] = None
    reason: Optional[str] = None


@io
class ControlInput:
    t_sim: Optional[float] = None
    t_wall: Optional[float] = None
    dt: Optional[float] = None
    alt: Optional[float] = None
    vs: Optional[float] = None
    pitch: Optional[float] = None
    mode: Optional[str] = None
    target_vs: Optional[float] = None
    target_pitch: Optional[float] = None


@io
class AutopilotCommand:
    t_sim: Optional[float] = None
    t_wall: Optional[float] = None
    thrust_cmd: Optional[float] = None
    pitch_cmd: Optional[float] = None
    target_vs: Optional[float] = None
    mode: Optional[str] = None


@io
class AircraftInput:
    t_sim: Optional[float] = None
    t_wall: Optional[float] = None
    dt: Optional[float] = None
    thrust_cmd: Optional[float] = None
    pitch_cmd: Optional[float] = None
    mode: Optional[str] = None


@io
class AircraftState:
    t_sim: Optional[float] = None
    t_wall: Optional[float] = None
    dt: Optional[float] = None
    alt: Optional[float] = None
    vs: Optional[float] = None
    pitch: Optional[float] = None
    pitch_rate: Optional[float] = None
    thrust_cmd: Optional[float] = None
    mode: Optional[str] = None


class AutopilotModeManager(Flow[ModeInput, AutopilotMode]):
    def __init__(
        self,
        *,
        takeoff_alt: float,
        cruise_alt: float,
        flare_alt: float,
        cruise_time: float,
        miss_limit: int,
    ):
        super().__init__()
        self.takeoff_alt = float(takeoff_alt)
        self.cruise_alt = float(cruise_alt)
        self.flare_alt = float(flare_alt)
        self.cruise_time = float(cruise_time)
        self.miss_limit = int(miss_limit)

    def init_config(self) -> dict:
        return {
            "takeoff_alt": self.takeoff_alt,
            "cruise_alt": self.cruise_alt,
            "flare_alt": self.flare_alt,
            "cruise_time": self.cruise_time,
            "miss_limit": self.miss_limit,
        }

    def init(self) -> None:
        self.mode = "takeoff"

    def step(self, input_data: ModeInput) -> AutopilotMode:
        if input_data.t_sim is None:
            return AutopilotMode()

        alt = float(input_data.alt) if input_data.alt is not None else 0.0
        t_sim = float(input_data.t_sim)
        miss_count = int(input_data.miss_count or 0)

        reason = "altitude"
        if miss_count >= self.miss_limit:
            self.mode = "safe"
            reason = "deadline"
        else:
            if self.mode == "takeoff" and alt >= self.takeoff_alt:
                self.mode = "climb"
            elif self.mode == "climb" and alt >= self.cruise_alt:
                self.mode = "cruise"
            elif self.mode == "cruise" and t_sim >= self.cruise_time:
                self.mode = "descent"
            elif self.mode == "descent" and alt <= self.flare_alt:
                self.mode = "flare"

        if self.mode == "takeoff":
            target_vs = 3.0
            target_pitch = math.radians(12.0)
        elif self.mode == "climb":
            target_vs = 2.0
            target_pitch = math.radians(8.0)
        elif self.mode == "cruise":
            target_vs = 0.0
            target_pitch = math.radians(2.0)
        elif self.mode == "descent":
            target_vs = -1.5
            target_pitch = math.radians(-4.0)
        elif self.mode == "flare":
            target_vs = -0.3
            target_pitch = math.radians(-2.0)
        else:
            target_vs = 0.0
            target_pitch = 0.0

        return AutopilotMode(
            t_sim=t_sim,
            t_wall=input_data.t_wall,
            mode=self.mode,
            target_vs=target_vs,
            target_pitch=target_pitch,
            reason=reason,
        )


class AutopilotController(Flow[ControlInput, AutopilotCommand]):
    def __init__(
        self,
        *,
        kp_vs: float,
        kp_pitch: float,
        base_thrust: float,
        max_pitch_deg: float,
        work_ms: float,
    ):
        super().__init__()
        self.kp_vs = float(kp_vs)
        self.kp_pitch = float(kp_pitch)
        self.base_thrust = float(base_thrust)
        self.max_pitch = math.radians(float(max_pitch_deg))
        self.work_ms = float(work_ms)

    def init_config(self) -> dict:
        return {
            "kp_vs": self.kp_vs,
            "kp_pitch": self.kp_pitch,
            "base_thrust": self.base_thrust,
            "max_pitch_deg": math.degrees(self.max_pitch),
            "work_ms": self.work_ms,
        }

    def step(self, input_data: ControlInput) -> AutopilotCommand:
        if input_data.t_sim is None:
            return AutopilotCommand()

        sleep_ms(self.work_ms)

        vs = float(input_data.vs) if input_data.vs is not None else 0.0
        target_vs = (
            float(input_data.target_vs) if input_data.target_vs is not None else 0.0
        )
        target_pitch = (
            float(input_data.target_pitch)
            if input_data.target_pitch is not None
            else 0.0
        )
        mode = input_data.mode or "cruise"

        err_vs = target_vs - vs
        pitch_cmd = clamp(
            target_pitch + self.kp_pitch * err_vs, -self.max_pitch, self.max_pitch
        )
        thrust_cmd = clamp(self.base_thrust + self.kp_vs * err_vs, 0.0, 1.0)

        if mode == "safe":
            thrust_cmd = 0.4
            pitch_cmd = 0.0

        return AutopilotCommand(
            t_sim=float(input_data.t_sim),
            t_wall=input_data.t_wall,
            thrust_cmd=thrust_cmd,
            pitch_cmd=pitch_cmd,
            target_vs=target_vs,
            mode=mode,
        )


class AircraftSim(Flow[AircraftInput, AircraftState]):
    def __init__(self, *, max_accel: float, pitch_gain: float, drag: float, tau: float):
        super().__init__()
        self.max_accel = float(max_accel)
        self.pitch_gain = float(pitch_gain)
        self.drag = float(drag)
        self.tau = float(tau)
        self.g = 9.81

    def init_config(self) -> dict:
        return {
            "max_accel": self.max_accel,
            "pitch_gain": self.pitch_gain,
            "drag": self.drag,
            "tau": self.tau,
        }

    def init(self) -> None:
        self.alt = 0.0
        self.vs = 0.0
        self.pitch = 0.0
        self.pitch_rate = 0.0

    def step(self, input_data: AircraftInput) -> AircraftState:
        if input_data.t_sim is None or input_data.dt is None:
            return AircraftState()

        dt = float(input_data.dt)
        t_sim = float(input_data.t_sim)
        thrust_cmd = (
            float(input_data.thrust_cmd) if input_data.thrust_cmd is not None else 0.5
        )
        pitch_cmd = (
            float(input_data.pitch_cmd) if input_data.pitch_cmd is not None else 0.0
        )
        mode = input_data.mode or "cruise"

        self.pitch_rate = (pitch_cmd - self.pitch) / max(self.tau, 1e-3)
        self.pitch += self.pitch_rate * dt

        accel = (
            thrust_cmd * self.max_accel
            + self.pitch_gain * self.pitch
            - self.drag * self.vs
            - self.g
        )
        self.vs += accel * dt
        self.alt += self.vs * dt

        if self.alt <= 0.0 and self.vs < 0.0:
            self.alt = 0.0
            self.vs = 0.0

        return AircraftState(
            t_sim=t_sim,
            t_wall=input_data.t_wall,
            dt=dt,
            alt=self.alt,
            vs=self.vs,
            pitch=self.pitch,
            pitch_rate=self.pitch_rate,
            thrust_cmd=thrust_cmd,
            mode=mode,
        )


class AutopilotVizFlow(Flow[AircraftState, None]):
    def __init__(
        self,
        *,
        print_every: int,
        log_rerun: bool,
        namespace: str,
        trail_len: int,
        ground_half_width: float,
        marker_radius: float,
        takeoff_alt: float,
        cruise_alt: float,
        flare_alt: float,
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
        self.takeoff_alt = float(takeoff_alt)
        self.cruise_alt = float(cruise_alt)
        self.flare_alt = float(flare_alt)
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
            "takeoff_alt": self.takeoff_alt,
            "cruise_alt": self.cruise_alt,
            "flare_alt": self.flare_alt,
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

    def step(self, input_data: AircraftState) -> None:
        if input_data.t_sim is None or input_data.alt is None or input_data.vs is None:
            return None

        self.step_idx += 1
        if self.print_every > 0 and self.step_idx % self.print_every == 0:
            print(
                f"[{self.namespace}] t={input_data.t_sim:6.2f} alt={input_data.alt:6.2f} "
                f"vs={input_data.vs:6.2f} pitch={math.degrees(input_data.pitch or 0.0):5.1f} "
                f"mode={input_data.mode}",
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
        takeoff = [
            [0.0, viz_sign * self.takeoff_alt],
            [x_end, viz_sign * self.takeoff_alt],
        ]
        cruise = [
            [0.0, viz_sign * self.cruise_alt],
            [x_end, viz_sign * self.cruise_alt],
        ]
        flare = [
            [0.0, viz_sign * self.flare_alt],
            [x_end, viz_sign * self.flare_alt],
        ]
        rr.log(f"{self.namespace}/takeoff_alt", rr.LineStrips2D([takeoff]))
        rr.log(f"{self.namespace}/cruise_alt", rr.LineStrips2D([cruise]))
        rr.log(f"{self.namespace}/flare_alt", rr.LineStrips2D([flare]))

        stem = [[x_pos, 0.0], pos]
        rr.log(f"{self.namespace}/stem", rr.LineStrips2D([stem]))
        rr.log(f"{self.namespace}/aircraft", rr.Points2D([pos], radii=self.marker_radius))
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
        rr.log(f"{self.namespace}/vertical_speed", Scalars(float(input_data.vs)))
        rr.log(
            f"{self.namespace}/pitch_deg",
            Scalars(float(math.degrees(input_data.pitch or 0.0))),
        )
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
                rrb.TimeSeriesView(origin=f"/{self.namespace}", name="Autopilot Signals"),
                rrb.TimeSeriesView(origin="/timing/autopilot", name="Timing"),
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
                "# Autopilot profile",
                "- X axis = time (scaled by `--profile-scale`)",
                "- Y axis = altitude",
                "- Lines = target altitudes (takeoff / cruise / flare)",
                "- Dot + stem = current aircraft state",
                "- Small dots = mode transitions",
            ]
        )
        media_type = "text/markdown"
        if hasattr(rr, "MediaType"):
            media_type = getattr(rr.MediaType, "MARKDOWN", media_type)
        rr.log(f"{self.namespace}/legend", rr.TextDocument(legend, media_type=media_type))
        self._legend_logged = True


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    retriever.init(default_sync=Latest())
    pipe = Pipeline("autopilot_mode_manager")

    time_map = {"t_sim": "t_sim", "t_wall": "t_wall", "dt": "dt"}

    with pipe:
        clock = SimClock(dt=1.0 / args.hz, use_wall=not args.fixed_dt) @ Rate(
            hz=args.hz, on_lag=args.on_lag
        )
        mode = AutopilotModeManager(
            takeoff_alt=args.takeoff_alt,
            cruise_alt=args.cruise_alt,
            flare_alt=args.flare_alt,
            cruise_time=args.cruise_time,
            miss_limit=args.miss_limit,
        ) @ Rate(hz=args.mode_hz, on_lag=args.on_lag)
        controller = AutopilotController(
            kp_vs=args.kp_vs,
            kp_pitch=args.kp_pitch,
            base_thrust=args.base_thrust,
            max_pitch_deg=args.max_pitch_deg,
            work_ms=args.work_ms,
        ) @ Rate(hz=args.ctrl_hz, on_lag=args.on_lag)
        sim = AircraftSim(
            max_accel=args.max_accel,
            pitch_gain=args.pitch_gain,
            drag=args.drag,
            tau=args.tau,
        ) @ Trigger("t_sim")
        deadline = DeadlineMonitor(
            deadline_s=args.deadline_ms / 1000.0,
            label="autopilot_loop",
            print_every=args.print_every,
            log_rerun=not args.no_rerun,
            namespace="timing/autopilot",
        ) @ Trigger("t_wall")
        viz = AutopilotVizFlow(
            print_every=args.print_every,
            log_rerun=not args.no_rerun,
            namespace="aero/autopilot",
            trail_len=args.trail_len,
            ground_half_width=args.ground_width * 0.5,
            marker_radius=args.marker_radius,
            takeoff_alt=args.takeoff_alt,
            cruise_alt=args.cruise_alt,
            flare_alt=args.flare_alt,
            invert_viz=not args.no_invert_viz,
            profile_scale=args.profile_scale,
        ) @ Trigger("t_sim")

        pipe.connect(clock, sim, map=time_map, sync=Latest())
        pipe.connect(
            controller,
            sim,
            map={"thrust_cmd": "thrust_cmd", "pitch_cmd": "pitch_cmd", "mode": "mode"},
            sync=Latest(),
        )

        pipe.connect(clock, controller, map=time_map, sync=Latest())
        pipe.connect(
            sim,
            controller,
            map={"alt": "alt", "vs": "vs", "pitch": "pitch"},
            sync=Latest(),
        )
        pipe.connect(
            mode,
            controller,
            map={
                "mode": "mode",
                "target_vs": "target_vs",
                "target_pitch": "target_pitch",
            },
            sync=Latest(),
        )

        pipe.connect(clock, mode, map=time_map, sync=Latest())
        pipe.connect(sim, mode, map={"alt": "alt", "vs": "vs"}, sync=Latest())
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
    p = argparse.ArgumentParser(description="Autopilot mode manager with deadlines.")
    p.add_argument(
        "--backend", default="dora", choices=["multiprocessing", "dora", "in-process"]
    )
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--hz", type=float, default=120.0)
    p.add_argument("--ctrl-hz", type=float, default=40.0)
    p.add_argument("--mode-hz", type=float, default=2.0)
    p.add_argument("--takeoff-alt", type=float, default=1.0)
    p.add_argument("--cruise-alt", type=float, default=8.0)
    p.add_argument("--flare-alt", type=float, default=1.5)
    p.add_argument("--cruise-time", type=float, default=6.0)
    p.add_argument("--miss-limit", type=int, default=3)
    p.add_argument("--max-accel", type=float, default=15.0)
    p.add_argument("--pitch-gain", type=float, default=2.0)
    p.add_argument("--drag", type=float, default=0.3)
    p.add_argument("--tau", type=float, default=0.4)
    p.add_argument("--kp-vs", type=float, default=0.2)
    p.add_argument("--kp-pitch", type=float, default=0.4)
    p.add_argument("--base-thrust", type=float, default=0.7)
    p.add_argument("--max-pitch-deg", type=float, default=12.0)
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
    p.add_argument("--trail-len", type=int, default=260)
    p.add_argument("--ground-width", type=float, default=1.6)
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
        f"[autopilot_mode_manager] backend={args.backend} duration={args.duration}s "
        f"rerun={'off' if args.no_rerun else 'on'}"
    )
    pipe.visualize(open_browser=True)

    visualize = None if args.no_rerun else "rerun"
    pipe.run(
        backend=args.backend, duration=args.duration, visualize=visualize, blocking=True
    )


if __name__ == "__main__":
    main()

"""
Hierarchical physics flows (double pendulum + three-body).

Run:
  pixi run python examples/advanced/hierarchical_physics_demo/app.py --demo double_pendulum --duration 6
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

from retriever.flow import Flow, io


@io
class SimTime:
    t: Optional[float] = None
    dt: Optional[float] = None


@io
class PendulumState:
    t: Optional[float] = None
    theta1: Optional[float] = None
    theta2: Optional[float] = None
    omega1: Optional[float] = None
    omega2: Optional[float] = None
    energy: Optional[float] = None
    p1: Optional[tuple[float, float]] = None
    p2: Optional[tuple[float, float]] = None


@io
class PendulumViz:
    points: Optional[list[list[float]]] = None
    trail: Optional[list[list[float]]] = None
    energy: Optional[float] = None
    t: Optional[float] = None


@io
class NBodyState:
    t: Optional[float] = None
    positions: Optional[list[list[float]]] = None
    velocities: Optional[list[list[float]]] = None
    energy: Optional[float] = None


@io
class NBodyViz:
    points: Optional[list[list[float]]] = None
    trails: Optional[list[list[list[float]]]] = None
    energy: Optional[float] = None
    t: Optional[float] = None


@io
class NCoupledPendulumState:
    t: Optional[float] = None
    n: Optional[int] = None
    thetas: Optional[list[float]] = None
    omegas: Optional[list[float]] = None
    energy: Optional[float] = None
    positions: Optional[list[list[float]]] = None
    pivot_positions: Optional[list[list[float]]] = None


@io
class NCoupledPendulumViz:
    pivot_points: Optional[list[list[float]]] = None
    bob_points: Optional[list[list[float]]] = None
    rod_segments: Optional[list[list[list[float]]]] = None
    spring_segments: Optional[list[list[list[float]]]] = None
    trails: Optional[list[list[list[float]]]] = None
    energy: Optional[float] = None
    t: Optional[float] = None


@io
class NCoupledSpringState:
    t: Optional[float] = None
    n: Optional[int] = None
    positions: Optional[list[list[float]]] = None
    velocities: Optional[list[list[float]]] = None
    energy: Optional[float] = None


@io
class NCoupledSpringViz:
    mass_points: Optional[list[list[float]]] = None
    spring_segments: Optional[list[list[list[float]]]] = None
    trails: Optional[list[list[list[float]]]] = None
    energy: Optional[float] = None
    t: Optional[float] = None


class PipelineVizFlow(Flow[SimTime, None]):
    def __init__(
        self,
        html_path: str,
        *,
        ascii_path: str | None = None,
        log_rerun: bool,
        namespace: str,
    ):
        super().__init__()
        self.html_path = str(html_path)
        self.ascii_path = str(ascii_path) if ascii_path else None
        self.log_rerun = bool(log_rerun)
        self.namespace = str(namespace)

    def init_config(self) -> dict:
        return {
            "html_path": self.html_path,
            "ascii_path": self.ascii_path,
            "log_rerun": self.log_rerun,
            "namespace": self.namespace,
        }

    def init(self) -> None:
        self._logged = False

    def step(self, input_data: SimTime) -> None:
        if self._logged or input_data.t is None:
            return None

        self._logged = True
        html_path = Path(self.html_path)
        if not html_path.exists():
            print(f"[{self.namespace}] pipeline viz missing: {html_path}", flush=True)
            return None

        ascii_path = Path(self.ascii_path) if self.ascii_path else None
        if ascii_path and not ascii_path.exists():
            ascii_path = None

        print(f"[{self.namespace}] pipeline viz html: {html_path}", flush=True)
        if ascii_path:
            print(f"[{self.namespace}] pipeline viz ascii: {ascii_path}", flush=True)

        if not self.log_rerun:
            return None

        try:
            import rerun as rr
        except Exception as exc:
            print(f"[{self.namespace}] pipeline viz log failed: {exc}", flush=True)
            return None

        if hasattr(rr, "set_time_seconds"):
            rr.set_time_seconds("sim_time", float(input_data.t))
        else:
            rr.set_time("sim_time", timestamp=float(input_data.t))

        ascii_text = None
        doc_lines = [f"Pipeline HTML: `{html_path}`"]
        if ascii_path:
            try:
                ascii_text = ascii_path.read_text()
                doc_lines.extend(["", "Pipeline Graph (ASCII):", "```", ascii_text, "```"])
            except Exception as exc:
                doc_lines.append(f"\nPipeline ASCII read failed: {exc}")

        doc_text = "\n".join(doc_lines)

        media_type = "text/markdown"
        if hasattr(rr, "MediaType"):
            media_type = getattr(rr.MediaType, "MARKDOWN", media_type)

        rr.log(f"{self.namespace}/pipeline_doc", rr.TextDocument(doc_text, media_type=media_type))
        rr.log(f"{self.namespace}/pipeline_path", rr.TextDocument(str(html_path)))
        return None


class SimClock(Flow[None, SimTime]):
    def __init__(self, dt: float, use_wall: bool = False):
        super().__init__()
        self.dt = float(dt)
        self.use_wall = bool(use_wall)

    def init_config(self) -> dict:
        return {"dt": self.dt, "use_wall": self.use_wall}

    def init(self) -> None:
        self.t = 0.0
        self._last_wall = time.perf_counter()

    def step(self, _input: None) -> SimTime:
        if self.use_wall:
            now = time.perf_counter()
            dt = max(now - self._last_wall, 1e-6)
            self._last_wall = now
        else:
            dt = self.dt

        self.t += dt
        return SimTime(t=self.t, dt=dt)


class DoublePendulumSim(Flow[SimTime, PendulumState]):
    def __init__(
        self,
        length1: float = 1.0,
        length2: float = 1.0,
        mass1: float = 1.0,
        mass2: float = 1.0,
        gravity: float = 9.81,
        damping: float = 0.01,
    ):
        super().__init__()
        self.length1 = float(length1)
        self.length2 = float(length2)
        self.mass1 = float(mass1)
        self.mass2 = float(mass2)
        self.gravity = float(gravity)
        self.damping = float(damping)

    def init_config(self) -> dict:
        return {
            "length1": self.length1,
            "length2": self.length2,
            "mass1": self.mass1,
            "mass2": self.mass2,
            "gravity": self.gravity,
            "damping": self.damping,
        }

    def init(self) -> None:
        self.theta1 = 1.3
        self.theta2 = 1.0
        self.omega1 = 0.0
        self.omega2 = 0.0

    def step(self, input_data: SimTime) -> PendulumState:
        if input_data.t is None or input_data.dt is None:
            return PendulumState()

        dt = float(input_data.dt)
        t = float(input_data.t)
        m1 = self.mass1
        m2 = self.mass2
        l1 = self.length1
        l2 = self.length2
        g = self.gravity

        delta = self.theta1 - self.theta2
        denom = 2 * m1 + m2 - m2 * math.cos(2 * delta)
        denom1 = l1 * denom
        denom2 = l2 * denom
        if abs(denom1) < 1e-6:
            denom1 = 1e-6
        if abs(denom2) < 1e-6:
            denom2 = 1e-6

        domega1 = (
            -g * (2 * m1 + m2) * math.sin(self.theta1)
            - m2 * g * math.sin(self.theta1 - 2 * self.theta2)
            - 2
            * math.sin(delta)
            * m2
            * (self.omega2 * self.omega2 * l2 + self.omega1 * self.omega1 * l1 * math.cos(delta))
        ) / denom1

        domega2 = (
            2
            * math.sin(delta)
            * (
                self.omega1 * self.omega1 * l1 * (m1 + m2)
                + g * (m1 + m2) * math.cos(self.theta1)
                + self.omega2 * self.omega2 * l2 * m2 * math.cos(delta)
            )
        ) / denom2

        self.omega1 += domega1 * dt
        self.omega2 += domega2 * dt

        if self.damping > 0.0:
            damp = max(0.0, 1.0 - self.damping * dt)
            self.omega1 *= damp
            self.omega2 *= damp

        self.theta1 += self.omega1 * dt
        self.theta2 += self.omega2 * dt

        x1 = l1 * math.sin(self.theta1)
        y1 = -l1 * math.cos(self.theta1)
        x2 = x1 + l2 * math.sin(self.theta2)
        y2 = y1 - l2 * math.cos(self.theta2)

        v1_sq = (l1 * self.omega1) ** 2
        v2_sq = v1_sq + (l2 * self.omega2) ** 2 + 2 * l1 * l2 * self.omega1 * self.omega2 * math.cos(delta)
        kinetic = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
        potential = -(m1 + m2) * g * l1 * math.cos(self.theta1) - m2 * g * l2 * math.cos(self.theta2)
        energy = kinetic + potential

        return PendulumState(
            t=t,
            theta1=self.theta1,
            theta2=self.theta2,
            omega1=self.omega1,
            omega2=self.omega2,
            energy=energy,
            p1=(x1, y1),
            p2=(x2, y2),
        )


class DoublePendulumVizFlow(Flow[PendulumState, PendulumViz]):
    def __init__(self, trail_len: int, *, print_every: int, log_rerun: bool, namespace: str):
        super().__init__()
        self.trail_len = int(trail_len)
        self.print_every = int(print_every)
        self.log_rerun = bool(log_rerun)
        self.namespace = str(namespace)

    def init_config(self) -> dict:
        return {
            "trail_len": self.trail_len,
            "print_every": self.print_every,
            "log_rerun": self.log_rerun,
            "namespace": self.namespace,
        }

    def init(self) -> None:
        self.trail: list[list[float]] = []
        self.step_idx = 0

    def step(self, input_data: PendulumState) -> PendulumViz:
        if input_data.p1 is None or input_data.p2 is None or input_data.t is None:
            return PendulumViz()

        self.trail.append([input_data.p2[0], input_data.p2[1]])
        if len(self.trail) > self.trail_len:
            self.trail = self.trail[-self.trail_len :]

        points = [
            [0.0, 0.0],
            [input_data.p1[0], input_data.p1[1]],
            [input_data.p2[0], input_data.p2[1]],
        ]

        viz = PendulumViz(
            points=points,
            trail=list(self.trail),
            energy=input_data.energy,
            t=input_data.t,
        )

        self.step_idx += 1
        if self.print_every > 0 and self.step_idx % self.print_every == 0:
            energy = input_data.energy if input_data.energy is not None else 0.0
            print(
                f"[{self.namespace}] t={input_data.t:6.2f} "
                f"theta1={input_data.theta1:6.3f} theta2={input_data.theta2:6.3f} "
                f"energy={energy:8.3f}",
                flush=True,
            )

        if self.log_rerun:
            self._log_rerun(viz)

        return viz

    def _log_rerun(self, viz: PendulumViz) -> None:
        try:
            import rerun as rr
            from rerun.archetypes import Scalars
        except Exception:
            return

        if viz.t is not None:
            if hasattr(rr, "set_time_seconds"):
                rr.set_time_seconds("sim_time", viz.t)
            else:
                rr.set_time("sim_time", timestamp=viz.t)

        if not viz.points:
            return

        rr.log(f"{self.namespace}/links", rr.LineStrips2D([viz.points]))
        rr.log(f"{self.namespace}/masses", rr.Points2D(viz.points[1:], radii=0.06))
        rr.log(f"{self.namespace}/pivot", rr.Points2D([viz.points[0]], radii=0.03))
        if viz.trail:
            rr.log(f"{self.namespace}/trail", rr.LineStrips2D([viz.trail]))
        if viz.energy is not None:
            rr.log(f"{self.namespace}/energy", Scalars(viz.energy))


class NBodySim(Flow[SimTime, NBodyState]):
    FIGURE_EIGHT_POS = [
        (0.97000436, -0.24308753),
        (-0.97000436, 0.24308753),
        (0.0, 0.0),
    ]
    FIGURE_EIGHT_VEL = [
        (0.4662036850, 0.4323657300),
        (0.4662036850, 0.4323657300),
        (-0.93240737, -0.86473146),
    ]

    def __init__(self, gravity: float = 1.0, softening: float = 1e-3):
        super().__init__()
        self.gravity = float(gravity)
        self.softening = float(softening)
        self.masses = [1.0, 1.0, 1.0]

    def init_config(self) -> dict:
        return {"gravity": self.gravity, "softening": self.softening}

    def init(self) -> None:
        self.positions = [[x, y] for x, y in self.FIGURE_EIGHT_POS]
        self.velocities = [[vx, vy] for vx, vy in self.FIGURE_EIGHT_VEL]

    def step(self, input_data: SimTime) -> NBodyState:
        if input_data.t is None or input_data.dt is None:
            return NBodyState()

        dt = float(input_data.dt)
        t = float(input_data.t)

        accelerations = [[0.0, 0.0] for _ in self.positions]
        for i, pos_i in enumerate(self.positions):
            for j, pos_j in enumerate(self.positions):
                if i == j:
                    continue
                dx = pos_j[0] - pos_i[0]
                dy = pos_j[1] - pos_i[1]
                dist2 = dx * dx + dy * dy + self.softening * self.softening
                inv_dist = 1.0 / math.sqrt(dist2)
                inv_dist3 = inv_dist * inv_dist * inv_dist
                accelerations[i][0] += self.gravity * self.masses[j] * dx * inv_dist3
                accelerations[i][1] += self.gravity * self.masses[j] * dy * inv_dist3

        for i in range(len(self.positions)):
            self.velocities[i][0] += accelerations[i][0] * dt
            self.velocities[i][1] += accelerations[i][1] * dt
            self.positions[i][0] += self.velocities[i][0] * dt
            self.positions[i][1] += self.velocities[i][1] * dt

        kinetic = 0.0
        for i, vel in enumerate(self.velocities):
            kinetic += 0.5 * self.masses[i] * (vel[0] * vel[0] + vel[1] * vel[1])

        potential = 0.0
        for i in range(len(self.positions)):
            for j in range(i + 1, len(self.positions)):
                dx = self.positions[j][0] - self.positions[i][0]
                dy = self.positions[j][1] - self.positions[i][1]
                dist = math.sqrt(dx * dx + dy * dy + self.softening * self.softening)
                potential -= self.gravity * self.masses[i] * self.masses[j] / dist

        return NBodyState(
            t=t,
            positions=[list(p) for p in self.positions],
            velocities=[list(v) for v in self.velocities],
            energy=kinetic + potential,
        )


class NBodyVizFlow(Flow[NBodyState, NBodyViz]):
    def __init__(self, trail_len: int, *, print_every: int, log_rerun: bool, namespace: str):
        super().__init__()
        self.trail_len = int(trail_len)
        self.print_every = int(print_every)
        self.log_rerun = bool(log_rerun)
        self.namespace = str(namespace)

    def init_config(self) -> dict:
        return {
            "trail_len": self.trail_len,
            "print_every": self.print_every,
            "log_rerun": self.log_rerun,
            "namespace": self.namespace,
        }

    def init(self) -> None:
        self.trails: list[list[list[float]]] = []
        self.step_idx = 0

    def step(self, input_data: NBodyState) -> NBodyViz:
        if not input_data.positions or input_data.t is None:
            return NBodyViz()

        if not self.trails:
            self.trails = [[] for _ in input_data.positions]

        for idx, pos in enumerate(input_data.positions):
            self.trails[idx].append([pos[0], pos[1]])
            if len(self.trails[idx]) > self.trail_len:
                self.trails[idx] = self.trails[idx][-self.trail_len :]

        viz = NBodyViz(
            points=[list(p) for p in input_data.positions],
            trails=[list(t) for t in self.trails],
            energy=input_data.energy,
            t=input_data.t,
        )

        self.step_idx += 1
        if self.print_every > 0 and self.step_idx % self.print_every == 0:
            energy = input_data.energy if input_data.energy is not None else 0.0
            p0 = input_data.positions[0]
            print(
                f"[{self.namespace}] t={input_data.t:6.2f} "
                f"p0=({p0[0]:6.3f},{p0[1]:6.3f}) energy={energy:8.3f}",
                flush=True,
            )

        if self.log_rerun:
            self._log_rerun(viz)

        return viz

    def _log_rerun(self, viz: NBodyViz) -> None:
        try:
            import rerun as rr
            from rerun.archetypes import Scalars
        except Exception:
            return

        if viz.t is not None:
            if hasattr(rr, "set_time_seconds"):
                rr.set_time_seconds("sim_time", viz.t)
            else:
                rr.set_time("sim_time", timestamp=viz.t)

        if not viz.points:
            return

        rr.log(f"{self.namespace}/bodies", rr.Points2D(viz.points, radii=0.05))
        if viz.trails:
            for idx, trail in enumerate(viz.trails):
                if trail:
                    rr.log(f"{self.namespace}/trail/{idx}", rr.LineStrips2D([trail]))
        if viz.energy is not None:
            rr.log(f"{self.namespace}/energy", Scalars(viz.energy))


class NCoupledPendulumSim(Flow[SimTime, NCoupledPendulumState]):
    def __init__(
        self,
        n: int = 5,
        length: float = 1.0,
        spacing: float = 0.5,
        mass: float = 1.0,
        spring_k: float = 10.0,
        gravity: float = 9.81,
        damping: float = 0.02,
        init_mode: str = "wave",
        init_amplitude: float = 0.5,
    ):
        super().__init__()
        self.n = int(n)
        self.length = float(length)
        self.spacing = float(spacing)
        self.mass = float(mass)
        self.spring_k = float(spring_k)
        self.gravity = float(gravity)
        self.damping = float(damping)
        self.init_mode = str(init_mode)
        self.init_amplitude = float(init_amplitude)

    def init_config(self) -> dict:
        return {
            "n": self.n,
            "length": self.length,
            "spacing": self.spacing,
            "mass": self.mass,
            "spring_k": self.spring_k,
            "gravity": self.gravity,
            "damping": self.damping,
            "init_mode": self.init_mode,
            "init_amplitude": self.init_amplitude,
        }

    def init(self) -> None:
        self.thetas = [0.0] * self.n
        self.omegas = [0.0] * self.n

        if self.init_mode == "wave":
            for i in range(self.n):
                self.thetas[i] = self.init_amplitude * math.sin(2 * math.pi * i / self.n)
        elif self.init_mode == "impulse":
            self.thetas[0] = self.init_amplitude
        elif self.init_mode == "random":
            import random
            for i in range(self.n):
                self.thetas[i] = random.uniform(-0.1, 0.1)

        self.pivot_positions = [[i * self.spacing, 0.0] for i in range(self.n)]

    def step(self, input_data: SimTime) -> NCoupledPendulumState:
        if input_data.t is None or input_data.dt is None:
            return NCoupledPendulumState()

        dt = float(input_data.dt)
        t = float(input_data.t)

        positions = []
        for i in range(self.n):
            x = self.pivot_positions[i][0] + self.length * math.sin(self.thetas[i])
            y = self.pivot_positions[i][1] - self.length * math.cos(self.thetas[i])
            positions.append([x, y])

        alphas = []
        for i in range(self.n):
            gravity_torque = -(self.mass * self.gravity * self.length * math.sin(self.thetas[i]))

            spring_torque = 0.0
            if i > 0:
                dx = positions[i - 1][0] - positions[i][0]
                dy = positions[i - 1][1] - positions[i][1]
                spring_length = math.sqrt(dx * dx + dy * dy)
                rest_length = self.spacing
                spring_force = self.spring_k * (spring_length - rest_length)

                force_x = spring_force * dx / (spring_length + 1e-10)
                force_y = spring_force * dy / (spring_length + 1e-10)

                lever_x = positions[i][0] - self.pivot_positions[i][0]
                lever_y = positions[i][1] - self.pivot_positions[i][1]

                spring_torque += lever_x * force_y - lever_y * force_x

            if i < self.n - 1:
                dx = positions[i + 1][0] - positions[i][0]
                dy = positions[i + 1][1] - positions[i][1]
                spring_length = math.sqrt(dx * dx + dy * dy)
                rest_length = self.spacing
                spring_force = self.spring_k * (spring_length - rest_length)

                force_x = spring_force * dx / (spring_length + 1e-10)
                force_y = spring_force * dy / (spring_length + 1e-10)

                lever_x = positions[i][0] - self.pivot_positions[i][0]
                lever_y = positions[i][1] - self.pivot_positions[i][1]

                spring_torque += lever_x * force_y - lever_y * force_x

            total_torque = gravity_torque + spring_torque
            moment_of_inertia = self.mass * self.length * self.length
            alpha = total_torque / moment_of_inertia
            alphas.append(alpha)

        for i in range(self.n):
            self.omegas[i] += alphas[i] * dt

        if self.damping > 0.0:
            damp = max(0.0, 1.0 - self.damping * dt)
            for i in range(self.n):
                self.omegas[i] *= damp

        for i in range(self.n):
            self.thetas[i] += self.omegas[i] * dt

        kinetic = 0.0
        for i in range(self.n):
            v_sq = (self.length * self.omegas[i]) ** 2
            kinetic += 0.5 * self.mass * v_sq

        potential = 0.0
        for i in range(self.n):
            potential += self.mass * self.gravity * positions[i][1]

        spring_potential = 0.0
        for i in range(self.n - 1):
            dx = positions[i + 1][0] - positions[i][0]
            dy = positions[i + 1][1] - positions[i][1]
            spring_length = math.sqrt(dx * dx + dy * dy)
            rest_length = self.spacing
            spring_potential += 0.5 * self.spring_k * (spring_length - rest_length) ** 2

        energy = kinetic + potential + spring_potential

        return NCoupledPendulumState(
            t=t,
            n=self.n,
            thetas=list(self.thetas),
            omegas=list(self.omegas),
            energy=energy,
            positions=[list(p) for p in positions],
            pivot_positions=[list(p) for p in self.pivot_positions],
        )


class NCoupledPendulumVizFlow(Flow[NCoupledPendulumState, NCoupledPendulumViz]):
    def __init__(self, trail_len: int, *, print_every: int, log_rerun: bool, namespace: str):
        super().__init__()
        self.trail_len = int(trail_len)
        self.print_every = int(print_every)
        self.log_rerun = bool(log_rerun)
        self.namespace = str(namespace)

    def init_config(self) -> dict:
        return {
            "trail_len": self.trail_len,
            "print_every": self.print_every,
            "log_rerun": self.log_rerun,
            "namespace": self.namespace,
        }

    def init(self) -> None:
        self.trails: list[list[list[float]]] = []
        self.step_idx = 0

    def step(self, input_data: NCoupledPendulumState) -> NCoupledPendulumViz:
        if (
            not input_data.positions
            or not input_data.pivot_positions
            or input_data.t is None
            or input_data.n is None
        ):
            return NCoupledPendulumViz()

        if not self.trails:
            self.trails = [[] for _ in range(input_data.n)]

        for idx, pos in enumerate(input_data.positions):
            self.trails[idx].append([pos[0], pos[1]])
            if len(self.trails[idx]) > self.trail_len:
                self.trails[idx] = self.trails[idx][-self.trail_len :]

        rod_segments = []
        for i in range(input_data.n):
            rod_segments.append([input_data.pivot_positions[i], input_data.positions[i]])

        spring_segments = []
        for i in range(input_data.n - 1):
            spring_segments.append([input_data.positions[i], input_data.positions[i + 1]])

        viz = NCoupledPendulumViz(
            pivot_points=[list(p) for p in input_data.pivot_positions],
            bob_points=[list(p) for p in input_data.positions],
            rod_segments=rod_segments,
            spring_segments=spring_segments,
            trails=[list(t) for t in self.trails],
            energy=input_data.energy,
            t=input_data.t,
        )

        self.step_idx += 1
        if self.print_every > 0 and self.step_idx % self.print_every == 0:
            energy = input_data.energy if input_data.energy is not None else 0.0
            theta0 = input_data.thetas[0] if input_data.thetas else 0.0
            print(
                f"[{self.namespace}] t={input_data.t:6.2f} "
                f"theta0={theta0:6.3f} energy={energy:8.3f}",
                flush=True,
            )

        if self.log_rerun:
            self._log_rerun(viz, input_data.n)

        return viz

    def _log_rerun(self, viz: NCoupledPendulumViz, n: int) -> None:
        try:
            import rerun as rr
            from rerun.archetypes import Scalars
        except Exception:
            return

        if viz.t is not None:
            if hasattr(rr, "set_time_seconds"):
                rr.set_time_seconds("sim_time", viz.t)
            else:
                rr.set_time("sim_time", timestamp=viz.t)

        if not viz.bob_points:
            return

        def hue_to_rgb(hue: float) -> list[int]:
            h = hue / 60.0
            x = 1.0 - abs((h % 2) - 1.0)
            if h < 1:
                r, g, b = 1.0, x, 0.0
            elif h < 2:
                r, g, b = x, 1.0, 0.0
            elif h < 3:
                r, g, b = 0.0, 1.0, x
            elif h < 4:
                r, g, b = 0.0, x, 1.0
            elif h < 5:
                r, g, b = x, 0.0, 1.0
            else:
                r, g, b = 1.0, 0.0, x
            return [int(r * 255), int(g * 255), int(b * 255)]

        colors = []
        for i in range(n):
            hue = 240.0 - (240.0 * i / max(n - 1, 1))
            colors.append(hue_to_rgb(hue))

        if viz.pivot_points:
            rr.log(f"{self.namespace}/pivots", rr.Points2D(viz.pivot_points, radii=0.03))

        if viz.rod_segments:
            for i, segment in enumerate(viz.rod_segments):
                rr.log(
                    f"{self.namespace}/rod/{i}",
                    rr.LineStrips2D([segment], colors=[colors[i]]),
                )

        if viz.bob_points:
            for i, point in enumerate(viz.bob_points):
                rr.log(
                    f"{self.namespace}/bob/{i}",
                    rr.Points2D([point], radii=0.06, colors=[colors[i]]),
                )

        if viz.spring_segments:
            for i, segment in enumerate(viz.spring_segments):
                rr.log(
                    f"{self.namespace}/spring/{i}",
                    rr.LineStrips2D([segment], colors=[[128, 128, 128]]),
                )

        if viz.trails:
            for idx, trail in enumerate(viz.trails):
                if trail:
                    rr.log(
                        f"{self.namespace}/trail/{idx}",
                        rr.LineStrips2D([trail], colors=[colors[idx]]),
                    )

        if viz.energy is not None:
            rr.log(f"{self.namespace}/energy", Scalars(viz.energy))


class NCoupledSpringSim(Flow[SimTime, NCoupledSpringState]):
    def __init__(
        self,
        n: int = 5,
        mass: float = 1.0,
        spring_k: float = 10.0,
        rest_length: float = 1.0,
        damping: float = 0.02,
        gravity: float = 0.0,
        init_mode: str = "wave",
        init_amplitude: float = 0.5,
    ):
        super().__init__()
        self.n = int(n)
        self.mass = float(mass)
        self.spring_k = float(spring_k)
        self.rest_length = float(rest_length)
        self.damping = float(damping)
        self.gravity = float(gravity)
        self.init_mode = str(init_mode)
        self.init_amplitude = float(init_amplitude)

    def init_config(self) -> dict:
        return {
            "n": self.n,
            "mass": self.mass,
            "spring_k": self.spring_k,
            "rest_length": self.rest_length,
            "damping": self.damping,
            "gravity": self.gravity,
            "init_mode": self.init_mode,
            "init_amplitude": self.init_amplitude,
        }

    def init(self) -> None:
        self.positions = [[i * self.rest_length, 0.0] for i in range(self.n)]
        self.velocities = [[0.0, 0.0] for _ in range(self.n)]

        if self.init_mode == "wave":
            for i in range(self.n):
                self.positions[i][0] += self.init_amplitude * math.sin(2 * math.pi * i / self.n)
        elif self.init_mode == "impulse":
            self.positions[0][0] += self.init_amplitude
        elif self.init_mode == "compress":
            for i in range(self.n):
                self.positions[i][0] = i * (self.rest_length - self.init_amplitude / self.n)
        elif self.init_mode == "random":
            import random
            for i in range(self.n):
                self.positions[i][0] += random.uniform(-0.1, 0.1)

    def step(self, input_data: SimTime) -> NCoupledSpringState:
        if input_data.t is None or input_data.dt is None:
            return NCoupledSpringState()

        dt = float(input_data.dt)
        t = float(input_data.t)

        accelerations = [[0.0, 0.0] for _ in range(self.n)]

        for i in range(self.n):
            if i > 0:
                dx = self.positions[i][0] - self.positions[i - 1][0]
                dy = self.positions[i][1] - self.positions[i - 1][1]
                dist = math.sqrt(dx * dx + dy * dy)
                spring_force = self.spring_k * (dist - self.rest_length)

                if dist > 1e-10:
                    force_x = -spring_force * dx / dist
                    force_y = -spring_force * dy / dist
                    accelerations[i][0] += force_x / self.mass
                    accelerations[i][1] += force_y / self.mass

            if i < self.n - 1:
                dx = self.positions[i][0] - self.positions[i + 1][0]
                dy = self.positions[i][1] - self.positions[i + 1][1]
                dist = math.sqrt(dx * dx + dy * dy)
                spring_force = self.spring_k * (dist - self.rest_length)

                if dist > 1e-10:
                    force_x = -spring_force * dx / dist
                    force_y = -spring_force * dy / dist
                    accelerations[i][0] += force_x / self.mass
                    accelerations[i][1] += force_y / self.mass

            accelerations[i][1] -= self.gravity

        for i in range(self.n):
            self.velocities[i][0] += accelerations[i][0] * dt
            self.velocities[i][1] += accelerations[i][1] * dt

        if self.damping > 0.0:
            damp = max(0.0, 1.0 - self.damping * dt)
            for i in range(self.n):
                self.velocities[i][0] *= damp
                self.velocities[i][1] *= damp

        for i in range(self.n):
            self.positions[i][0] += self.velocities[i][0] * dt
            self.positions[i][1] += self.velocities[i][1] * dt

        kinetic = 0.0
        for i in range(self.n):
            v_sq = self.velocities[i][0] ** 2 + self.velocities[i][1] ** 2
            kinetic += 0.5 * self.mass * v_sq

        spring_potential = 0.0
        for i in range(self.n - 1):
            dx = self.positions[i + 1][0] - self.positions[i][0]
            dy = self.positions[i + 1][1] - self.positions[i][1]
            dist = math.sqrt(dx * dx + dy * dy)
            spring_potential += 0.5 * self.spring_k * (dist - self.rest_length) ** 2

        gravitational_potential = 0.0
        if self.gravity > 0.0:
            for i in range(self.n):
                gravitational_potential += self.mass * self.gravity * self.positions[i][1]

        energy = kinetic + spring_potential + gravitational_potential

        return NCoupledSpringState(
            t=t,
            n=self.n,
            positions=[list(p) for p in self.positions],
            velocities=[list(v) for v in self.velocities],
            energy=energy,
        )


class NCoupledSpringVizFlow(Flow[NCoupledSpringState, NCoupledSpringViz]):
    def __init__(self, trail_len: int, *, print_every: int, log_rerun: bool, namespace: str):
        super().__init__()
        self.trail_len = int(trail_len)
        self.print_every = int(print_every)
        self.log_rerun = bool(log_rerun)
        self.namespace = str(namespace)

    def init_config(self) -> dict:
        return {
            "trail_len": self.trail_len,
            "print_every": self.print_every,
            "log_rerun": self.log_rerun,
            "namespace": self.namespace,
        }

    def init(self) -> None:
        self.trails: list[list[list[float]]] = []
        self.step_idx = 0

    def step(self, input_data: NCoupledSpringState) -> NCoupledSpringViz:
        if not input_data.positions or input_data.t is None or input_data.n is None:
            return NCoupledSpringViz()

        if not self.trails:
            self.trails = [[] for _ in range(input_data.n)]

        for idx, pos in enumerate(input_data.positions):
            self.trails[idx].append([pos[0], pos[1]])
            if len(self.trails[idx]) > self.trail_len:
                self.trails[idx] = self.trails[idx][-self.trail_len :]

        spring_segments = []
        for i in range(input_data.n - 1):
            spring_segments.append([input_data.positions[i], input_data.positions[i + 1]])

        viz = NCoupledSpringViz(
            mass_points=[list(p) for p in input_data.positions],
            spring_segments=spring_segments,
            trails=[list(t) for t in self.trails],
            energy=input_data.energy,
            t=input_data.t,
        )

        self.step_idx += 1
        if self.print_every > 0 and self.step_idx % self.print_every == 0:
            energy = input_data.energy if input_data.energy is not None else 0.0
            pos0 = input_data.positions[0]
            print(
                f"[{self.namespace}] t={input_data.t:6.2f} "
                f"pos0=({pos0[0]:6.3f},{pos0[1]:6.3f}) energy={energy:8.3f}",
                flush=True,
            )

        if self.log_rerun:
            self._log_rerun(viz, input_data.n)

        return viz

    def _log_rerun(self, viz: NCoupledSpringViz, n: int) -> None:
        try:
            import rerun as rr
            from rerun.archetypes import Scalars
        except Exception:
            return

        if viz.t is not None:
            if hasattr(rr, "set_time_seconds"):
                rr.set_time_seconds("sim_time", viz.t)
            else:
                rr.set_time("sim_time", timestamp=viz.t)

        if not viz.mass_points:
            return

        def hue_to_rgb(hue: float) -> list[int]:
            h = hue / 60.0
            x = 1.0 - abs((h % 2) - 1.0)
            if h < 1:
                r, g, b = 1.0, x, 0.0
            elif h < 2:
                r, g, b = x, 1.0, 0.0
            elif h < 3:
                r, g, b = 0.0, 1.0, x
            elif h < 4:
                r, g, b = 0.0, x, 1.0
            elif h < 5:
                r, g, b = x, 0.0, 1.0
            else:
                r, g, b = 1.0, 0.0, x
            return [int(r * 255), int(g * 255), int(b * 255)]

        colors = []
        for i in range(n):
            hue = 240.0 - (240.0 * i / max(n - 1, 1))
            colors.append(hue_to_rgb(hue))

        if viz.mass_points:
            for i, point in enumerate(viz.mass_points):
                rr.log(
                    f"{self.namespace}/mass/{i}",
                    rr.Points2D([point], radii=0.08, colors=[colors[i]]),
                )

        if viz.spring_segments:
            for i, segment in enumerate(viz.spring_segments):
                rr.log(
                    f"{self.namespace}/spring/{i}",
                    rr.LineStrips2D([segment], colors=[[128, 128, 128]]),
                )

        if viz.trails:
            for idx, trail in enumerate(viz.trails):
                if trail:
                    rr.log(
                        f"{self.namespace}/trail/{idx}",
                        rr.LineStrips2D([trail], colors=[colors[idx]]),
                    )

        if viz.energy is not None:
            rr.log(f"{self.namespace}/energy", Scalars(viz.energy))

"""
Shared flows for real-time hybrid system demos.
"""

from __future__ import annotations

import time
from typing import Optional

from retriever.flow import Flow, io


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sleep_ms(work_ms: float) -> None:
    if work_ms <= 0.0:
        return
    time.sleep(work_ms / 1000.0)


@io
class TimeTick:
    t_sim: Optional[float] = None
    t_wall: Optional[float] = None
    dt: Optional[float] = None


@io
class DeadlineInput:
    t_sim: Optional[float] = None
    t_wall: Optional[float] = None


@io
class DeadlineStatus:
    t_sim: Optional[float] = None
    t_wall: Optional[float] = None
    dt: Optional[float] = None
    deadline_s: Optional[float] = None
    missed: Optional[bool] = None
    miss_count: Optional[int] = None
    label: Optional[str] = None


class SimClock(Flow[None, TimeTick]):
    def __init__(self, dt: float, use_wall: bool = True):
        super().__init__()
        self.dt = float(dt)
        self.use_wall = bool(use_wall)

    def init_config(self) -> dict:
        return {"dt": self.dt, "use_wall": self.use_wall}

    def init(self) -> None:
        self.t_sim = 0.0
        self._last_wall = time.perf_counter()

    def step(self, _input: None) -> TimeTick:
        if self.use_wall:
            now = time.perf_counter()
            dt = max(now - self._last_wall, 1e-6)
            self._last_wall = now
        else:
            dt = self.dt
            self._last_wall = time.perf_counter()

        self.t_sim += dt
        return TimeTick(t_sim=self.t_sim, t_wall=self._last_wall, dt=dt)


class DeadlineMonitor(Flow[DeadlineInput, DeadlineStatus]):
    def __init__(
        self,
        deadline_s: float,
        *,
        label: str,
        print_every: int,
        log_rerun: bool,
        namespace: str,
    ):
        super().__init__()
        self.deadline_s = float(deadline_s)
        self.label = str(label)
        self.print_every = int(print_every)
        self.log_rerun = bool(log_rerun)
        self.namespace = str(namespace)

    def init_config(self) -> dict:
        return {
            "deadline_s": self.deadline_s,
            "label": self.label,
            "print_every": self.print_every,
            "log_rerun": self.log_rerun,
            "namespace": self.namespace,
        }

    def init(self) -> None:
        self._last_wall = None
        self._miss_count = 0
        self._step_idx = 0

    def step(self, input_data: DeadlineInput) -> DeadlineStatus:
        if input_data.t_wall is None:
            return DeadlineStatus()

        t_wall = float(input_data.t_wall)
        t_sim = float(input_data.t_sim) if input_data.t_sim is not None else None
        label = self.label

        if self._last_wall is None:
            self._last_wall = t_wall
            return DeadlineStatus(
                t_sim=t_sim,
                t_wall=t_wall,
                dt=0.0,
                deadline_s=self.deadline_s,
                missed=False,
                miss_count=self._miss_count,
                label=label,
            )

        dt = max(t_wall - self._last_wall, 0.0)
        missed = dt > self.deadline_s
        if missed:
            self._miss_count += 1

        self._step_idx += 1
        if self.print_every > 0 and self._step_idx % self.print_every == 0:
            status = "MISS" if missed else "ok"
            print(
                f"[{self.namespace}] {label} dt={dt*1000.0:6.1f}ms "
                f"deadline={self.deadline_s*1000.0:6.1f}ms {status} "
                f"misses={self._miss_count}",
                flush=True,
            )

        if self.log_rerun:
            try:
                import rerun as rr
                from rerun.archetypes import Scalars
            except Exception:
                rr = None
            if rr is not None:
                if t_sim is not None:
                    if hasattr(rr, "set_time_seconds"):
                        rr.set_time_seconds("sim_time", t_sim)
                    else:
                        rr.set_time("sim_time", timestamp=t_sim)
                base = f"{self.namespace}/{label}"
                rr.log(f"{base}/dt_ms", Scalars(dt * 1000.0))
                rr.log(f"{base}/deadline_ms", Scalars(self.deadline_s * 1000.0))
                rr.log(f"{base}/missed", Scalars(1.0 if missed else 0.0))
                rr.log(f"{base}/miss_count", Scalars(float(self._miss_count)))
                if missed:
                    rr.log(f"{base}/event", rr.TextLog("deadline_miss"))

        self._last_wall = t_wall
        return DeadlineStatus(
            t_sim=t_sim,
            t_wall=t_wall,
            dt=dt,
            deadline_s=self.deadline_s,
            missed=missed,
            miss_count=self._miss_count,
            label=label,
        )

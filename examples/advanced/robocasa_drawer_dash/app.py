"""Drawer dash: a Panda grasps a drawer handle and pulls, as Retriever Flows.

The drawer's slide joint carries no actuator. It is passive and damped, so the
drawer can only move if the gripper has hold of its handle — which makes the
grasp, not a scripted joint target, the thing the run actually proves.

Run the mock-safe smoke test (no simulator, no assets, no GPU):

  pixi run demo-drawer-dash-mock

Run against MuJoCo after installing the optional simulator dependencies and
building the scene (see this example's README for the RoboCasa asset packs):

  pixi run python -m pip install -e ".[drawer_dash]"
  pixi run demo-drawer-dash-scene
  pixi run demo-drawer-dash

The mock lane reproduces the choreography's timeline and the drawer travel it
commands, so the phase sequence, the grasp window and the open/shut thresholds
are all exercised without MuJoCo present. It does not reproduce contact
physics: only the `mujoco` lane can show that the grasp is what moves the
drawer, and `verify.py` is what asserts it.
"""

from __future__ import annotations

import argparse
from typing import Any

from retriever.flow import Flow, Latest, Pipeline, Rate, Trigger, io

from examples.advanced.robocasa_drawer_dash.plan import (
    GRASPING,
    OPENED_MIN,
    PHASES,
    SHUT_MAX,
    STROKE,
    TOTAL_SECONDS,
    phase_at,
)


@io
class DrawerDashAction:
    """One tick of the routine: which phase, and what it commands."""

    phase: str | None = None
    phase_index: int | None = None
    blend: float | None = None
    grip: float | None = None
    stroke: float | None = None
    grasping: bool | None = None
    elapsed: float | None = None


@io
class DrawerDashState:
    step: int | None = None
    source: str | None = None
    phase: str | None = None
    drawer_open: float | None = None
    peak_open: float | None = None
    grasped: bool | None = None
    progress: float | None = None
    done: bool | None = None
    success: bool | None = None


class ChoreographyPolicy(Flow[DrawerDashState, DrawerDashAction]):
    """Walks the phase schedule on its own clock and commands each tick.

    The schedule lives in `plan.py` and is shared with the MuJoCo lane, so the
    mock and the simulator run the same routine rather than two lookalikes.
    """

    def __init__(self, *, hz: float) -> None:
        super().__init__()
        self.hz = max(1e-6, float(hz))

    def reset(self) -> None:
        self.elapsed = 0.0

    def step(self, state: DrawerDashState | None) -> DrawerDashAction:
        index, phase, blend = phase_at(self.elapsed)
        start, end = phase.stroke
        stroke = start + (end - start) * blend if phase.mode == "carry" else 0.0
        action = DrawerDashAction(
            phase=phase.label,
            phase_index=index,
            blend=blend,
            grip=phase.grip,
            stroke=stroke,
            grasping=phase.label in GRASPING,
            elapsed=self.elapsed,
        )
        self.elapsed += 1.0 / self.hz
        return action


class DrawerDashSimulator(Flow[DrawerDashAction, DrawerDashState]):
    """Runs the commanded routine against MuJoCo, or a deterministic mock."""

    def __init__(self, *, mode: str, scene: str | None, camera: str, hz: float) -> None:
        super().__init__()
        self.mode = mode
        self.scene = scene
        self.camera = camera
        self.hz = max(1e-6, float(hz))
        self.latest: DrawerDashState | None = None

    def reset(self) -> None:
        self.step_idx = 0
        self.peak_open = 0.0
        self.drawer_open = 0.0
        self.grasped = False
        self._rig: Any | None = None

        if self.mode == "mujoco":
            self._rig = self._open_rig()

    def _open_rig(self) -> Any:
        from pathlib import Path

        try:
            from examples.advanced.robocasa_drawer_dash.verify import Rig
        except ImportError as exc:  # pragma: no cover - exercised without mujoco
            raise RuntimeError(
                "MuJoCo is not installed. Install the optional simulator "
                'dependencies with `pixi run python -m pip install -e ".[drawer_dash]"`, '
                "or run `pixi run demo-drawer-dash-mock` for the mock-safe smoke test."
            ) from exc

        scene = Path(self.scene) if self.scene else Path(__file__).resolve().parent / "scene.xml"
        if not scene.exists():
            raise RuntimeError(
                f"scene file {scene.name} has not been generated. Build it with "
                "`pixi run demo-drawer-dash-scene`, which needs RoboCasa and its "
                "fixture and object asset packs. See this example's README."
            )
        return Rig(scene, camera=self.camera)

    def step(self, action: DrawerDashAction | None) -> DrawerDashState:
        self.step_idx += 1
        if self.mode == "mujoco":
            state = self._step_mujoco(action)
        else:
            state = self._step_mock(action)
        self.latest = state
        return state

    def _finish(self, *, source: str, action: DrawerDashAction | None) -> DrawerDashState:
        elapsed = 0.0 if action is None or action.elapsed is None else action.elapsed
        progress = min(1.0, elapsed / TOTAL_SECONDS)
        done = progress >= 1.0
        phase = None if action is None else action.phase
        # The same claim `verify.py` asserts: it came far enough out, and it
        # went back shut. Only true once the routine has actually finished.
        success = done and self.peak_open >= OPENED_MIN and self.drawer_open <= SHUT_MAX
        return DrawerDashState(
            step=self.step_idx,
            source=source,
            phase=phase,
            drawer_open=self.drawer_open,
            peak_open=self.peak_open,
            grasped=self.grasped,
            progress=progress,
            done=done,
            success=success,
        )

    def _step_mock(self, action: DrawerDashAction | None) -> DrawerDashState:
        # No contact model: the mock grants the grasp during the phases that
        # ask for it, and lets the drawer follow the commanded stroke. What it
        # does reproduce is the timeline, the travel and the thresholds.
        if action is not None:
            self.grasped = bool(action.grasping)
            commanded = 0.0 if action.stroke is None else float(action.stroke)
            if self.grasped:
                self.drawer_open = min(STROKE, max(0.0, commanded))
            self.peak_open = max(self.peak_open, self.drawer_open)
        return self._finish(source="mock", action=action)

    def _step_mujoco(self, action: DrawerDashAction | None) -> DrawerDashState:
        if self._rig is None:
            raise RuntimeError("the MuJoCo rig was not initialized")
        if action is not None:
            self._rig.apply(action, seconds=1.0 / self.hz)
        self.drawer_open = float(self._rig.plan.drawer_open)
        self.grasped = bool(self._rig.plan.gripping())
        self.peak_open = max(self.peak_open, self.drawer_open)
        return self._finish(source="mujoco", action=action)

    def finalize(self) -> None:
        if self._rig is not None:
            self._rig.close()


class DrawerDashPrinter(Flow[DrawerDashState, None]):
    def __init__(self, *, print_every: int) -> None:
        super().__init__()
        self.print_every = max(1, int(print_every))

    def step(self, state: DrawerDashState) -> None:
        if state.step is None or state.step % self.print_every != 0:
            return None
        print(
            f"[{state.source} step={state.step:04d}] "
            f"phase={_pad(state.phase)} "
            f"drawer={_fmt(state.drawer_open)}m "
            f"grasped={bool(state.grasped)} "
            f"progress={_pct(state.progress)} success={bool(state.success)}"
        )
        return None


def _fmt(value: float | None) -> str:
    return "None" if value is None else f"{value:.3f}"


def _pct(value: float | None) -> str:
    return "None" if value is None else f"{value * 100:.1f}%"


def _pad(label: str | None) -> str:
    return f"{label or 'none':<16}"


def build_pipeline(args: argparse.Namespace) -> tuple[Pipeline, DrawerDashSimulator]:
    simulator = DrawerDashSimulator(
        mode=args.mode,
        scene=getattr(args, "scene", None),
        camera=getattr(args, "camera", "threequarter"),
        hz=args.hz,
    )
    pipe = Pipeline("robocasa_drawer_dash")
    with pipe:
        sim = simulator @ Rate(hz=args.hz)
        policy = ChoreographyPolicy(hz=args.hz) @ Rate(hz=args.hz)
        printer = DrawerDashPrinter(print_every=args.print_every) @ Trigger("step")
        pipe.connect(policy, sim, sync=Latest())
        pipe.connect(sim, policy, sync=Latest())
        pipe.connect(sim, printer, sync=Latest())
    return pipe, simulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retriever + RoboCasa drawer-dash demo: grasp a handle and work the drawer."
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "mujoco"],
        default="mock",
        help="mock needs nothing installed; mujoco needs the simulator and a generated scene.xml.",
    )
    parser.add_argument("--scene", default=None, help="Path to a generated scene.xml.")
    parser.add_argument("--camera", default="threequarter", help="Scene camera for the mujoco lane.")
    parser.add_argument("--steps", type=int, default=38, help="Pipeline steps to run.")
    parser.add_argument("--dt", type=float, default=0.5, help="Seconds of wall clock per step.")
    parser.add_argument("--hz", type=float, default=2.0, help="Routine clock rate.")
    parser.add_argument("--print-every", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe, simulator = build_pipeline(args)
    try:
        for _ in range(args.steps):
            pipe.step(dt=args.dt)
    finally:
        pipe.close_stepper()

    final = simulator.latest
    if final is not None:
        print(
            f"\nroutine complete: peak drawer travel {_fmt(final.peak_open)} m "
            f"(needs >= {OPENED_MIN}), shut to {_fmt(final.drawer_open)} m "
            f"(needs <= {SHUT_MAX}), {len(PHASES)} phases over {TOTAL_SECONDS:.1f} s "
            f"-> success={bool(final.success)}"
        )


if __name__ == "__main__":
    main()

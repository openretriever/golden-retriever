"""A Panda opens a drawer and seasons a plated meal, as Retriever Flows.

The drawer's slide joint carries no actuator, and neither does the pepper
shaker standing in it. Both are passive — a damped slide and a free body — so
the drawer can only move while the gripper has hold of its handle, and the
shaker can only reach the plate if the gripper carries it there. That makes
the grasps, not a scripted joint target, the thing the run actually proves.

Run the mock-safe smoke test (no simulator, no assets, no GPU):

  pixi run demo-drawer-mock

Run against MuJoCo after installing the optional simulator dependencies and
the RoboCasa asset packs (see this example's README):

  pixi run python -m pip install -e ".[robocasa_drawer]"
  pixi run demo-drawer-assets
  pixi run demo-drawer-flow

`pixi run demo-drawer` is the browser demo rather than this lane; see
`viewer.py`.

The mock lane reproduces the choreography's timeline, the drawer travel it
commands and the roll it commands over the plate, so the phase sequence, both
grasp windows and the pass thresholds are all exercised without MuJoCo
present. It does not reproduce contact physics: only the `mujoco` lane can
show that the grasp is what moves the drawer, and `verify.py` is what asserts
it.
"""

from __future__ import annotations

import argparse
import math
from typing import Any

from retriever.flow import Flow, Latest, Pipeline, Rate, Trigger, io

from examples.advanced.robocasa_drawer.plan import (
    CARRYING,
    GRASPING,
    OPENED_MIN,
    PHASES,
    SHAKER_RELEASED,
    SHUT_MAX,
    STROKE,
    TIP_MIN,
    TOTAL_SECONDS,
    phase_at,
    tip_at,
)

# Long enough for the mock to walk the whole routine at the default 0.5 s tick,
# with a few ticks in hand so the final phase is reported finished.
MOCK_STEPS = int(TOTAL_SECONDS / 0.5) + 6


@io
class DrawerAction:
    """One tick of the routine: which phase, and what it commands."""

    phase: str | None = None
    phase_index: int | None = None
    blend: float | None = None
    grip: float | None = None
    stroke: float | None = None
    tip: float | None = None
    grasping: bool | None = None
    carrying: bool | None = None
    elapsed: float | None = None


@io
class DrawerState:
    step: int | None = None
    source: str | None = None
    phase: str | None = None
    drawer_open: float | None = None
    peak_open: float | None = None
    grasped: bool | None = None
    carrying: bool | None = None
    tip: float | None = None
    seasoned: bool | None = None
    stowed: bool | None = None
    progress: float | None = None
    done: bool | None = None
    success: bool | None = None


class ChoreographyPolicy(Flow[DrawerState, DrawerAction]):
    """Walks the phase schedule on its own clock and commands each tick.

    The schedule lives in `plan.py` and is shared with the MuJoCo lane, so the
    mock and the simulator run the same routine rather than two lookalikes.
    """

    def __init__(self, *, hz: float) -> None:
        super().__init__()
        self.hz = max(1e-6, float(hz))

    def reset(self) -> None:
        self.elapsed = 0.0
        # Where the plan has asked the drawer to be. A carry phase eases from
        # wherever the previous one left it to its own `open_to`, so this is
        # what makes "pull to 0.22" and "push back to shut" one continuous
        # commanded travel rather than two absolute jumps.
        self.commanded = 0.0
        self.carry_index: int | None = None
        self.carry_from = 0.0

    def step(self, state: DrawerState | None) -> DrawerAction:
        index, phase, blend = phase_at(self.elapsed)
        if phase.mode == "carry":
            if index != self.carry_index:
                self.carry_index = index
                self.carry_from = self.commanded
            self.commanded = (self.carry_from
                              + (phase.open_to - self.carry_from) * blend)
        action = DrawerAction(
            phase=phase.label,
            phase_index=index,
            blend=blend,
            grip=phase.grip,
            stroke=self.commanded,
            tip=tip_at(phase, blend),
            grasping=phase.label in GRASPING,
            carrying=phase.label in CARRYING,
            elapsed=self.elapsed,
        )
        self.elapsed += 1.0 / self.hz
        return action


class DrawerSimulator(Flow[DrawerAction, DrawerState]):
    """Runs the commanded routine against MuJoCo, or a deterministic mock."""

    def __init__(self, *, mode: str, scene: str | None, camera: str, hz: float) -> None:
        super().__init__()
        self.mode = mode
        self.scene = scene
        self.camera = camera
        self.hz = max(1e-6, float(hz))
        self.latest: DrawerState | None = None

    def reset(self) -> None:
        self.step_idx = 0
        self.peak_open = 0.0
        self.drawer_open = 0.0
        self.grasped = False
        self.carrying = False
        self.tip = 0.0
        self.seasoned = False
        self.stowed = False
        self._rig: Any | None = None

        if self.mode == "mujoco":
            self._rig = self._open_rig()

    def _open_rig(self) -> Any:
        from pathlib import Path

        try:
            from examples.advanced.robocasa_drawer.verify import Rig
        except ImportError as exc:  # pragma: no cover - exercised without mujoco
            raise RuntimeError(
                "MuJoCo is not installed. Install the optional simulator "
                'dependencies with `pixi run python -m pip install -e '
                '".[robocasa_drawer]"`, or run `pixi run demo-drawer-mock` for '
                "the mock-safe smoke test."
            ) from exc

        from examples.advanced.robocasa_drawer.scene import ensure_scene

        # Built on first use rather than demanded up front, so a fresh clone
        # with the asset packs installed can run this lane straight away.
        return Rig(ensure_scene(Path(self.scene) if self.scene else None),
                   camera=self.camera)

    def step(self, action: DrawerAction | None) -> DrawerState:
        self.step_idx += 1
        if self.mode == "mujoco":
            state = self._step_mujoco(action)
        else:
            state = self._step_mock(action)
        self.latest = state
        return state

    def _finish(self, *, source: str, action: DrawerAction | None) -> DrawerState:
        elapsed = 0.0 if action is None or action.elapsed is None else action.elapsed
        progress = min(1.0, elapsed / TOTAL_SECONDS)
        done = progress >= 1.0
        phase = None if action is None else action.phase
        # The same claims `verify.py` asserts: the drawer came far enough out
        # and went back shut, the shaker was tipped cap-down over the food, and
        # it was put back in the drawer. Only true once the routine finished.
        success = (done and self.peak_open >= OPENED_MIN
                   and self.drawer_open <= SHUT_MAX
                   and self.seasoned and self.stowed)
        return DrawerState(
            step=self.step_idx,
            source=source,
            phase=phase,
            drawer_open=self.drawer_open,
            peak_open=self.peak_open,
            grasped=self.grasped,
            carrying=self.carrying,
            tip=self.tip,
            seasoned=self.seasoned,
            stowed=self.stowed,
            progress=progress,
            done=done,
            success=success,
        )

    def _step_mock(self, action: DrawerAction | None) -> DrawerState:
        # No contact model: the mock grants each grasp during the phases that
        # ask for it, and lets the drawer and the shaker follow what is
        # commanded. What it does reproduce is the timeline, the travel, the
        # roll and the thresholds.
        if action is not None:
            self.grasped = bool(action.grasping)
            self.carrying = bool(action.carrying)
            self.tip = 0.0 if action.tip is None else float(action.tip)
            commanded = 0.0 if action.stroke is None else float(action.stroke)
            if self.grasped:
                self.drawer_open = min(STROKE, max(0.0, commanded))
            self.peak_open = max(self.peak_open, self.drawer_open)
            # Seasoning happens only while the shaker is actually held: an
            # ungrasped shaker cannot be tipped over anything.
            if self.carrying and self.tip >= TIP_MIN:
                self.seasoned = True
            # It is back in the drawer from the moment the fingers open on it.
            if action.phase == SHAKER_RELEASED:
                self.stowed = True
        return self._finish(source="mock", action=action)

    def _step_mujoco(self, action: DrawerAction | None) -> DrawerState:
        if self._rig is None:
            raise RuntimeError("the MuJoCo rig was not initialized")
        if action is not None:
            self._rig.apply(action, seconds=1.0 / self.hz)
        self.drawer_open = float(self._rig.routine.drawer_open)
        self.grasped = bool(self._rig.routine.gripping())
        self.carrying = bool(self._rig.routine.holding())
        self.tip = float(self._rig.routine.tip())
        if self.carrying and self.tip >= TIP_MIN:
            self.seasoned = True
        if action is not None and action.phase == SHAKER_RELEASED:
            self.stowed = bool(self._rig.shaker_stowed())
        self.peak_open = max(self.peak_open, self.drawer_open)
        return self._finish(source="mujoco", action=action)

    def finalize(self) -> None:
        if self._rig is not None:
            self._rig.close()


class DrawerPrinter(Flow[DrawerState, None]):
    def __init__(self, *, print_every: int) -> None:
        super().__init__()
        self.print_every = max(1, int(print_every))

    def step(self, state: DrawerState) -> None:
        if state.step is None or state.step % self.print_every != 0:
            return None
        holding = "handle" if state.grasped else ("shaker" if state.carrying
                                                  else "nothing")
        print(
            f"[{state.source} step={state.step:04d}] "
            f"phase={_pad(state.phase)} "
            f"drawer={_fmt(state.drawer_open)}m "
            f"holding={holding:7s} "
            f"tip={_deg(state.tip)} "
            f"progress={_pct(state.progress)} success={bool(state.success)}"
        )
        return None


def _fmt(value: float | None) -> str:
    return "None" if value is None else f"{value:.3f}"


def _pct(value: float | None) -> str:
    return "None" if value is None else f"{value * 100:.1f}%"


def _deg(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{math.degrees(value):3.0f}deg"


def _pad(label: str | None) -> str:
    return f"{label or 'none':<22}"


def build_pipeline(args: argparse.Namespace) -> tuple[Pipeline, DrawerSimulator]:
    simulator = DrawerSimulator(
        mode=args.mode,
        scene=getattr(args, "scene", None),
        camera=getattr(args, "camera", "action"),
        hz=args.hz,
    )
    pipe = Pipeline("robocasa_drawer")
    with pipe:
        sim = simulator @ Rate(hz=args.hz)
        policy = ChoreographyPolicy(hz=args.hz) @ Rate(hz=args.hz)
        printer = DrawerPrinter(print_every=args.print_every) @ Trigger("step")
        pipe.connect(policy, sim, sync=Latest())
        pipe.connect(sim, policy, sync=Latest())
        pipe.connect(sim, printer, sync=Latest())
    return pipe, simulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retriever + RoboCasa drawer demo: open a drawer by its "
                    "handle and season the plate with what is inside it."
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "mujoco"],
        default="mock",
        help="mock needs nothing installed; mujoco needs the simulator and a generated scene.xml.",
    )
    parser.add_argument("--scene", default=None, help="Path to a generated scene.xml.")
    parser.add_argument("--camera", default="action", help="Scene camera for the mujoco lane.")
    parser.add_argument("--steps", type=int, default=MOCK_STEPS,
                        help="Pipeline steps to run.")
    parser.add_argument("--dt", type=float, default=0.5, help="Seconds of wall clock per step.")
    parser.add_argument("--hz", type=float, default=2.0, help="Routine clock rate.")
    parser.add_argument("--print-every", type=int, default=10)
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
            f"(needs <= {SHUT_MAX}), seasoned={bool(final.seasoned)}, "
            f"shaker back in the drawer={bool(final.stowed)}, "
            f"{len(PHASES)} phases over {TOTAL_SECONDS:.1f} s "
            f"-> success={bool(final.success)}"
        )


if __name__ == "__main__":
    main()

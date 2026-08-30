"""Executes the `plan.py` choreography against a real MuJoCo model.

The schedule itself — the phases, their durations, the stroke and the grip
each commands — lives in `plan.py` and carries no simulator dependency, so the
mock lane in `app.py` runs the same routine this module drives. What lives
here is only the part that needs a model: turning a phase into a gripper pose
and asking `arm_control.Arm` for it.

The handle is a cylinder lying along world x, standing about 8 mm proud of the
drawer front. To take hold of it the fingers have to close vertically across
it, which is why this needs orientation control and not just a reach.
"""

from __future__ import annotations

import mujoco
import numpy as np

from examples.advanced.robocasa_drawer_dash.plan import (
    DRAWER_JOINT,
    GRASPING,
    HANDLE_BODY,
    HANDLE_GEOM,
    OPEN,
    PHASES,
    SQUEEZE,
    STANDOFF,
    STROKE,
    TOTAL_SECONDS,
    Phase,
    phase_at,
    smoothstep,
)

__all__ = [
    "GRASP_ROTATION",
    "GRASPING",
    "PHASES",
    "STROKE",
    "TOTAL_SECONDS",
    "Choreography",
    "Phase",
    "phase_at",
    "smoothstep",
    "OPEN",
    "SQUEEZE",
    "STANDOFF",
]

# Gripper pose for the grasp, as columns of a rotation matrix:
#   local x (the closing axis) -> world +z, so the fingers pinch top-to-bottom
#   local z (the approach axis) -> world +y, straight at the drawer front
GRASP_ROTATION = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
])


class Choreography:
    """Drives the arm through the plan, tick by tick. The drawer just follows."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, arm) -> None:
        self.model = model
        self.data = data
        self.arm = arm

        self.handle_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                             HANDLE_GEOM)
        self.handle_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                             HANDLE_BODY)
        if self.handle_geom < 0 and self.handle_body < 0:
            raise RuntimeError("no drawer handle found in the model")

        self.drawer_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                              DRAWER_JOINT)
        self.travel = float(abs(model.jnt_range[self.drawer_joint][0]))

        # Every finger pad, and every geom belonging to the handle: enough to
        # answer "is the gripper actually holding it right now?"
        self.pad_geoms = {g for g in range(model.ngeom)
                          if "pad_collision" in (model.geom(g).name or "")}
        self.handle_geoms = {g for g in range(model.ngeom)
                             if model.geom_bodyid[g] == self.handle_body}

        self.anchor: np.ndarray | None = None
        self.reset()

    # -- state --------------------------------------------------------------
    def reset(self) -> None:
        self.index = 0
        self.ticks = 0
        self.anchor = None

    @property
    def phase(self) -> Phase:
        return PHASES[self.index]

    @property
    def bar(self) -> np.ndarray:
        """World position of the handle bar, wherever the drawer has got to."""
        if self.handle_geom >= 0:
            return self.data.geom_xpos[self.handle_geom].copy()
        return self.data.xpos[self.handle_body].copy()

    @property
    def drawer_open(self) -> float:
        """How far the top drawer is pulled out, in metres."""
        return abs(float(self.data.qpos[self.model.jnt_qposadr[self.drawer_joint]]))

    def gripping(self) -> bool:
        """True while a finger pad is in contact with the handle."""
        for c in range(self.data.ncon):
            contact = self.data.contact[c]
            a, b = contact.geom1, contact.geom2
            if ((a in self.pad_geoms and b in self.handle_geoms)
                    or (b in self.pad_geoms and a in self.handle_geoms)):
                return True
        return False

    def target_pose(self, phase: Phase, blend: float):
        """Where the hand is asked to be this tick, or None to go home."""
        if phase.mode == "home":
            return None
        if phase.mode == "standoff":
            return self.bar + np.array([0.0, -STANDOFF, 0.0]), GRASP_ROTATION
        if phase.mode == "engage":
            return self.bar, GRASP_ROTATION
        # carry: drive the hand along -y from where the bar was when we grabbed
        start, end = phase.stroke
        pulled = start + (end - start) * blend
        return self.anchor + np.array([0.0, -pulled, 0.0]), GRASP_ROTATION

    # -- driving ------------------------------------------------------------
    def apply(self, phase: Phase, blend: float, dt: float | None = None) -> Phase:
        """Command one tick of `phase` without advancing the internal clock.

        This is the seam the Flow lane drives through: `app.py`'s policy owns
        the schedule and hands the phase in, so the mock and the simulator are
        replaying one timeline rather than two that have to be kept in step.
        """
        dt = self.model.opt.timestep if dt is None else dt

        if phase.mode == "carry":
            if self.anchor is None:
                # Remember where the bar was when we took hold of it. The hand
                # is commanded from there; the drawer has to come along.
                self.anchor = self.bar
        else:
            self.anchor = None

        pose = self.target_pose(phase, blend)
        if pose is None:
            self.arm.go_home(dt=dt)
        else:
            self.arm.reach(pose[0], pose[1], dt=dt)
        self.arm.set_gripper(phase.grip)
        return phase

    def step(self, dt: float | None = None) -> Phase:
        """Command one tick on this object's own clock. Returns the phase used."""
        dt = self.model.opt.timestep if dt is None else dt
        phase = self.phase
        # Count ticks rather than accumulating a float: a phase has to end on
        # exactly the tick its caller expects, or the schedule slips a frame
        # per phase and the labels stop matching what is on screen.
        span = max(1, int(round(phase.seconds / dt)))
        blend = smoothstep(self.ticks / span)
        self.apply(phase, blend, dt=dt)

        self.ticks += 1
        if self.ticks >= span:
            self.ticks = 0
            self.index = (self.index + 1) % len(PHASES)
        return phase

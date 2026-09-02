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

from examples.advanced.robocasa_drawer.plan import (
    DRAWER_JOINT,
    GRASPING,
    HANDLE_BODY,
    HANDLE_GEOM,
    HANDLE_TO_INTERIOR,
    TRANSIT_STANDOFF,
    TRANSIT_Z,
    HOLDING_JAR,
    JAR_CLEAR_Z,
    JAR_GRASP_Z,
    JAR_SQUEEZE,
    MOVING_DRAWER,
    OPEN,
    PHASES,
    SLOT_CLEAR_Z,
    SLOT_PLACE_Z,
    SLOT_X,
    SLOT_Y_FROM_CENTRE,
    SQUEEZE,
    STANDOFF,
    STROKE,
    TOTAL_SECONDS,
    Phase,
    phase_at,
    smoothstep,
)

WORKTOP_JAR_BODY = "cinnamon_main"

__all__ = [
    "GRASP_ROTATION",
    "JAR_ROTATION",
    "HOLDING_JAR",
    "MOVING_DRAWER",
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


# Gripper pose for the seasoning jar. The jar is an upright cylinder, so the
# hand comes straight down on it and the fingers close horizontally across the
# barrel -- the opposite of the handle, which is pinched top to bottom.
#   local x (the closing axis) -> world +x
#   local z (the approach axis) -> world -z, straight down
JAR_ROTATION = np.array([
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
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

        self.jar_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                          WORKTOP_JAR_BODY)
        self.jar_geoms = {g for g in range(model.ngeom)
                          if model.geom_bodyid[g] == self.jar_body}

        self.anchor: np.ndarray | None = None
        self.jar_rest: np.ndarray | None = None
        self.reset()

    # -- state --------------------------------------------------------------
    def reset(self) -> None:
        self.index = 0
        self.ticks = 0
        self.anchor = None
        # Where the jar is standing before anything touches it. Captured once,
        # because once it is grasped its own position follows the hand and
        # aiming at it would chase itself.
        self.jar_rest = (self.data.xpos[self.jar_body].copy()
                         if self.jar_body >= 0 else None)

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

    def _touching(self, targets: set[int]) -> bool:
        for c in range(self.data.ncon):
            contact = self.data.contact[c]
            a, b = contact.geom1, contact.geom2
            if ((a in self.pad_geoms and b in targets)
                    or (b in self.pad_geoms and a in targets)):
                return True
        return False

    def gripping(self) -> bool:
        """True while a finger pad is in contact with the handle."""
        return self._touching(self.handle_geoms)

    def holding_jar(self) -> bool:
        """True while a finger pad is in contact with the seasoning jar."""
        return self._touching(self.jar_geoms)

    @property
    def jar_position(self) -> np.ndarray:
        """Where the seasoning jar has got to."""
        return self.data.xpos[self.jar_body].copy()

    @property
    def slot(self) -> np.ndarray:
        """The spot in the drawer the jar is put down on, in world coordinates.

        Measured back from the handle bar, so it travels out with the drawer
        instead of staying where the drawer used to be.
        """
        centre_y = float(self.bar[1]) + HANDLE_TO_INTERIOR
        return np.array([SLOT_X, centre_y + SLOT_Y_FROM_CENTRE, 0.0])

    def target_pose(self, phase: Phase, blend: float):
        """Where the hand is asked to be this tick, or None to go home."""
        if phase.mode == "home":
            return None
        if phase.mode == "standoff":
            return self.bar + np.array([0.0, -STANDOFF, 0.0]), GRASP_ROTATION
        if phase.mode == "engage":
            return self.bar, GRASP_ROTATION
        if phase.mode == "transit_front":
            # Directly above the handle bar: in front of the drawer, over it,
            # never through it.
            return np.array([self.bar[0], self.bar[1], TRANSIT_Z]), JAR_ROTATION
        if phase.mode == "transit_worktop":
            rest = self.jar_rest if self.jar_rest is not None else self.jar_position
            return (np.array([rest[0], rest[1] - TRANSIT_STANDOFF, TRANSIT_Z]),
                    JAR_ROTATION)
        if phase.mode in {"jar_clear", "jar_grasp"}:
            rest = self.jar_rest if self.jar_rest is not None else self.jar_position
            height = JAR_CLEAR_Z if phase.mode == "jar_clear" else JAR_GRASP_Z
            return np.array([rest[0], rest[1], height]), JAR_ROTATION
        if phase.mode in {"slot_clear", "slot_place"}:
            slot = self.slot
            height = SLOT_CLEAR_Z if phase.mode == "slot_clear" else SLOT_PLACE_Z
            return np.array([slot[0], slot[1], height]), JAR_ROTATION
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

        if phase.mode == "carry" and self.anchor is None:
            # Remember where the bar was when we first took hold of it. The
            # hand is commanded from there; the drawer has to come along. It
            # is kept for the rest of the routine rather than cleared between
            # phases, because "push drawer shut" comes back to it long after
            # the pull -- with the whole jar errand in between -- and measures
            # its stroke from the same origin.
            self.anchor = self.bar

        pose = self.target_pose(phase, blend)
        if pose is None:
            self.arm.go_home(dt=dt)
        else:
            self.arm.reach(pose[0], pose[1], dt=dt)
        self.arm.set_gripper(phase.grip)
        self.arm.set_torso(phase.torso)
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

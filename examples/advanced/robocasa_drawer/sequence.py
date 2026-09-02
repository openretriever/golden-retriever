"""Executes the `plan.py` choreography against a real MuJoCo model.

The schedule itself — the phases, their durations, the stroke, the roll and the
grip each commands — lives in `plan.py` and carries no simulator dependency, so
the mock lane in `app.py` runs the same routine this module drives. What lives
here is only the part that needs a model: turning a phase into a gripper pose
and asking `arm_control.Arm` for it.

Two grasps, two orientations. The handle is a cylinder lying along world x,
standing about 8 mm proud of the drawer front, so the fingers have to close
*vertically* across it. The shaker stands upright on the drawer floor, so the
fingers close *horizontally* around its waist and the hand comes straight down
on it. Tipping the shaker over the plate is then a roll about the closing axis:
the fingers keep their hold and the shaker turns over with them. None of that
is reachable with position control alone, which is why this needs orientation
control and not just a reach.
"""

from __future__ import annotations

import mujoco
import numpy as np

from examples.advanced.robocasa_drawer.plan import (
    CARRYING,
    CLEARANCE,
    DRAWER_JOINT,
    FRONT_GAP,
    GRASPING,
    GRIP_RISE,
    HANDLE_BODY,
    HANDLE_GEOM,
    HANDLE_SQUEEZE,
    HOVER_RISE,
    OPEN,
    PHASES,
    PLACE_RISE,
    PLATE_BODY,
    SHAKE_GAIN,
    SHAKE_HZ,
    SHAKE_LIFT,
    SHAKE_RATE,
    SHAKE_ROLL,
    SHAKE_SPIN_GAIN,
    SHAKER_BODY,
    SHAKER_SQUEEZE,
    SQUARE_TO_HANDLE,
    STANDOFF,
    STROKE,
    TIP_ANGLE,
    TORSO_HIGH,
    TORSO_SPEED,
    TOTAL_SECONDS,
    TRANSIT_Z,
    WORK_RISE,
    Phase,
    phase_at,
    smoothstep,
)

__all__ = [
    "CARRYING",
    "GRASPING",
    "GRASP_ROTATION",
    "PICK_ROTATION",
    "PHASES",
    "SQUARE_TO_HANDLE",
    "STROKE",
    "TOTAL_SECONDS",
    "Choreography",
    "Phase",
    "phase_at",
    "smoothstep",
    "OPEN",
    "HANDLE_SQUEEZE",
    "SHAKER_SQUEEZE",
    "STANDOFF",
]

# --- gripper poses, as columns of a rotation matrix -------------------------
# GRASP: local x (the closing axis) -> world +z, so the fingers pinch the handle
#        top-to-bottom; local z (the approach axis) -> world +y, at the drawer.
GRASP_ROTATION = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
])
# PICK: local x -> world +x, so the fingers pinch the shaker across the drawer;
#       local z -> world -z, straight down onto it.
PICK_ROTATION = np.array([
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
])


def _roll(angle: float) -> np.ndarray:
    """Rotation about world x — the axis the fingers pinch along when picking."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


class Choreography:
    """Drives the arm through `PHASES`, tick by tick.

    The drawer and the shaker are not driven at all; they move because the
    gripper is on them.
    """

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
        self.shaker_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                             SHAKER_BODY)
        self.plate_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                            PLATE_BODY)

        # Every finger pad, and every geom belonging to the handle and to the
        # shaker: enough to answer "is the gripper actually holding it?"
        self.pad_geoms = {g for g in range(model.ngeom)
                          if "pad_collision" in (model.geom(g).name or "")}
        self.handle_geoms = {g for g in range(model.ngeom)
                             if model.geom_bodyid[g] == self.handle_body}
        self.shaker_geoms = {g for g in range(model.ngeom)
                             if model.geom_bodyid[g] == self.shaker_body}

        # Where the shake happens, read off the scene rather than hard-coded.
        self.work_z = float(data.xpos[self.plate_body][2]) + WORK_RISE
        self.reset()

    # -- state --------------------------------------------------------------
    def reset(self) -> None:
        self.index = 0
        self.ticks = 0
        self.entered: Phase | None = None       # the phase `_enter` last ran for
        self.anchor: np.ndarray | None = None   # hand position when carrying
        self.anchor_open = 0.0                  # ... and the drawer's travel then
        self.start: np.ndarray | None = None    # hand position at phase entry
        self.goal: np.ndarray | None = None     # frozen waypoint for this phase
        self.square: np.ndarray | None = None   # frozen rotation for this phase
        self.grip_point: np.ndarray | None = None   # where the shaker was taken
        self.twins: dict[str, np.ndarray] = {}  # the half-turn chosen per grasp
        self.torso = TORSO_HIGH                 # commanded height of the base lift

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
    def shaker(self) -> np.ndarray:
        """World position of the pepper shaker's body origin."""
        return self.data.xpos[self.shaker_body].copy()

    @property
    def drawer_open(self) -> float:
        """How far the top drawer is pulled out, in metres."""
        return abs(float(self.data.qpos[self.model.jnt_qposadr[self.drawer_joint]]))

    def _touching(self, geoms: set[int]) -> bool:
        for c in range(self.data.ncon):
            contact = self.data.contact[c]
            a, b = contact.geom1, contact.geom2
            if ((a in self.pad_geoms and b in geoms)
                    or (b in self.pad_geoms and a in geoms)):
                return True
        return False

    def gripping(self) -> bool:
        """True while a finger pad is in contact with the handle."""
        return self._touching(self.handle_geoms)

    def holding(self) -> bool:
        """True while a finger pad is in contact with the shaker."""
        return self._touching(self.shaker_geoms)

    def tip(self) -> float:
        """How far the shaker is from upright right now, in radians."""
        axis = self.data.xmat[self.shaker_body].reshape(3, 3)[:, 2]
        return float(np.arccos(np.clip(axis[2], -1.0, 1.0)))

    # -- waypoints ----------------------------------------------------------
    def waypoint(self, name: str) -> np.ndarray:
        """Resolve a named point, in world coordinates, from the live scene."""
        bar = self.bar
        if name == "front_high":
            # In front of the open drawer and above it: the one place the hand
            # can cross between the drawer's inside and the tabletop.
            return np.array([bar[0], bar[1] - FRONT_GAP, TRANSIT_Z])
        if name == "plate_high":
            plate = self.data.xpos[self.plate_body]
            return np.array([plate[0], plate[1], TRANSIT_Z])
        if name == "shaker_hover":
            return self.shaker + np.array([0.0, 0.0, HOVER_RISE])
        if name == "shaker_grip":
            return self.shaker + np.array([0.0, 0.0, GRIP_RISE])
        if name == "straight_up":
            return np.array([self.start[0], self.start[1], TRANSIT_Z])
        if name == "work":
            plate = self.data.xpos[self.plate_body]
            return np.array([plate[0], plate[1], self.work_z])
        if name == "over_slot":
            return np.array([self.grip_point[0], self.grip_point[1], TRANSIT_Z])
        if name == "place":
            return self.grip_point + np.array([0.0, 0.0, PLACE_RISE])
        raise KeyError(name)

    def target_pose(self, phase: Phase, blend: float):
        """Where the hand is asked to be this tick, or None to go home."""
        if phase.mode == "home":
            return None
        rotation = self.square if self.square is not None else PICK_ROTATION
        if phase.mode == "standoff":
            gap = STANDOFF if self.drawer_open < 0.05 else CLEARANCE
            return self.bar + np.array([0.0, -gap, 0.0]), rotation
        if phase.mode == "engage":
            return self.bar, rotation
        if phase.mode == "carry":
            # Drive the hand along -y from where the bar was when we grabbed it.
            # The travel is measured against how far out the drawer actually was
            # at that moment, not against where the routine assumed it would be:
            # anything that has nudged the drawer since is corrected for here.
            pulled = (phase.open_to - self.anchor_open) * blend
            return self.anchor + np.array([0.0, -pulled, 0.0]), rotation
        if phase.mode == "hold":
            return self.start, rotation
        if phase.mode == "move":
            goal = self.goal if self.goal is not None else self.waypoint(phase.goal)
            return self.start + (goal - self.start) * blend, rotation
        if phase.mode in ("tilt", "shake"):
            base = self.goal
            first, last = phase.tilt
            if phase.mode == "tilt":
                angle = first + (last - first) * blend
                return base, _roll(angle) @ rotation
            wave = np.sin(2.0 * np.pi * SHAKE_HZ * blend * phase.seconds)
            angle = TIP_ANGLE + SHAKE_ROLL * wave
            point = base + np.array([0.0, 0.0, SHAKE_LIFT * wave])
            return point, _roll(angle) @ rotation
        raise KeyError(phase.mode)

    # -- driving ------------------------------------------------------------
    def _enter(self, phase: Phase) -> None:
        """Freeze whatever this phase measures once, at its first tick."""
        self.entered = phase
        self.start = self.arm.tip
        if phase.mark_grip:
            self.grip_point = self.arm.tip
        if phase.mode == "carry":
            if self.anchor is None:
                # Remember where the hand closed on the bar, and how far out
                # the drawer stood at that moment. The hand is commanded from
                # there; the drawer has to come along.
                self.anchor = self.arm.tip
                self.anchor_open = self.drawer_open
        else:
            self.anchor = None
        if phase.mode in ("tilt", "shake"):
            # Stay where the previous phase left off; only the roll changes.
            self.goal = self.start
        elif phase.mode == "move":
            self.goal = self.waypoint(phase.goal)
        else:
            self.goal = None
        # A parallel jaw grasps the same either way up, so hand the solver
        # whichever half-turn the wrist is already nearer — that is what keeps
        # joint 7 off its stop. Picked once per grasp and then kept: re-picking
        # would flip the wrist 180 degrees part way through a roll, and coming
        # back to the handle after the plate the near twin is the one the wrist
        # cannot actually reach.
        family = "handle" if phase.mode in SQUARE_TO_HANDLE else "shaker"
        if family not in self.twins:
            wanted = GRASP_ROTATION if family == "handle" else PICK_ROTATION
            self.twins[family] = self.arm.equivalent(wanted)
        self.square = self.twins[family]

    def apply(self, phase: Phase, blend: float, dt: float | None = None) -> Phase:
        """Command one tick of `phase` without advancing the internal clock.

        This is the seam the Flow lane drives through: `app.py`'s policy owns
        the schedule and hands the phase in, so the mock and the simulator are
        replaying one timeline rather than two that have to be kept in step.
        A phase entered from outside still gets its `_enter` — the waypoints
        this routine aims at are frozen at the tick a phase begins, so they
        have to be frozen wherever that tick comes from.
        """
        dt = self.model.opt.timestep if dt is None else dt
        if phase is not self.entered:
            self._enter(phase)

        pose = self.target_pose(phase, blend)
        if pose is None:
            self.arm.go_home(dt=dt)
        elif phase.mode == "shake":
            self.arm.reach(pose[0], pose[1], dt=dt, gain=SHAKE_GAIN,
                           spin_gain=SHAKE_SPIN_GAIN, rate_limit=SHAKE_RATE)
        else:
            self.arm.reach(pose[0], pose[1], dt=dt)
        self.arm.set_gripper(phase.grip)
        # Ease the base's lift towards this phase's height. It moves under the
        # arm while the arm is tracking, which the reach loop simply absorbs.
        step = TORSO_SPEED * dt
        self.torso += float(np.clip(phase.torso - self.torso, -step, step))
        self.arm.set_torso(self.torso)
        return phase

    def step(self, dt: float | None = None) -> Phase:
        """Command one tick on this object's own clock. Returns the phase used."""
        dt = self.model.opt.timestep if dt is None else dt
        phase = self.phase
        # Count ticks rather than accumulating a float: a phase has to end on
        # exactly the tick its caller expects, or the schedule slips a frame
        # per phase and the labels stop matching what is on screen.
        span = max(1, int(round(phase.seconds / dt)))
        raw = self.ticks / span
        # A shake is not eased; it is a wave.
        blend = raw if phase.mode == "shake" else smoothstep(raw)
        self.apply(phase, blend, dt=dt)

        self.ticks += 1
        if self.ticks >= span:
            self.ticks = 0
            self.index = (self.index + 1) % len(PHASES)
            if self.index == 0:
                self.grip_point = None
                self.twins.clear()
        return phase

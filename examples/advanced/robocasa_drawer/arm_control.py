"""Minimal Cartesian control for the Panda in `scene.xml`.

The scene drives the arm through position actuators, so all this needs to do is
turn a desired gripper pose into joint targets. That is one damped
least-squares step against the site Jacobian per control tick, on both position
and orientation — orientation matters here because the fingers have to end up
across the handle bar rather than merely near it, and because tipping a shaker
over a plate is entirely an orientation move.

Seven joints against a six-row task leaves one degree of freedom over, and this
uses it to stay away from the joint stops: a posture term projected into the
Jacobian's nullspace, so it never disturbs the commanded pose. Without it the
arm reaches the plate with the shoulder pinned at its limit and then cannot tip
the shaker at all.

It is deliberately not a robosuite controller: no impedance, no force limits
beyond the actuators' own.
"""

from __future__ import annotations

import mujoco
import numpy as np

ARM_JOINTS = tuple(f"robot0_joint{i}" for i in range(1, 8))
FINGER_JOINTS = ("gripper0_finger_joint1", "gripper0_finger_joint2")
TORSO_JOINT = "mobilebase0_joint_torso_height"
GRIP_SITE = "gripper0_grip_site"

DAMPING = 0.12       # DLS lambda; larger is steadier near singularities
NULL_DAMPING = 0.02  # the same, for the nullspace projector only — see reach()
TIP_SPEED = 0.60     # metres per second the gripper is allowed to travel
SPIN_SPEED = 2.5     # radians per second the gripper is allowed to rotate
JOINT_SPEED = 1.8    # radians per second any one joint is allowed to turn
APPROACH_GAIN = 4.0  # 1/s; how hard the tip is pulled at the remaining error
ALIGN_GAIN = 5.0     # 1/s; the same, for orientation
POSTURE_GAIN = 1.2   # 1/s; nullspace pull back towards the middle of each joint

# The two finger joints run in opposite directions; see scene.py.
FINGER_SIGNS = (1.0, -1.0)
OPEN = 0.04          # metres of travel per finger

# A parallel jaw grasps the same way after a half turn about its approach axis:
# the two fingers simply swap places. `Arm.equivalent` uses that to hand the
# solver whichever of the pair the wrist is already nearer, which is what keeps
# joint 7 off its stop when the hand has to end up pointing backwards.
HALF_TURN = np.diag([-1.0, -1.0, 1.0])


class Arm:
    """Joint targets for the Panda, addressed either by pose or by gripper point."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in ARM_JOINTS]
        self.dof_ids = np.array([model.jnt_dofadr[j] for j in self.joint_ids])
        self.qpos_ids = np.array([model.jnt_qposadr[j] for j in self.joint_ids])
        self.act_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_act")
            for n in ARM_JOINTS])
        self.finger_act_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_act")
            for n in FINGER_JOINTS])
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
        self.torso_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                           f"{TORSO_JOINT}_act")
        self.torso_qpos = model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, TORSO_JOINT)]

        self.limits = model.jnt_range[self.joint_ids].copy()
        self.middle = self.limits.mean(axis=1)
        self.half_span = (self.limits[:, 1] - self.limits[:, 0]) / 2.0
        # Whatever the home keyframe put in ctrl is the starting target.
        self.home = data.ctrl[self.act_ids].copy()
        self.target = self.home.copy()

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # -- queries ------------------------------------------------------------
    @property
    def tip(self) -> np.ndarray:
        """Current gripper position: the point midway between the fingertips."""
        return self.data.site_xpos[self.site_id].copy()

    @property
    def orientation(self) -> np.ndarray:
        """Gripper rotation as a 3x3, columns [closing axis, _, approach axis]."""
        return self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def distance_to(self, point: np.ndarray) -> float:
        return float(np.linalg.norm(self.tip - np.asarray(point, dtype=float)))

    def misalignment(self, rotation: np.ndarray) -> float:
        """Angle in radians between the gripper's pose and `rotation`."""
        return float(np.linalg.norm(_rotation_error(self.orientation, rotation)))

    def equivalent(self, rotation: np.ndarray) -> np.ndarray:
        """`rotation`, or its half-turn twin, whichever the gripper is nearer.

        Both describe the same physical grasp, so picking the near one saves
        the wrist a 180-degree roll it has no room for.
        """
        rotation = np.asarray(rotation, dtype=float)
        twin = rotation @ HALF_TURN
        return rotation if self.misalignment(rotation) <= self.misalignment(twin) \
            else twin

    def limit_margin(self) -> float:
        """How close the closest joint is to its stop, as a fraction of its range.

        0 means a joint is sitting on a limit — the configuration has run out of
        room and the next Cartesian command in that direction will not be met.
        """
        q = self.data.qpos[self.qpos_ids]
        return float(np.min(np.minimum(q - self.limits[:, 0],
                                       self.limits[:, 1] - q) / (2 * self.half_span)))

    # -- commands -----------------------------------------------------------
    def apply(self) -> None:
        """Write the current joint targets into the actuators."""
        self.data.ctrl[self.act_ids] = self.target

    def go_home(self, dt: float | None = None) -> None:
        """Ease the targets back towards the home pose."""
        dt = self.model.opt.timestep if dt is None else dt
        step = JOINT_SPEED * dt
        self.target += np.clip(self.home - self.target, -step, step)
        self.apply()

    @property
    def torso(self) -> float:
        """How far the base's lift has raised the shoulder, in metres."""
        return float(self.data.qpos[self.torso_qpos])

    def set_torso(self, height: float) -> None:
        """Command the base's lift. The reach loop absorbs the shoulder moving
        under it — it works from the measured fingertip, not from a model."""
        self.data.ctrl[self.torso_act] = float(height)

    def set_gripper(self, opening: float) -> None:
        """Finger opening in metres, 0 (closed) to 0.04 (fully open).

        Commanding a value below what the grasped object allows is how the
        squeeze is produced: the actuators stall against it and hold.
        """
        value = float(np.clip(opening, 0.0, OPEN))
        self.data.ctrl[self.finger_act_ids] = [s * value for s in FINGER_SIGNS]

    def reach(self, point, rotation=None, dt: float | None = None,
              gain: float | None = None, spin_gain: float | None = None,
              rate_limit: float | None = None) -> float:
        """Advance the gripper towards `point` (and `rotation`) by one tick.

        The pose error is turned into a speed-limited twist, the twist into
        joint rates through a damped least-squares inverse, a posture term that
        backs every joint away from its stops is added in the nullspace, and the
        result is integrated into the joint targets. Call it every tick —
        including while the target is moving, which is how the arm carries a
        drawer out.

        `gain`, `spin_gain` and `rate_limit` override the module defaults for
        this tick; the shake needs a much brisker loop than a reach does.

        Returns the distance still to cover, so a caller can wait on it.
        """
        dt = self.model.opt.timestep if dt is None else dt
        gain = APPROACH_GAIN if gain is None else gain
        spin_gain = ALIGN_GAIN if spin_gain is None else spin_gain
        rate_limit = JOINT_SPEED if rate_limit is None else rate_limit
        point = np.asarray(point, dtype=float)

        error = point - self.tip
        distance = float(np.linalg.norm(error))
        velocity = np.zeros(3)
        if distance > 1e-9:
            velocity = error / distance * min(TIP_SPEED, gain * distance)

        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        if rotation is None:
            jac = self._jacp[:, self.dof_ids]
            twist = velocity
        else:
            turn = _rotation_error(self.orientation, np.asarray(rotation, dtype=float))
            angle = float(np.linalg.norm(turn))
            spin = np.zeros(3)
            if angle > 1e-9:
                spin = turn / angle * min(SPIN_SPEED, spin_gain * angle)
            jac = np.vstack([self._jacp[:, self.dof_ids], self._jacr[:, self.dof_ids]])
            twist = np.concatenate([velocity, spin])

        # dq/dt = J^+ twist + (I - J^# J) posture, with J^+ the damped inverse
        # J^T (J J^T + lambda^2 I)^-1. The projector uses a *lighter* damping
        # than the solve: how much the posture term leaks into the commanded
        # pose goes as lambda^2, and at the solve's damping it leaks about a
        # centimetre — enough to miss the handle.
        eye = np.eye(jac.shape[0])
        pinv = jac.T @ np.linalg.inv(jac @ jac.T + DAMPING ** 2 * eye)
        proj = jac.T @ np.linalg.inv(jac @ jac.T + NULL_DAMPING ** 2 * eye)
        posture = -POSTURE_GAIN * (self.target - self.middle) / self.half_span
        rate = pinv @ twist + (np.eye(len(self.target)) - proj @ jac) @ posture
        rate = np.clip(rate, -rate_limit, rate_limit)

        self.target = np.clip(self.target + rate * dt,
                              self.limits[:, 0], self.limits[:, 1])
        self.apply()
        return distance


def _rotation_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Rotation taking `current` to `desired`, as a world-frame rotation vector."""
    relative = desired @ current.T
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, relative.flatten())
    axis_angle = np.zeros(3)
    mujoco.mju_quat2Vel(axis_angle, quat, 1.0)
    return axis_angle

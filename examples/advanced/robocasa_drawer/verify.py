"""Prove the arm really does the job, headlessly.

Runs the routine in `sequence.py` against `scene.xml` and asserts on it. Two
claims are central, and both are structural before they are numerical:

  * the drawer is opened and shut by the grasp on its handle and nothing else —
    so the first check is that no actuator acts on a drawer slide joint;
  * the shaker reaches the plate because the gripper carried it — so the second
    is that the shaker is a free body with no actuator on it either.

If either were driven, every other number here would be worthless.

  * no actuator drives any drawer, and none drives the shaker
  * the gripper squares up to the handle bar and closes on it
  * a finger pad stays in contact with the handle for the whole pull and push
  * the drawer comes out under that grasp, and goes back in
  * the fingers close on the shaker and hold it for the whole excursion
  * the shaker leaves the drawer, ends up over the plate, and is tipped past
    horizontal there — cap downwards, over the food
  * it is shaken: its height reverses direction several times over the plate
  * it goes back into the drawer, within a couple of centimetres of where it
    was picked up, and is standing upright again
  * the drawer ends shut, and the arm back at its home pose
  * nothing else in the scene is disturbed: the other three seasonings stay
    standing in the drawer, the jar stays on the worktop, and the plate and its
    food stay where they were put

    pixi run demo-drawer-verify                    # assertions + logs, no render
    pixi run demo-drawer-verify -- --video out/scene.mp4 --sheet out/scene.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from examples.advanced.robocasa_drawer.arm_control import Arm
from examples.advanced.robocasa_drawer.plan import (
    CARRYING,
    GRASPING,
    LIFT_MIN,
    OPENED_MIN,
    PHASES,
    SHUT_MAX,
    STROKE,
    TIP_MIN,
)
from examples.advanced.robocasa_drawer.scene import TOP_DRAWER_FLOOR, ensure_scene
from examples.advanced.robocasa_drawer.sequence import (
    GRASP_ROTATION,
    PICK_ROTATION,
    SQUARE_TO_HANDLE,
    Choreography,
)

HERE = Path(__file__).resolve().parent

POSE_TOL = 0.05        # metres the hand may sit off its commanded point
ANGLE_TOL = np.radians(6.0)   # how far off square the gripper may be
HOME_TOL = 0.05        # radians per joint once the arm has withdrawn
CONTACT_TOL = 0.95     # fraction of a carry phase a pad must touch the handle
GRIP_TOL = 0.50        # ... and of the phase where it is still closing
NUDGE_MAX = 0.03       # metres the drawer may drift while the arm is elsewhere
# OPENED_MIN / SHUT_MAX / LIFT_MIN / TIP_MIN come from plan.py, shared with the
# mock lane, so the two lanes are judged against one set of numbers.

SHAKER = "pepper_main"
PLATE = "plate_main"
FOOD = ("steak_main", "broccoli_main")
DRAWER_MATES = ("salt_main", "paprika_main", "cinnamon_main")
WORKTOP_JAR = "worktop_jar_main"

OVER_PLATE_TOL = 0.10  # metres the shaker may sit off the plate's centre
SHAKE_TRAVEL_MIN = 0.03       # metres of up-and-down over the plate
SHAKE_REVERSALS_MIN = 4       # ... and how many times it has to change direction
RETURN_TOL = 0.03      # metres from where the shaker was picked up
UPRIGHT_TOL = np.radians(20)  # how far a standing seasoning may lean
BYSTANDER_TOL = 0.02   # metres anything the arm never touches may move


class Rig:
    def __init__(self, xml: Path, width: int = 960, height: int = 720,
                 camera: str = "action") -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.width, self.height = width, height
        self.camera = camera
        self._renderer: mujoco.Renderer | None = None
        self.frames: list[np.ndarray] = []
        self.marks: dict[str, int] = {}

        # The scene ships a `home` keyframe: arm rest pose plus matching targets.
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self.arm = Arm(self.model, self.data)
        self.routine = Choreography(self.model, self.data, self.arm)
        # Filled in after the `settle` phase: the food is dropped onto the plate
        # rather than placed on it, so it takes a moment to come to rest, and
        # "did the arm disturb it" is a question about where it settled.
        self.start: dict[str, np.ndarray] = {}

    def mark_start(self) -> None:
        self.start = {n: self.body_pos(n) for n in
                      (PLATE, WORKTOP_JAR, *FOOD, *DRAWER_MATES)}

    # -- state ---------------------------------------------------------------
    def body_pos(self, name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[bid].copy()

    def tilt(self, name: str) -> float:
        """Angle between the body's own z axis and world up, in radians."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        axis = self.data.xmat[bid].reshape(3, 3)[:, 2]
        return float(np.arccos(np.clip(axis[2], -1.0, 1.0)))

    def home_error(self) -> float:
        """Largest joint-angle gap between the arm and its home pose."""
        actual = self.data.qpos[self.arm.qpos_ids]
        return float(np.max(np.abs(actual - self.arm.home)))

    def driven(self, needle: str) -> list[str]:
        """Actuators acting on a joint whose name contains `needle`. Must be []."""
        found = []
        for a in range(self.model.nu):
            joint = self.model.actuator_trnid[a][0]
            if joint < 0:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if name and needle in name:
                found.append(mujoco.mj_id2name(self.model,
                                               mujoco.mjtObj.mjOBJ_ACTUATOR, a))
        return found

    def shaker_stowed(self) -> bool:
        """True while the shaker is standing upright down inside the drawer.

        Height alone is enough to separate "in the drawer" from "in the air":
        the drawer's own floor is at `TOP_DRAWER_FLOOR` and the routine's
        transit lane runs 20 cm above the rim. The lean is what says it was set
        down rather than dropped.
        """
        pos = self.body_pos(SHAKER)
        return bool(pos[2] < TOP_DRAWER_FLOOR + 0.11
                    and self.tilt(SHAKER) < UPRIGHT_TOL)

    def shaker_joints(self) -> list[str]:
        """The shaker's own joints — it should have exactly one free joint and
        no actuator on it, i.e. it is loose in the drawer."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, SHAKER)
        out = []
        for j in range(self.model.njnt):
            if self.model.jnt_bodyid[j] == bid:
                out.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j))
        return out

    # -- rendering -----------------------------------------------------------
    def _ensure_renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.height,
                                             width=self.width)
            # robosuite convention: group 0 is collision, group 1 is visual.
            # Showing group 0 paints the scene in translucent red hulls.
            self._scene_option = mujoco.MjvOption()
            self._scene_option.geomgroup[:] = 0
            self._scene_option.geomgroup[1] = 1
            self._scene_option.geomgroup[2] = 1
        return self._renderer

    def capture(self) -> None:
        r = self._ensure_renderer()
        r.update_scene(self.data, camera=self.camera,
                       scene_option=self._scene_option)
        self.frames.append(r.render())

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # -- stepping ------------------------------------------------------------
    def apply(self, action, seconds: float | None = None) -> None:
        """Advance `seconds` of physics under an externally commanded phase.

        `app.py` owns the schedule in the Flow lane, so the phase arrives from
        outside instead of from this rig's own clock. One Flow tick is worth
        many physics ticks — the routine is commanded at tens of hertz and the
        model integrates at 500 — so this substeps to cover the tick rather
        than advancing a single 2 ms step per call, which would let the
        policy's clock run away from simulated time.

        `run_phase` below is the standalone path and still uses the clock in
        `Choreography`.
        """
        index = 0 if action.phase_index is None else int(action.phase_index)
        blend = 0.0 if action.blend is None else float(action.blend)
        phase = PHASES[index]
        timestep = self.model.opt.timestep
        span = timestep if seconds is None else max(timestep, float(seconds))
        for _ in range(max(1, int(round(span / timestep)))):
            self.routine.apply(phase, blend)
            mujoco.mj_step(self.model, self.data)

    def run_phase(self, render: bool, fps: int = 30) -> dict:
        """Play one phase to its end and report what happened during it."""
        phase = self.routine.phase
        steps = max(1, int(round(phase.seconds / self.model.opt.timestep)))
        every = max(1, int(round(1.0 / (fps * self.model.opt.timestep))))
        on_handle = on_shaker = 0
        square = GRASP_ROTATION if phase.mode in SQUARE_TO_HANDLE else PICK_ROTATION
        worst_angle = 0.0
        heights: list[float] = []
        tips: list[float] = []
        for s in range(steps):
            self.routine.step()
            mujoco.mj_step(self.model, self.data)
            on_handle += self.routine.gripping()
            on_shaker += self.routine.holding()
            if phase.label in GRASPING:
                worst_angle = max(worst_angle, self._square_error(square))
            heights.append(float(self.body_pos(SHAKER)[2]))
            tips.append(self.tilt(SHAKER))
            if render and s % every == 0:
                self.capture()
        self.marks[phase.label] = max(0, len(self.frames) - 1)

        pose = self.routine.target_pose(phase, 1.0)
        return {
            "label": phase.label,
            "handle": on_handle / steps,
            "shaker": on_shaker / steps,
            "angle": (self._square_error(pose[1]) if pose is not None
                      else float("nan")),
            "square": self._square_error(square),
            "worst_angle": worst_angle,
            "pose_error": self.arm.distance_to(pose[0]) if pose else float("nan"),
            "drawer": self.routine.drawer_open,
            "heights": np.asarray(heights),
            "tips": np.asarray(tips),
        }

    def _square_error(self, square: np.ndarray) -> float:
        """Misalignment against whichever half-turn of `square` is in use."""
        return min(self.arm.misalignment(square),
                   self.arm.misalignment(self.arm.equivalent(square)))


def _reversals(series: np.ndarray, window: int = 25) -> int:
    """How many times a (smoothed) series changes direction."""
    if len(series) < 3 * window:
        return 0
    smooth = np.convolve(series, np.ones(window) / window, mode="valid")
    step = np.diff(smooth)
    return int(np.sum(np.sign(step[1:]) != np.sign(step[:-1])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, default=None,
                    help="Path to scene.xml; built on first run if absent.")
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--sheet", type=Path, default=None)
    ap.add_argument("--camera", default="action")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    scene_path = ensure_scene(args.scene)
    render = args.video is not None or args.sheet is not None
    rig = Rig(scene_path, camera=args.camera)
    failures: list[str] = []

    print(f"scene: {scene_path}")
    print(f"slide travel: {rig.routine.travel:.3f} m, commanded pull: {STROKE:.3f} m")

    driven = rig.driven("slidejoint")
    if driven:
        failures.append("drawer slide joints are actuated by "
                        f"{', '.join(driven)}; the pull would not prove anything")
    else:
        print("drawer slides: passive, no actuator — only a grasp can move them")
    joints = rig.shaker_joints()
    if rig.driven(SHAKER.replace("_main", "")):
        failures.append("the shaker is actuated; carrying it would prove nothing")
    else:
        print(f"shaker:        free body on {len(joints)} joint"
              f"{'' if len(joints) == 1 else 's'}, no actuator — "
              "only a grasp can move it")
    if render:
        rig.capture()

    pick_at = place_at = None
    drawer_when_left = None
    tipped = 0.0
    lifted = 0.0
    shake_travel = 0.0
    shake_reversals = 0

    for phase in PHASES:
        r = rig.run_phase(render=render, fps=args.fps)
        label = r["label"]
        shaker = rig.body_pos(SHAKER)
        print(f"[{label:<22}] drawer={r['drawer']:.3f}"
              f"  hand_err={r['pose_error']:.4f}"
              f"  orient={np.degrees(r['angle']):5.1f}deg"
              f"  handle={r['handle'] * 100:3.0f}%"
              f"  shaker={r['shaker'] * 100:3.0f}%"
              f"  shaker_xyz=({shaker[0]:+.3f},{shaker[1]:+.3f},{shaker[2]:.3f})"
              f"  tip={np.degrees(rig.tilt(SHAKER)):5.1f}deg")

        if phase.mode != "home" and r["pose_error"] > POSE_TOL:
            failures.append(f"{label}: hand ended {r['pose_error']:.3f} m off "
                            f"its commanded point (limit {POSE_TOL:.2f} m)")
        if label == "settle":
            rig.mark_start()
        if label in GRASPING:
            if r["square"] > ANGLE_TOL:
                failures.append(
                    f"{label}: gripper ended {np.degrees(r['square']):.1f}deg off "
                    f"square to the handle "
                    f"(limit {np.degrees(ANGLE_TOL):.0f}deg)")
            # Once it is holding the bar it has to stay square for the whole
            # phase, not merely arrive square.
            if r["worst_angle"] > ANGLE_TOL:
                failures.append(
                    f"{label}: gripper twisted "
                    f"{np.degrees(r['worst_angle']):.1f}deg off square while "
                    f"holding the handle "
                    f"(limit {np.degrees(ANGLE_TOL):.0f}deg)")
            floor = GRIP_TOL if phase.mode == "engage" else CONTACT_TOL
            if r["handle"] < floor:
                failures.append(
                    f"{label}: fingers touched the handle for only "
                    f"{r['handle'] * 100:.0f}% of the phase "
                    f"(need {floor * 100:.0f}%)")
        if label in CARRYING:
            floor = GRIP_TOL if label == "grip the shaker" else CONTACT_TOL
            if r["shaker"] < floor:
                failures.append(
                    f"{label}: fingers touched the shaker for only "
                    f"{r['shaker'] * 100:.0f}% of the phase "
                    f"(need {floor * 100:.0f}%)")

        if label == "pull the drawer open":
            if r["drawer"] < OPENED_MIN:
                failures.append(
                    f"the grasp only pulled the drawer out {r['drawer']:.3f} m "
                    f"(need {OPENED_MIN:.2f} m)")
            drawer_when_left = r["drawer"]
        if label == "grip the shaker":
            pick_at = shaker.copy()
        if label == "lift it out" and pick_at is not None:
            lifted = float(shaker[2] - pick_at[2])
            if lifted < LIFT_MIN:
                failures.append(f"the shaker only came up {lifted:.3f} m out of "
                                f"the drawer (need {LIFT_MIN:.2f} m)")
        if label == "down to the plate":
            plate = rig.body_pos(PLATE)
            over_plate = float(np.linalg.norm(shaker[:2] - plate[:2]))
            if over_plate > OVER_PLATE_TOL:
                failures.append(f"the shaker ended {over_plate:.3f} m to one "
                                f"side of the plate's centre "
                                f"(limit {OVER_PLATE_TOL:.2f} m)")
        if label == "shake the seasoning":
            tipped = float(np.min(r["tips"]))
            shake_travel = float(r["heights"].max() - r["heights"].min())
            shake_reversals = _reversals(r["heights"])
            plate = rig.body_pos(PLATE)
            if float(np.linalg.norm(shaker[:2] - plate[:2])) > OVER_PLATE_TOL:
                failures.append("the shaker drifted off the plate while shaking")
            if tipped < TIP_MIN:
                failures.append(
                    f"the shaker was never tipped past "
                    f"{np.degrees(TIP_MIN):.0f}deg while shaking "
                    f"(least tip {np.degrees(tipped):.0f}deg) — it has to be "
                    f"cap-down over the food")
            if shake_travel < SHAKE_TRAVEL_MIN:
                failures.append(f"the shaker moved only {shake_travel * 1000:.0f} mm "
                                f"over the plate; that is not a shake")
            if shake_reversals < SHAKE_REVERSALS_MIN:
                failures.append(f"the shaker changed direction only "
                                f"{shake_reversals} times over the plate "
                                f"(need {SHAKE_REVERSALS_MIN})")
        if label == "let go of the shaker":
            place_at = shaker.copy()
        if label == "lift clear" and drawer_when_left is not None:
            drift = abs(r["drawer"] - drawer_when_left)
            if drift > NUDGE_MAX:
                failures.append(
                    f"the drawer moved {drift:.3f} m while the arm was away at "
                    f"the plate — the arm knocked it (limit {NUDGE_MAX:.2f} m)")
        if label == "push the drawer shut" and r["drawer"] > SHUT_MAX:
            failures.append(f"drawer left standing {r['drawer']:.3f} m proud "
                            f"(limit {SHUT_MAX:.2f} m)")
        if label == "withdraw":
            error = rig.home_error()
            print(f"    arm back at home to within {error:.4f} rad")
            if error > HOME_TOL:
                failures.append(
                    f"arm did not return home: worst joint off by {error:.3f} rad")

    # -- where everything ended up ------------------------------------------
    print()
    if pick_at is not None and place_at is not None:
        back = float(np.linalg.norm(place_at[:2] - pick_at[:2]))
        print(f"{SHAKER:<14} picked up at ({pick_at[0]:+.3f},{pick_at[1]:+.3f}), "
              f"put back {back * 1000:.0f} mm away, lifted {lifted:.3f} m clear")
        if back > RETURN_TOL:
            failures.append(f"the shaker was put back {back:.3f} m from where it "
                            f"was picked up (limit {RETURN_TOL:.2f} m)")
    print(f"{'shake':<14} tipped to {np.degrees(tipped):.0f}deg, "
          f"{shake_travel * 1000:.0f} mm of travel, "
          f"{shake_reversals} direction changes over the plate")

    ended = rig.body_pos(SHAKER)
    drawer_floor = rig.routine.waypoint("shaker_grip")  # tracks the shut drawer
    if abs(ended[2] - drawer_floor[2]) > 0.05:
        failures.append(f"the shaker did not end up standing in the drawer "
                        f"(z={ended[2]:.3f}, drawer floor level "
                        f"{drawer_floor[2]:.3f})")
    if rig.tilt(SHAKER) > UPRIGHT_TOL:
        failures.append(f"the shaker ended up on its side, leaning "
                        f"{np.degrees(rig.tilt(SHAKER)):.0f}deg")

    for name in DRAWER_MATES:
        moved = float(np.linalg.norm(rig.body_pos(name) - rig.start[name]))
        lean = rig.tilt(name)
        print(f"{name:<14} still standing in the drawer, moved "
              f"{moved * 1000:.0f} mm, leaning {np.degrees(lean):.0f}deg")
        if lean > UPRIGHT_TOL:
            failures.append(f"{name} was knocked over "
                            f"({np.degrees(lean):.0f}deg from upright)")
    for name in (PLATE, WORKTOP_JAR, *FOOD):
        moved = float(np.linalg.norm(rig.body_pos(name) - rig.start[name]))
        print(f"{name:<14} moved {moved * 1000:.0f} mm")
        if moved > BYSTANDER_TOL:
            failures.append(f"{name} was disturbed: it moved {moved:.3f} m "
                            f"(limit {BYSTANDER_TOL:.2f} m)")

    if render:
        _write_media(rig, args)
    rig.close()

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll checks passed: nothing actuates the drawer or the shaker, the "
          "gripper opens the drawer by its handle, lifts the pepper shaker out, "
          "tips it cap-down over the plate and shakes it there, puts it back "
          "where it came from and pushes the drawer shut.")
    return 0


def _write_media(rig: Rig, args) -> None:
    import imageio.v3 as iio

    frames = np.asarray(rig.frames)
    if args.video is not None:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(args.video, frames, fps=args.fps, codec="libx264")
        print(f"\nwrote {args.video}  ({len(frames)} frames)")
    if args.sheet is not None:
        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        picks = ["pull the drawer open", "grip the shaker",
                 "shake the seasoning", "push the drawer shut"]
        idx = [rig.marks[p] for p in picks if p in rig.marks]
        tiles = [frames[i] for i in idx[:4]]
        while len(tiles) < 4:
            tiles.append(frames[-1])
        sheet = np.concatenate(
            [np.concatenate(tiles[0:2], axis=1),
             np.concatenate(tiles[2:4], axis=1)], axis=0)
        iio.imwrite(args.sheet, sheet[::2, ::2])
        print(f"wrote {args.sheet}")


if __name__ == "__main__":
    raise SystemExit(main())

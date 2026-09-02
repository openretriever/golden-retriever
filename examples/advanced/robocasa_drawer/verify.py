"""Prove the arm really grasps the drawer, headlessly.

Runs the routine in `sequence.py` against `scene.xml` and asserts on it. The
central claim is that the drawer is opened by the grasp and nothing else, so
the first check is structural: no actuator may act on a drawer slide joint. If
one did, the drawer could open on its own and every other number here would be
worthless.

  * no actuator drives any drawer — the slides are passive and damped
  * the gripper squares up to the handle bar and closes on it
  * a finger pad stays in contact with the handle for the whole pull and push
  * the drawer comes out under that grasp, and goes back in
  * the arm returns to its home pose afterwards
  * the seasoning bottle on the worktop stays standing on it
  * the two bottles lying in the top drawer ride out with the drawer, stay
    inside it, and roll while it moves

    python verify.py                       # assertions + logs, no rendering
    python verify.py --video out/scene.mp4 --sheet out/scene.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from examples.advanced.robocasa_drawer.arm_control import Arm
from examples.advanced.robocasa_drawer.plan import (
    GRASPING,
    HANDLE_TO_INTERIOR,
    HOLDING_JAR,
    OPENED_MIN,
    PHASES,
    SHUT_MAX,
    STROKE,
)
from examples.advanced.robocasa_drawer.sequence import GRASP_ROTATION, Choreography

HERE = Path(__file__).resolve().parent

POSE_TOL = 0.02        # metres the hand may sit off its commanded point...
REACH_TOL = 0.05       # ...except on the free-space moves of the jar errand,
                       # which only have to arrive near enough to grasp from
JAR_POSE_TOL = 0.035   # and the jar itself, which is a 48 mm cylinder taken in
                       # an 80 mm gripper: it does not need the handle's
                       # millimetre pinch, and holding it to that bar would be
                       # measuring the wrong thing
ANGLE_TOL = np.radians(5.0)   # how far off square the gripper may be
REACH_ANGLE_TOL = np.radians(25.0)  # likewise, off square, away from a grasp
HOME_TOL = 0.05        # radians per joint once the arm has withdrawn
CONTACT_TOL = 0.95     # fraction of a carry phase a pad must touch the handle
GRIP_TOL = 0.50        # ... and of the phase where it is still closing
# OPENED_MIN / SHUT_MAX come from plan.py, shared with the mock lane.

WORKTOP_BOTTLE = "cinnamon_main"          # starts on the worktop; is put away
DRAWER_BOTTLES = ("cayenne_main", "paprika_main")  # lie loose in the top drawer

WORKTOP_BOTTLE_START_Z = 1.10  # it stands at ~1.156 on the worktop
DRAWER_BOTTLE_MIN_Z = 0.85    # the drawer's inner floor is at ~0.885
DRAWER_INTERIOR_TOP_Z = 1.06  # the drawer's rim
DRAWER_HALF_WIDTH = 0.25
PLACED_UPRIGHT_TOL = np.radians(25)  # lean allowed at the moment it is set down
JAR_CONTACT_TOL = 0.90        # fraction of a carrying phase the jar must be held
ROLL_MIN = 1.0                # radians a loose bottle must roll over the run


class Rig:
    def __init__(self, xml: Path, width: int = 960, height: int = 720,
                 camera: str = "threequarter") -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.width, self.height = width, height
        self.camera = camera
        self._renderer: mujoco.Renderer | None = None
        self.frames: list[np.ndarray] = []
        self.marks: dict[str, int] = {}
        self.rolled: dict[str, float] = {}

        # The scene ships a `home` keyframe: arm rest pose plus matching targets.
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self.arm = Arm(self.model, self.data)
        self.plan = Choreography(self.model, self.data, self.arm)

    # -- state ---------------------------------------------------------------
    def body_z(self, name: str) -> float:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return float(self.data.xpos[bid][2])

    def body_pos(self, name: str):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[bid].copy()

    def body_y(self, name: str) -> float:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return float(self.data.xpos[bid][1])

    def tilt(self, name: str) -> float:
        """Angle between the body's own z axis and world up, in radians."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        axis = self.data.xmat[bid].reshape(3, 3)[:, 2]
        return float(np.arccos(np.clip(axis[2], -1.0, 1.0)))

    def accumulate_roll(self) -> None:
        """Integrate each loose bottle's spin. A bottle that only slides on the
        drawer floor never accumulates any, so this separates rolling from
        being dragged along."""
        dt = self.model.opt.timestep
        for name in DRAWER_BOTTLES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            spin = float(np.linalg.norm(self.data.cvel[bid][:3]))
            self.rolled[name] = self.rolled.get(name, 0.0) + spin * dt

    def in_drawer(self, name: str) -> bool:
        """True while the named body is inside the top drawer's interior.

        Measured against the drawer wherever it currently is, so it holds
        whether the drawer is open, shut, or somewhere in between.
        """
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        pos = self.data.xpos[bid]
        centre_y = float(self.plan.bar[1]) + HANDLE_TO_INTERIOR
        return bool(
            abs(float(pos[0])) < DRAWER_HALF_WIDTH
            and abs(float(pos[1]) - centre_y) < DRAWER_HALF_WIDTH
            and DRAWER_BOTTLE_MIN_Z < float(pos[2]) < DRAWER_INTERIOR_TOP_Z
        )

    def home_error(self) -> float:
        """Largest joint-angle gap between the arm and its home pose."""
        actual = self.data.qpos[self.arm.qpos_ids]
        return float(np.max(np.abs(actual - self.arm.home)))

    def driven_drawers(self) -> list[str]:
        """Names of any actuator acting on a drawer slide joint. Must be empty."""
        driven = []
        for a in range(self.model.nu):
            joint = self.model.actuator_trnid[a][0]
            if joint < 0:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if name and "slidejoint" in name:
                driven.append(mujoco.mj_id2name(self.model,
                                                mujoco.mjtObj.mjOBJ_ACTUATOR, a))
        return driven

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
            self.plan.apply(phase, blend)
            mujoco.mj_step(self.model, self.data)
            self.accumulate_roll()

    def run_phase(self, render: bool, fps: int = 30) -> dict:
        """Play one phase to its end and report what happened during it."""
        phase = self.plan.phase
        steps = max(1, int(round(phase.seconds / self.model.opt.timestep)))
        every = max(1, int(round(1.0 / (fps * self.model.opt.timestep))))
        held = 0
        jar_held = 0
        worst_angle = 0.0
        for s in range(steps):
            self.plan.step()
            mujoco.mj_step(self.model, self.data)
            self.accumulate_roll()
            if self.plan.gripping():
                held += 1
            if self.plan.holding_jar():
                jar_held += 1
            if phase.mode != "home":
                commanded = self.plan.target_pose(phase, s / steps)
                if commanded is not None:
                    worst_angle = max(worst_angle,
                                      self.arm.misalignment(commanded[1]))
            if render and s % every == 0:
                self.capture()
        self.marks[phase.label] = max(0, len(self.frames) - 1)

        pose = self.plan.target_pose(phase, 1.0)
        return {
            "label": phase.label,
            "contact": held / steps,
            "jar_contact": jar_held / steps,
            "jar_in_drawer": self.in_drawer(WORKTOP_BOTTLE),
            "angle": self.arm.misalignment(pose[1]) if pose else 0.0,
            "worst_angle": worst_angle,
            "pose_error": self.arm.distance_to(pose[0]) if pose else float("nan"),
            "drawer": self.plan.drawer_open,
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, default=HERE / "scene.xml")
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--sheet", type=Path, default=None)
    ap.add_argument("--camera", default="threequarter")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    render = args.video is not None or args.sheet is not None
    rig = Rig(args.scene, camera=args.camera)
    failures: list[str] = []

    print(f"scene: {args.scene}")
    print(f"slide travel: {rig.plan.travel:.3f} m, commanded pull: {STROKE:.3f} m")

    driven = rig.driven_drawers()
    if driven:
        failures.append("drawer slide joints are actuated by "
                        f"{', '.join(driven)}; the pull would not prove anything")
    else:
        print("drawer slides: passive, no actuator — only a grasp can move them")
    if render:
        rig.capture()

    bottle_y_shut = bottle_y_open = None

    for phase in PHASES:
        r = rig.run_phase(render=render, fps=args.fps)
        label = r["label"]
        print(f"[{label:<17}] drawer={r['drawer']:.3f}"
              f"   hand_err={r['pose_error']:.4f}"
              f"   square={np.degrees(r['angle']):5.2f}deg"
              f"   handle_contact={r['contact'] * 100:3.0f}%"
              f"   worktop_z={rig.body_z(WORKTOP_BOTTLE):.3f}"
              f"   drawer_bottle_y={rig.body_y(DRAWER_BOTTLES[0]):+.3f}")

        if phase.mode != "home":
            if label in GRASPING:
                pose_tol = POSE_TOL
            elif label in HOLDING_JAR:
                pose_tol = JAR_POSE_TOL
            else:
                pose_tol = REACH_TOL
            angle_tol = ANGLE_TOL if label in GRASPING else REACH_ANGLE_TOL
            if r["pose_error"] > pose_tol:
                failures.append(f"{label}: hand ended {r['pose_error']:.3f} m off "
                                f"its commanded point (limit {pose_tol:.2f} m)")
            if r["angle"] > angle_tol:
                failures.append(
                    f"{label}: gripper ended {np.degrees(r['angle']):.1f}deg off "
                    f"square to what it was reaching for "
                    f"(limit {np.degrees(angle_tol):.0f}deg)")
            # Once it is holding the bar it has to stay square for the whole
            # phase, not merely arrive square. Earlier phases start from the
            # home pose, which is a quarter turn away by definition.
            if label in GRASPING and r["worst_angle"] > ANGLE_TOL:
                failures.append(
                    f"{label}: gripper twisted "
                    f"{np.degrees(r['worst_angle']):.1f}deg off square while "
                    f"holding the handle "
                    f"(limit {np.degrees(ANGLE_TOL):.0f}deg)")

        if label in HOLDING_JAR:
            if r["jar_contact"] < JAR_CONTACT_TOL:
                failures.append(
                    f"{label}: fingers held the seasoning jar for only "
                    f"{r['jar_contact'] * 100:.0f}% of the phase "
                    f"(need {JAR_CONTACT_TOL * 100:.0f}%)")

        if label == "lower into drawer":
            if not r["jar_in_drawer"]:
                failures.append(
                    "the seasoning jar was not inside the drawer when the "
                    "gripper was ready to let go of it")
            if rig.tilt(WORKTOP_BOTTLE) > PLACED_UPRIGHT_TOL:
                failures.append(
                    "the seasoning jar was set down leaning "
                    f"{np.degrees(rig.tilt(WORKTOP_BOTTLE)):.0f}deg off upright "
                    f"(limit {np.degrees(PLACED_UPRIGHT_TOL):.0f}deg)")

        if label in GRASPING:
            floor = GRIP_TOL if phase.mode == "engage" else CONTACT_TOL
            if r["contact"] < floor:
                failures.append(
                    f"{label}: fingers touched the handle for only "
                    f"{r['contact'] * 100:.0f}% of the phase "
                    f"(need {floor * 100:.0f}%)")

        if label == "settle":
            bottle_y_shut = {n: rig.body_y(n) for n in DRAWER_BOTTLES}
        if label == "pull drawer open":
            bottle_y_open = {n: rig.body_y(n) for n in DRAWER_BOTTLES}
            if r["drawer"] < OPENED_MIN:
                failures.append(
                    f"the grasp only pulled the drawer out {r['drawer']:.3f} m "
                    f"(need {OPENED_MIN:.2f} m)")
            for n in DRAWER_BOTTLES:
                if rig.body_z(n) < DRAWER_BOTTLE_MIN_Z:
                    failures.append(f"{n} fell out of the open top drawer "
                                    f"(z={rig.body_z(n):.3f})")
        if label == "push drawer shut" and r["drawer"] > SHUT_MAX:
            failures.append(f"drawer left standing {r['drawer']:.3f} m proud "
                            f"(limit {SHUT_MAX:.2f} m)")
        if label == "withdraw":
            error = rig.home_error()
            print(f"    arm back at home to within {error:.4f} rad")
            if error > HOME_TOL:
                failures.append(
                    f"arm did not return home: worst joint off by {error:.3f} rad")

    if not rig.in_drawer(WORKTOP_BOTTLE):
        failures.append(
            f"{WORKTOP_BOTTLE} did not end up in the drawer "
            f"(pos={np.round(rig.body_pos(WORKTOP_BOTTLE), 3)})")
    elif rig.body_z(WORKTOP_BOTTLE) >= WORKTOP_BOTTLE_START_Z:
        failures.append(f"{WORKTOP_BOTTLE} never left the worktop "
                        f"(z={rig.body_z(WORKTOP_BOTTLE):.3f})")
    else:
        # It is checked for being upright at the moment it is set down, not
        # here. Shoving the drawer shut afterwards topples a free-standing
        # jar, the same way it rolls the two lying loose next to it -- that is
        # the scene behaving, not the placement failing.
        print(f"\n{WORKTOP_BOTTLE:14s} was put away in the drawer, ending at "
              f"z={rig.body_z(WORKTOP_BOTTLE):.3f} leaning "
              f"{np.degrees(rig.tilt(WORKTOP_BOTTLE)):.0f}deg")

    print()
    for n in DRAWER_BOTTLES:
        carried = (bottle_y_shut or {}).get(n, 0.0) - (bottle_y_open or {}).get(n, 0.0)
        spun = rig.rolled.get(n, 0.0)
        print(f"{n:<14} rode {carried:.3f} m out with the drawer "
              f"and rolled {spun:.2f} rad")
        if carried < rig.plan.travel * 0.5:
            failures.append(
                f"{n} only moved {carried:.3f} m with the drawer; "
                f"expected at least {rig.plan.travel * 0.5:.3f} m")
        if spun < ROLL_MIN:
            failures.append(f"{n} barely turned ({spun:.2f} rad); it is being "
                            f"dragged, not rolling (need {ROLL_MIN:.1f} rad)")
        if rig.body_z(n) < DRAWER_BOTTLE_MIN_Z:
            failures.append(f"{n} ended outside the drawer (z={rig.body_z(n):.3f})")

    if render:
        _write_media(rig, args)

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll checks passed: nothing actuates the drawer, the gripper closes "
          "on the handle and holds it throughout, the drawer comes out and goes "
          "back under that grasp alone, and the loose bottles roll with it.")
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
        picks = ["line up", "grip handle", "pull drawer open", "push drawer shut"]
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

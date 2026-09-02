"""Scripted pick-and-place in a real RoboCasa kitchen: counter -> drawer.

    python -m examples.advanced.robocasa_kitchen_mjviser.kitchen_pick_place \
        --render --out frames_kitchen --camera robot0_agentview_center

`PickPlaceCounterToDrawer` states its own errand — "Pick the whisk from the
counter and place it in the drawer" — and ships the props: a graspable object on
the counter (`obj_main`), a drawer fixture (`env.drawer`) with a slide joint, and
a distractor. All this adds is the motion, because the env has no policy.

Why this is written against the action space rather than joint targets
---------------------------------------------------------------------
The drawer-dash example drives position actuators it creates itself and does its
own IK. RoboCasa's PandaOmron is a `WheeledRobot` under a `HybridMobileBase`
composite controller with *torque* actuators (`robot0_torq_j1..7`), so writing
joint targets is not available. Instead every tick emits a 12-vector, whose
layout was established by probing one dimension at a time:

    0..2   end-effector translation, in the robot's frame:
           +a0 -> -x world, +a1 -> -y world, +a2 -> +z world
    3..5   end-effector rotation
    6      torso lift
    7..9   mobile base: forward, side, yaw
    10     gripper: +1 closes, -1 opens
    11     base mode switch (no effect here)

So the controller below is a proportional term on end-effector position with the
signs above, which is all a scripted demo needs — the OSC underneath does the
hard part.

STATUS: the pick works, the place does not. Read this before tweaking waypoints
-------------------------------------------------------------------------------
Reliably: the arm reaches the whisk, closes on it, and carries it 0.27 m across
to above the drawer — peak lift +0.204 m, repeatable.

It does not get it into the drawer, and the reason is geometric rather than a
matter of nudging numbers. A kitchen drawer sits UNDER a countertop, which is
the whole difference from the standalone dresser in the drawer-dash example:

* Lowering straight down onto the slot drives the whisk into the underside of
  the counter. It came to rest at z 0.946 three times running — x and y inside
  the interior region, z 87 mm high, which is exactly counter height.
* Opening the drawer further made it worse, not better: at 0.600 m the slot is
  beyond the arm's reach and the whisk was dropped even earlier.
* Approaching from the front and under the lip stalls: the end effector cannot
  reach that pose from where the base is parked, so the controller saturates and
  the whisk freezes mid-air for the rest of the run.

What this actually needs is either the mobile base repositioned so the drawer is
inside the arm's workspace (dims 7..9 are the base, currently unused here), or a
real motion planner instead of straight-line waypoints. Both are more than a
waypoint edit, which is why this stops here rather than guessing further.

Two things that cost time, recorded so they do not again
--------------------------------------------------------
* The drawer interior comes from `drawer.get_int_sites()`, not from arithmetic on
  `drawer.pos` / `drawer.size`. Guessing put the release point 14 cm out in y and
  10 cm high, and the whisk landed back on the counter.
* Placement is judged against that interior box on all three axes. Judging by
  "did z drop" is worthless here: the countertop and the open drawer's rim are
  within a few centimetres of each other, so the whisk can come to rest ON the
  rim and still pass a z test.
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import mujoco
import numpy as np
import robocasa  # noqa: F401  - registers the kitchen envs
import robosuite

VISUAL_GROUPS = (1, 2)
GRIP_SITE = "gripper0_right_grip_site"

# Translation signs, from the probe documented above.
AXIS_SIGN = np.array([-1.0, -1.0, +1.0])
GAIN = 6.0          # proportional gain, per metre of position error
CLIP = 0.9          # leave headroom; saturating all three axes fights the OSC


class Script:
    """A list of (label, target, grip, seconds) legs, run by a P controller."""

    def __init__(self, env) -> None:
        self.env = env
        self.model = env.sim.model._model
        self.data = env.sim.data._data
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
        self.obj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj_main")
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{env.drawer.name}_slidejoint")
        self.drawer_qpos = int(self.model.jnt_qposadr[jid]) if jid >= 0 else None
        self.legs = self._plan()

    @property
    def eef(self) -> np.ndarray:
        return self.data.site_xpos[self.site].copy()

    @property
    def obj_pos(self) -> np.ndarray:
        return self.data.xpos[self.obj].copy()

    @property
    def drawer_open(self) -> float:
        if self.drawer_qpos is None:
            return float("nan")
        # The kitchen's slide runs negative-open; report a positive extent.
        return abs(float(self.data.qpos[self.drawer_qpos]))

    def _plan(self):
        obj = self.obj_pos
        dr = self.env.drawer
        dpos = np.asarray(dr.pos, dtype=float)
        p0, px, py, pz = (np.asarray(v, dtype=float) for v in dr.get_int_sites()["int"])

        self.region_lo = dpos + np.array([min(p0[0], px[0]), min(p0[1], py[1]),
                                          min(p0[2], pz[2])])
        self.region_hi = dpos + np.array([max(p0[0], px[0]), max(p0[1], py[1]),
                                          max(p0[2], pz[2])])

        # The region describes the drawer FULLY open. This one is only
        # `drawer_open` metres out, so aim at the middle of the reachable front
        # slice rather than at the region centre.
        reach = min(self.drawer_open * 0.75, abs(py[1] - p0[1]) * 0.5)
        self.slot = dpos + np.array([
            (p0[0] + px[0]) / 2.0,          # centred across the width
            p0[1] + reach,                  # front slice; y runs p0 -> py
            min(p0[2], pz[2]) + 0.03,       # just above the drawer floor
        ])
        slot = self.slot

        # A kitchen drawer lives UNDER a countertop, which is the whole
        # difference from the standalone drawer-dash dresser. Lowering straight
        # down onto the slot drives the whisk into the underside of the counter:
        # it came to rest at z 0.946 every time, which is exactly counter height,
        # x and y correct and z 87 mm high. Opening the drawer further made it
        # worse, not better, because the slot then sits beyond the arm's reach.
        #
        # So approach from the front instead: out past the drawer face, down
        # below the counter lip, in over the open drawer, then release.
        front = slot + [0, -0.30, 0.10]     # clear of the face, below the lip
        over = slot + [0, 0, 0.06]          # inside, just above the floor
        return [
            ("settle",            self.eef,             -1.0, 0.6),
            ("above the whisk",   obj + [0, 0, 0.14],   -1.0, 2.0),
            ("down to the whisk", obj + [0, 0, 0.012],  -1.0, 1.6),
            ("close on it",       obj + [0, 0, 0.012],  +1.0, 1.0),
            ("lift clear",        obj + [0, 0, 0.24],   +1.0, 1.6),
            ("out past the face", front + [0, 0, 0.22], +1.0, 2.6),
            ("down below the lip", front,               +1.0, 2.0),
            ("in over the drawer", over,                +1.0, 2.2),
            ("lower in",          slot + [0, 0, 0.03],  +1.0, 1.6),
            ("let go",            slot + [0, 0, 0.03],  -1.0, 1.2),
            ("back out",          front,                -1.0, 1.8),
        ]

    def action(self, target, grip: float) -> np.ndarray:
        a = np.zeros(12)
        err = np.asarray(target, dtype=float) - self.eef
        a[:3] = np.clip(AXIS_SIGN * err * GAIN, -CLIP, CLIP)
        a[10] = grip
        return a

    def run(self, on_frame=None):
        hz = 1.0 / self.env.control_timestep
        track = []
        for label, target, grip, secs in self.legs:
            for _ in range(max(1, int(round(secs * hz)))):
                self.env.step(self.action(target, grip))
                track.append({
                    "phase": label,
                    "eef": np.round(self.eef, 4).tolist(),
                    "obj": np.round(self.obj_pos, 4).tolist(),
                    "drawer_open": round(self.drawer_open, 4),
                    "grip_cmd": grip,
                })
                if on_frame is not None:
                    on_frame(track[-1])
        return track


def write_png(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6))
                     + chunk(b"IEND", b""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickPlaceCounterToDrawer")
    ap.add_argument("--robot", default="PandaOmron")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--out", default="frames_kitchen")
    ap.add_argument("--camera", default="robot0_agentview_center")
    ap.add_argument("--width", type=int, default=1445)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    env = robosuite.make(env_name=args.task, robots=args.robot,
                         obj_registries=("lightwheel",), has_renderer=False,
                         has_offscreen_renderer=False, use_camera_obs=False,
                         control_freq=20, seed=args.seed)
    env.reset()
    script = Script(env)

    print(f"  errand: {env.get_ep_meta().get('lang')}")
    print(f"  drawer: {env.drawer.name}  open {script.drawer_open:.3f} m")
    print(f"  whisk at {np.round(script.obj_pos, 3)}, eef at {np.round(script.eef, 3)}")
    print(f"  release slot {np.round(script.slot, 3)}")

    renderer = None
    opt = None
    out = Path(args.out)
    if args.render:
        m = script.model
        m.vis.global_.offwidth = max(args.width, m.vis.global_.offwidth)
        m.vis.global_.offheight = max(args.height, m.vis.global_.offheight)
        renderer = mujoco.Renderer(m, height=args.height, width=args.width)
        opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(opt)
        for g in range(len(opt.geomgroup)):
            opt.geomgroup[g] = 1 if g in VISUAL_GROUPS else 0
        out.mkdir(parents=True, exist_ok=True)
        for stale in out.glob("*.png"):
            stale.unlink()

    n = {"i": 0}
    last = {"phase": None}

    def on_frame(rec) -> None:
        if rec["phase"] != last["phase"]:
            last["phase"] = rec["phase"]
            print(f"  {rec['phase']:<20} obj {rec['obj']}  drawer {rec['drawer_open']:.3f}",
                  flush=True)
        if renderer is not None:
            renderer.update_scene(script.data, camera=args.camera, scene_option=opt)
            write_png(out / f"k{n['i']:05d}.png", renderer.render())
            n["i"] += 1

    start = script.obj_pos.copy()
    track = script.run(on_frame=on_frame)
    end = script.obj_pos
    lifted = max(r["obj"][2] for r in track) - float(start[2])

    lo, hi = script.region_lo, script.region_hi
    inside = bool(np.all(end >= lo - 0.01) and np.all(end <= hi + 0.01))
    miss = np.maximum(np.maximum(lo - end, end - hi), 0.0)

    print("  --- result ---")
    print(f"  whisk {np.round(start, 3)} -> {np.round(end, 3)}, peak lift {lifted:+.3f} m")
    print(f"  interior x{np.round([lo[0], hi[0]], 3)} y{np.round([lo[1], hi[1]], 3)}"
          f" z{np.round([lo[2], hi[2]], 3)}")
    print(f"  lift:  {'ok' if lifted > 0.05 else 'FAILED - never left the counter'}")
    print("  place: " + ("IN THE DRAWER" if inside
                         else f"NOT in the drawer, off by {np.round(miss, 3)}"))
    if renderer is not None:
        print(f"  {n['i']} frames -> {out}")


if __name__ == "__main__":
    main()

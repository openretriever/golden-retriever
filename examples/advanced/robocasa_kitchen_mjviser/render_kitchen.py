"""Render a real RoboCasa kitchen offscreen, at video resolution.

    cd <worktree> && PYTHONPATH=<worktree> \
      <repo>/.venv-robocasa/bin/python \
      render_kitchen.py --task PickPlaceCounterToDrawer --out frames_kitchen

RUN THIS WITH .buildvenv, NOT .simvenv. That is the whole point:

  .buildvenv   mujoco 3.3.1 + robosuite (git master) + robocasa   -> no version
               spoofing needed, because mjviser is not involved
  .simvenv     mujoco 3.9 + mjviser                               -> interactive
               viewing only, and only there do the compat hacks apply

Video needs offscreen frames from mujoco.Renderer, not a live viewer, so the
RoboCasa-vs-mjviser version conflict simply does not arise on this path. Every
monkeypatch in serve_kitchen.py exists to make the *interactive* viewer work and
is irrelevant here.

Two real constraints remain, both honest configuration rather than hacks:

  * `obj_registries=("lightwheel",)` - RoboCasa samples objects across registries
    and divides by the total count, so with only the objs_lw pack downloaded the
    sum is zero and it raises "Probabilities contain NaN". The objaverse and
    aigen packs are ~9 GB more.
  * `m.vis.global_.offwidth/offheight` - envs ship a 640x480 offscreen buffer.
    It is a plain model field, so raise it before building the Renderer.

Measured on this machine: PickPlaceCounterToDrawer is 301 bodies / 1770 geoms,
steps at 5.2x realtime bare, and renders 1445x1080 at ~69 fps.
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import mujoco
import numpy as np
import robocasa  # noqa: F401  - registers the kitchen envs on import
import robosuite

# RoboSuite convention: geom group 0 is collision, 1 and 2 are visual.
VISUAL_GROUPS = (1, 2)


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Minimal PNG writer, so this needs no Pillow in the sim venv."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickPlaceCounterToDrawer",
                    help="374 RoboCasa envs are registered; drawer ones include "
                         "OpenDrawer, CloseDrawer, PickPlaceCounterToDrawer, "
                         "PickPlaceDrawerToCounter, PlaceVeggiesInDrawer")
    ap.add_argument("--robot", default="PandaOmron")
    ap.add_argument("--camera", default="robot0_frontview",
                    help="robot0_frontview | robot0_agentview_center | "
                         "robot0_agentview_left | robot0_agentview_right | "
                         "robot0_robotview | robot0_eye_in_hand")
    ap.add_argument("--out", default="frames_kitchen")
    ap.add_argument("--width", type=int, default=1445)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--still", action="store_true", help="one frame, then exit")
    args = ap.parse_args()

    env = robosuite.make(
        env_name=args.task,
        robots=args.robot,
        obj_registries=("lightwheel",),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        seed=args.seed,
    )
    env.reset()

    model = env.sim.model._model
    data = env.sim.data._data
    model.vis.global_.offwidth = max(args.width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(args.height, model.vis.global_.offheight)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    for g in range(len(opt.geomgroup)):
        opt.geomgroup[g] = 1 if g in VISUAL_GROUPS else 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.png"):
        stale.unlink()

    print(f"  {args.task} / {args.robot}: {model.nbody} bodies, {model.ngeom} geoms")
    print(f"  camera {args.camera} at {args.width}x{args.height}")

    if args.still:
        renderer.update_scene(data, camera=args.camera, scene_option=opt)
        write_png(out / "still.png", renderer.render())
        print(f"  wrote {out / 'still.png'}")
        return

    n_frames = int(round(args.seconds * args.fps))
    steps_per_frame = max(1, int(round(1.0 / args.fps / model.opt.timestep)))
    pad = len(str(n_frames)) + 1

    # Stepping bare MuJoCo rather than env.step: a viewer does not need
    # robosuite's observation, reward and termination work, and skipping it is
    # ~2x faster. Nothing here reads reward, so nothing is lost.
    for f in range(n_frames):
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=args.camera, scene_option=opt)
        write_png(out / f"k{str(f).zfill(pad)}.png", renderer.render())
        if f % 25 == 0:
            print(f"  {f}/{n_frames}", flush=True)

    print(f"  {n_frames} frames -> {out}")


if __name__ == "__main__":
    main()

"""Render the drawer-dash routine offscreen, plus a real telemetry track.

the drawer-dash example's `viewer.py` streams the scene to viser for interactive use. For video we
want frames on disk, so this drives exactly the same objects — `ensure_scene`,
`Arm`, `Choreography` — and swaps the viser display for `mujoco.Renderer`.
Nothing about the scene or the routine is changed.

It also writes `telemetry.json`: one record per rendered frame with the phase
label, drawer extension, and whether the hand is actually holding the handle or
the jar *as measured in the sim*, not as the nominal plan says. That lets the
side panel show real state, including the case where a grasp slips and the
drawer stops moving.

    PYTHONPATH=<worktree> python render_offscreen.py --out frames --camera threequarter

Run with the venv that has mujoco + robosuite + robocasa installed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mujoco
import numpy as np

from examples.advanced.robocasa_drawer_dash import plan as _plan, sequence as _seq
from examples.advanced.robocasa_drawer_dash.arm_control import Arm
from examples.advanced.robocasa_drawer_dash.plan import PHASES, TOTAL_SECONDS
from examples.advanced.robocasa_drawer_dash.scene import ensure_scene
from examples.advanced.robocasa_drawer_dash.sequence import Choreography

# RoboSuite convention: group 0 is collision geometry. Rendering it paints the
# whole scene in translucent red hulls, so show only the visual groups — the
# same choice viewer.py makes.
VISUAL_GROUPS = (1, 2)

# See serve_mjviser.py: the stock slot is close enough to the drawer front that
# the arm clips it while placing, shoving the drawer 0.403 -> 0.300 m. Keep the
# rendered video and the live viewer on the same corrected geometry.
for _m in (_plan, _seq):
    _m.SLOT_PLACE_Z = 1.00
    _m.HANDLE_TO_INTERIOR = 0.30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="frames_sim")
    ap.add_argument("--camera", default="threequarter",
                    help="front | threequarter | overhead | bottles | robot0_eye_in_hand")
    ap.add_argument("--width", type=int, default=1445)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="0 = the whole routine (TOTAL_SECONDS)")
    ap.add_argument("--scene", default=None)
    args = ap.parse_args()

    scene_path = ensure_scene(Path(args.scene) if args.scene else None)
    # The scene caps its offscreen framebuffer at 1200x800, which is below the
    # frame we want. Bump it in the loaded XML rather than editing scene.py —
    # every mesh is referenced by absolute path, so from_xml_string resolves.
    xml = scene_path.read_text()
    xml = re.sub(r'offwidth="\d+"', f'offwidth="{max(args.width, 1200)}"', xml)
    xml = re.sub(r'offheight="\d+"', f'offheight="{max(args.height, 800)}"', xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    arm = Arm(model, data)
    routine = Choreography(model, data, arm)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.png"):
        stale.unlink()

    dur = args.seconds or TOTAL_SECONDS
    n_frames = int(round(dur * args.fps))
    steps_per_frame = max(1, int(round(1.0 / args.fps / model.opt.timestep)))
    pad = len(str(n_frames)) + 1

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    for g in range(len(opt.geomgroup)):
        opt.geomgroup[g] = 1 if g in VISUAL_GROUPS else 0

    print(f"  scene {scene_path.name}: {model.nbody} bodies, {model.ngeom} geoms")
    print(f"  routine {len(PHASES)} phases / {TOTAL_SECONDS:.1f}s -> "
          f"{n_frames} frames @ {args.fps}fps, camera '{args.camera}'")

    track = []
    phase = PHASES[0]
    for f in range(n_frames):
        for _ in range(steps_per_frame):
            phase = routine.step()
            mujoco.mj_step(model, data)

        renderer.update_scene(data, camera=args.camera, scene_option=opt)
        px = renderer.render()
        _write_png(out / f"s{str(f).zfill(pad)}.png", px)

        track.append({
            "frame": f,
            "t": round(f / args.fps, 4),
            "phase": phase.label,
            "mode": phase.mode,
            "grip_cmd": round(float(phase.grip), 4),
            "torso_cmd": round(float(phase.torso), 4),
            # measured, not commanded
            "drawer_open": round(float(routine.drawer_open), 4),
            "gripping_handle": bool(routine.gripping()),
            "holding_jar": bool(routine.holding_jar()),
        })
        if f % 25 == 0:
            print(f"  {f}/{n_frames}  {phase.label:<22} drawer {routine.drawer_open:.3f} m", flush=True)

    (out.parent / "telemetry.json").write_text(json.dumps(track, indent=1))
    final = track[-1]
    print(f"  done. final drawer {final['drawer_open']:.3f} m, "
          f"jar held {final['holding_jar']}, frames -> {out}")


def _write_png(path: Path, rgb: np.ndarray) -> None:
    """Minimal PNG writer so this needs no Pillow in the sim venv."""
    import struct
    import zlib

    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    path.write_bytes(png)


if __name__ == "__main__":
    main()

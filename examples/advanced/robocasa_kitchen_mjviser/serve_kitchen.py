"""Serve a real RoboCasa kitchen through mjviser, live in a browser.

    PYTHONPATH=<worktree> python serve_kitchen.py --task OpenDrawer --port 8092

Why this can exist at all:

RoboCasa hard-asserts `mujoco == 3.3.1` and mjviser requires `mujoco >= 3.6`
(every one of its 14 releases), so on paper they cannot share an environment —
which is why the drawer-dash example talks to viser directly and why
`demo-robocasa-web --visualize mjviser` cannot run as written.

Measured, that pin is conservative: with robosuite installed from git master
(PyPI's build is missing `get_elements`, which RoboCasa needs), RoboCasa builds
and steps a full kitchen on mujoco 3.9 — 314 bodies, 1646 geoms for OpenDrawer.
So the assert is bypassed here deliberately, not accidentally.

The other gotcha: RoboCasa samples objects across several registries and divides
by the total count, so with only the `objs_lw` pack downloaded the sum is zero
and it dies with "Probabilities contain NaN". Restricting to `lightwheel` avoids
a ~9 GB download of the objaverse and aigen packs.
"""

from __future__ import annotations

import argparse

import mujoco

# RoboCasa's version assert fires at import. Neutralise it for the duration of
# the import only, then restore the truth so nothing downstream is misled.
_REAL_MUJOCO = mujoco.__version__
mujoco.__version__ = "3.3.1"
import numpy  # noqa: E402

_REAL_NUMPY = numpy.__version__
numpy.__version__ = "2.2.5"

import robocasa  # noqa: F401,E402
import robosuite  # noqa: E402

mujoco.__version__ = _REAL_MUJOCO
numpy.__version__ = _REAL_NUMPY

import numpy as np  # noqa: E402
import mjviser  # noqa: E402
import viser  # noqa: E402

VISUAL_GROUPS = [False, True, True, False, False, False]


def frame_on_visible_geometry(model, data, groups_visible):
    """Point the default camera at the room, not at a phantom bounding volume.

    MuJoCo derives stat.extent/center from EVERY geom. RoboCasa kitchens carry
    a lot that is never drawn - fully transparent placeholder fixtures, and
    robot0_base parked out at (10, 10, 0) - so the stats describe a volume
    several times larger than the visible room and centred outside it. mjviser
    frames its default camera at 3 * extent, which is why the kitchen arrives as
    a speck on a blank canvas. Recompute over exactly the geoms mjviser keeps.
    """
    mujoco.mj_forward(model, data)
    keep = [
        i for i in range(model.ngeom)
        if int(model.geom_group[i]) < len(groups_visible)
        and groups_visible[int(model.geom_group[i])]
        and model.geom_rgba[i, 3] != 0
        and int(model.geom_type[i]) != int(mujoco.mjtGeom.mjGEOM_PLANE)
    ]
    if not keep:
        return None
    # True world AABB per geom: 8 corners of the local AABB rotated into world.
    # geom_rbound (the bounding sphere) over-pads walls and floors enough to
    # double the extent on its own.
    aabb = model.geom_aabb[keep].reshape(-1, 6)
    ctr, half = aabb[:, :3], aabb[:, 3:]
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = ctr[:, None, :] + signs[None, :, :] * half[:, None, :]
    R = data.geom_xmat[keep].reshape(-1, 3, 3)
    world = np.einsum("nij,nkj->nki", R, corners) + data.geom_xpos[keep][:, None, :]
    flat = world.reshape(-1, 3)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    model.stat.center[:] = (lo + hi) / 2.0
    model.stat.extent = float(np.linalg.norm(hi - lo)) / 2.0
    return len(keep), model.stat.extent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="OpenDrawer",
                    help="any registered RoboCasa kitchen env, e.g. OpenDrawer, CloseDrawer")
    ap.add_argument("--robot", default="PandaOmron")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--layout", type=int, default=None)
    ap.add_argument("--style", type=int, default=None)
    ap.add_argument("--cam-back", type=float, default=2.2,
                    help="metres to dolly the camera back along its view axis")
    ap.add_argument("--cam-lift", type=float, default=0.5,
                    help="metres to raise the camera")
    ap.add_argument("--camera", default="robot0_frontview",
                    help="robot0_frontview | robot0_agentview_center | "
                         "robot0_agentview_left | robot0_agentview_right")
    args = ap.parse_args()

    kwargs = dict(
        env_name=args.task,
        robots=args.robot,
        # Only the lightwheel pack is on disk; see the module docstring.
        obj_registries=("lightwheel",),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )
    if args.layout is not None:
        kwargs["layout_ids"] = [args.layout]
    if args.style is not None:
        kwargs["style_ids"] = [args.style]

    env = robosuite.make(**kwargs)
    env.reset()

    # robosuite wraps MuJoCo; mjviser wants the raw handles underneath.
    model = env.sim.model._model
    data = env.sim.data._data
    zero = env.action_spec[0] * 0.0

    print(f"  task: {args.task}  robot: {args.robot}")
    print(f"  scene: {model.nbody} bodies, {model.ngeom} geoms  "
          f"(mujoco {_REAL_MUJOCO})")

    # mjviser's time budget assumes ONE step_fn call advances the sim by exactly
    # model.opt.timestep. `env.step` advances control_timestep — 25 model steps
    # at control_freq 20 — so calling it from step_fn overstates the work needed
    # by 25x: mjviser asks for 500 env.steps a second, never gets them, sits
    # permanently [CAPPED], and divides the true rate by 25 in the readout.
    # Measured: it reported 0.12x while the sim was really running at 3.1x.
    #
    # So step_fn advances exactly one model timestep and runs the controller on
    # the 20 Hz policy boundary, which is what env.step's inner loop does. The
    # resulting qpos trajectory is bit-identical to env.step (verified: max|dq|
    # = 0.0 over 1.5 s). mjviser then reports the truth, throttles itself to
    # 1.00x, and spends the surplus on rendering: 1.000x at a steady 60 FPS.
    SUBSTEPS = int(round(env.control_timestep / env.model_timestep))
    sim = env.sim
    steps = {"n": 0}

    def step_fn(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        # env.step's loop body, one substep at a time.
        sim.step1()
        env._pre_action(zero, policy_step=(steps["n"] % SUBSTEPS) == 0)
        sim.step2()
        steps["n"] += 1

    def reset_fn(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        env.reset()
        steps["n"] = 0

    # mjviser builds a slider per actuator AND per limited joint, deriving the
    # step from the range — then rounds the bounds to 3 decimals. RoboCasa
    # kitchens carry both kinds with degenerate ranges (min == max), which makes
    # step 0 and viser refuses the slider. Widen by more than the rounding, or
    # the widening is rounded straight back away.
    EPS = 0.01
    fixed = {"act": 0, "jnt": 0}
    for i in range(model.nu):
        lo, hi = model.actuator_ctrlrange[i]
        if hi - lo < EPS:
            model.actuator_ctrlrange[i] = (lo - EPS, hi + EPS)
            fixed["act"] += 1
    for j in range(model.njnt):
        lo, hi = model.jnt_range[j]
        if model.jnt_limited[j] and hi - lo < EPS:
            model.jnt_range[j] = (lo - EPS, hi + EPS)
            fixed["jnt"] += 1
    print(f"  widened {fixed['act']} actuator and {fixed['jnt']} joint range(s) for the GUI")

    before = float(model.stat.extent)
    framed = frame_on_visible_geometry(model, data, VISUAL_GROUPS)
    if framed:
        print(f"  framing: extent {before:.1f} -> {framed[1]:.1f} m "
              f"over {framed[0]} visible geoms")

    server = viser.ViserServer(port=args.port)
    viewer = mjviser.Viewer(model, data, step_fn=step_fn, reset_fn=reset_fn, server=server)

    # RoboSuite convention: geom group 0 is collision, 1 and 2 are visual.
    # mjviser shows 0-2 and draws convex hulls, which paints everything in
    # translucent red; show the visual groups only.
    viewer.scene.geom_groups_visible = VISUAL_GROUPS
    viewer.scene.show_convex_hull = False
    viewer.scene.rebuild_visual_handles()
    viewer.scene.request_update()

    # Framing the room still leaves the free camera OUTSIDE its walls, staring at
    # the back of one. The env ships its own cameras; drop the client camera onto
    # one of them so a fresh browser lands looking at the counter.
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    if cam_id >= 0:
        mujoco.mj_forward(model, data)
        # MuJoCo cameras look down their local -Z.
        fwd = -np.array(data.cam_xmat[cam_id], dtype=float).reshape(3, 3)[:, 2]
        # The env's agentview sits right on top of the robot. Dolly back along
        # its own view axis (and lift a little) so the counter, the drawers and
        # the arm are all in frame.
        eye = np.array(data.cam_xpos[cam_id], dtype=float) - fwd * args.cam_back
        eye[2] += args.cam_lift
        target = np.array(data.cam_xpos[cam_id], dtype=float) + fwd * 1.2

        import threading as _th
        import time as _t

        @server.on_client_connect
        def _place(client) -> None:
            # The client's camera is not ready the instant the socket opens, and
            # mjviser's own "Track camera" writes to it too. Set it a moment
            # later, twice, so ours is the one that sticks.
            def _later() -> None:
                for delay in (0.4, 1.2):
                    _t.sleep(delay)
                    try:
                        client.camera.up = (0.0, 0.0, 1.0)
                        client.camera.position = tuple(eye)
                        client.camera.look_at = tuple(target)
                    except Exception:
                        pass
            _th.Thread(target=_later, daemon=True).start()

        print(f"  camera '{args.camera}' at {np.round(eye, 2)} -> {np.round(target, 2)}")
    else:
        print(f"  camera '{args.camera}' not found; leaving the default free camera")

    with server.gui.add_folder("retriever"):
        server.gui.add_text("task", args.task, disabled=True)
        server.gui.add_text("robot", args.robot, disabled=True)
        server.gui.add_text("scene", f"{model.nbody} bodies", disabled=True)

    print(f"  KITCHEN URL: http://localhost:{server.get_port()}", flush=True)
    viewer.run()


if __name__ == "__main__":
    main()

"""Serve the drawer-dash scene to a browser, live.

Runs under plain `python` — no `mjpython`, no native window — and streams the
simulation to a viser page:

    pixi run demo-drawer-dash-viewer            # http://localhost:8087
    pixi run demo-drawer-dash-viewer --port 9000
    pixi run demo-drawer-dash-viewer --hold     # start with the routine paused

The "grasp" panel reports what the arm is doing, how far the drawer has come
out, and whether a finger pad is touching the handle right now. Nothing drives
the drawer, so that last readout is the whole story: when it says no, the
drawer stops moving. The last line counts how far the two loose seasoning
bottles in the drawer have rolled — they are not attached to anything either.

The scene is built on first run if it is not there yet, so a fresh clone with
the asset packs installed needs only this one command.

This talks to viser directly rather than through `mjviser`, which requires
`mujoco>=3.6`: RoboSuite's controllers do not survive the 3.10 `mj_fullM`
signature change, so this example is pinned to 3.3.1 and cannot take that
dependency.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import viser

from examples.advanced.robocasa_drawer_dash.arm_control import Arm
from examples.advanced.robocasa_drawer_dash.plan import OPEN, PHASES, TOTAL_SECONDS
from examples.advanced.robocasa_drawer_dash.scene import ensure_scene
from examples.advanced.robocasa_drawer_dash.sequence import Choreography

RENDER_FPS = 30.0
# RoboSuite convention: group 0 is collision, 1 and 2 are visual. Drawing
# group 0 paints the whole scene in translucent red hulls.
VISUAL_GROUPS = (1, 2)
ROLLING_BOTTLES = ("cayenne_main", "paprika_main")


def _box_mesh(size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = size
    verts = np.array([
        [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
        [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
    ], dtype=np.float32)
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
    ], dtype=np.uint32)
    return verts, faces


def _cylinder_mesh(radius: float, half_height: float, segments: int = 24):
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    lower = np.hstack([ring, np.full((segments, 1), -half_height)])
    upper = np.hstack([ring, np.full((segments, 1), half_height)])
    verts = np.vstack([lower, upper,
                       [[0.0, 0.0, -half_height]], [[0.0, 0.0, half_height]]])
    bottom_hub, top_hub = 2 * segments, 2 * segments + 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces += [[i, j, segments + j], [i, segments + j, segments + i]]
        faces.append([bottom_hub, j, i])
        faces.append([top_hub, segments + i, segments + j])
    return verts.astype(np.float32), np.array(faces, dtype=np.uint32)


def _plane_mesh(size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # A MuJoCo plane is infinite; size[0:2] is only its drawn extent, and 0
    # means "unbounded", which still has to be drawn as something finite.
    x = float(size[0]) or 8.0
    y = float(size[1]) or 8.0
    verts = np.array([[-x, -y, 0.0], [x, -y, 0.0], [x, y, 0.0], [-x, y, 0.0]],
                     dtype=np.float32)
    return verts, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)


def _mesh_asset(model: mujoco.MjModel, geom: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = model.geom_dataid[geom]
    va, vn = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    fa, fn = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
    verts = np.asarray(model.mesh_vert[va:va + vn], dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(model.mesh_face[fa:fa + fn], dtype=np.uint32).reshape(-1, 3)
    return verts, faces


def _geom_mesh(model: mujoco.MjModel, geom: int):
    """Triangles for one geom in its own frame, or None if it is not drawable."""
    kind = model.geom_type[geom]
    size = np.asarray(model.geom_size[geom], dtype=float)
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        return _mesh_asset(model, geom)
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return _box_mesh(size)
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return _cylinder_mesh(float(size[0]), float(size[1]))
    if kind == mujoco.mjtGeom.mjGEOM_PLANE:
        return _plane_mesh(size)
    return None


def _geom_colour(model: mujoco.MjModel, geom: int) -> tuple[int, int, int]:
    rgba = np.asarray(model.geom_rgba[geom], dtype=float)
    material = model.geom_matid[geom]
    # A geom that leaves rgba at the default white defers to its material;
    # most of the scanned RoboCasa meshes are coloured that way.
    if material >= 0 and np.allclose(rgba[:3], 1.0):
        rgba = np.asarray(model.mat_rgba[material], dtype=float)
    return tuple(int(round(255 * float(c))) for c in np.clip(rgba[:3], 0.0, 1.0))


def build_scene_handles(server: viser.ViserServer, model: mujoco.MjModel) -> list:
    """Push every visual geom to the browser once. Returns (geom id, handle)."""
    handles = []
    for geom in range(model.ngeom):
        if model.geom_group[geom] not in VISUAL_GROUPS:
            continue
        mesh = _geom_mesh(model, geom)
        if mesh is None:
            continue
        verts, faces = mesh
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or f"geom_{geom}"
        handle = server.scene.add_mesh_simple(
            f"/geoms/{geom}_{name}",
            vertices=verts,
            faces=faces,
            color=_geom_colour(model, geom),
            opacity=float(model.geom_rgba[geom][3]),
        )
        handles.append((geom, handle))
    return handles


def _mat_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=float).flatten())
    return quat


def sync_handles(handles: list, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Move each geom to wherever the physics has put it."""
    for geom, handle in handles:
        handle.position = tuple(float(v) for v in data.geom_xpos[geom])
        handle.wxyz = tuple(float(v) for v in _mat_to_wxyz(data.geom_xmat[geom]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the drawer-dash scene to a browser via viser.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scene", type=Path, default=None,
                        help="Path to scene.xml; built on first run if absent.")
    parser.add_argument("--port", type=int, default=8087)
    parser.add_argument("--hold", action="store_true",
                        help="Start with the routine paused, arm at its home pose.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scene_path = ensure_scene(args.scene)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    arm = Arm(model, data)
    routine = Choreography(model, data, arm)

    server = viser.ViserServer(port=args.port)
    handles = build_scene_handles(server, model)
    sync_handles(handles, model, data)

    spun = 0.0
    with server.gui.add_folder("grasp"):
        running = server.gui.add_checkbox("run routine", not args.hold)
        stage = server.gui.add_text("step", PHASES[0].label, disabled=True)
        opened = server.gui.add_text("drawer out", "0.000 m", disabled=True)
        holding = server.gui.add_text("holding handle", "no", disabled=True)
        rolled = server.gui.add_text("bottles rolled", "0.0 rad", disabled=True)
        restart = server.gui.add_button("restart routine")

    @restart.on_click
    def _(_event) -> None:
        nonlocal spun
        spun = 0.0
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        routine.reset()

    print(f"scene: {scene_path}")
    print(f"slide travel: {routine.travel:.3f} m, drawers are passive "
          f"(no actuator) — only the grasp moves them")
    print(f"routine: {len(PHASES)} phases, {TOTAL_SECONDS:.0f} s per cycle")
    print(f"drawer dash: http://localhost:{args.port}")

    bottles = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
               for n in ROLLING_BOTTLES]
    steps_per_frame = max(1, int(round(1.0 / RENDER_FPS / model.opt.timestep)))
    next_frame = time.time()
    phase = PHASES[0]

    while True:
        auto = running.value
        for _ in range(steps_per_frame):
            if auto:
                phase = routine.step()
            else:
                arm.go_home()
                arm.set_gripper(OPEN)
            mujoco.mj_step(model, data)
            spun += sum(float(np.linalg.norm(data.cvel[b][:3]))
                        for b in bottles) * model.opt.timestep

        # Refresh the readouts once a frame, not once a physics tick — every
        # assignment is a message to the browser.
        label = phase.label if auto else "paused"
        if stage.value != label:
            stage.value = label
        opened.value = f"{routine.drawer_open:.3f} m"
        holding.value = "yes" if routine.gripping() else "no"
        rolled.value = f"{spun:.1f} rad"

        sync_handles(handles, model, data)
        next_frame += 1.0 / RENDER_FPS
        time.sleep(max(0.0, next_frame - time.time()))


if __name__ == "__main__":
    main()

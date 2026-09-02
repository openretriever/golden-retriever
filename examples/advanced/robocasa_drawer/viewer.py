"""Serve the drawer scene to a browser, live.

Runs under plain `python` — no `mjpython`, no native window — and streams the
simulation to a viser page:

    pixi run demo-drawer                 # opens http://localhost:8087
    pixi run demo-drawer -- --port 9000
    pixi run demo-drawer -- --hold       # start with the routine paused
    pixi run demo-drawer -- --no-open    # serve without opening a browser
    pixi run demo-drawer -- --flat       # flat colours, no textures

Geoms whose material carries a colour map are sent as textured glTF, so the
wood, the labelled jars and the food arrive looking like themselves; the rest
are sent as flat meshes in their material colour. `--flat` skips the textures
altogether, which starts faster and is the fallback if a scan will not map.

The "routine" panel reports what the arm is doing, how far the drawer has come
out, and whether a finger pad is touching the handle or the pepper shaker right
now. Nothing drives the drawer or the shaker, so those two readouts are the
whole story: when they say no, nothing moves. The last line is how far the
shaker is tipped from upright, which passes 90 degrees only while it is being
shaken over the plate.

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
import webbrowser
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser
from PIL import Image

from examples.advanced.robocasa_drawer.arm_control import Arm
from examples.advanced.robocasa_drawer.plan import OPEN, PHASES, TOTAL_SECONDS
from examples.advanced.robocasa_drawer.scene import ensure_scene
from examples.advanced.robocasa_drawer.sequence import Choreography

RENDER_FPS = 30.0
# RoboSuite convention: group 0 is collision, 1 and 2 are visual. Drawing
# group 0 paints the whole scene in translucent red hulls.
VISUAL_GROUPS = (1, 2)
# RoboCasa ships 2048-pixel scans. Each one is re-encoded into a glB and pushed
# down a websocket at start-up, so cap what actually crosses the wire.
TEXTURE_MAX_PX = 512
# What the MuJoCo compiler leaves on a geom that never mentions a colour.
DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)


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


def _effective_rgba(model: mujoco.MjModel, geom: int) -> np.ndarray:
    rgba = np.asarray(model.geom_rgba[geom], dtype=float)
    material = model.geom_matid[geom]
    # MuJoCo's own renderer lets the material win unless the geom overrides it,
    # and a geom that never mentions rgba keeps the compiler's default grey.
    # Deferring only on white left almost the whole kitchen flat grey.
    if material >= 0 and (np.allclose(rgba[:3], 1.0) or np.allclose(rgba, DEFAULT_RGBA)):
        rgba = np.asarray(model.mat_rgba[material], dtype=float)
    return np.clip(rgba, 0.0, 1.0)


def _geom_colour(model: mujoco.MjModel, geom: int) -> tuple[int, int, int]:
    return tuple(int(round(255 * float(c))) for c in _effective_rgba(model, geom)[:3])


def _texture_image(model: mujoco.MjModel, texid: int) -> Image.Image | None:
    """A material's colour texture as an image, or None if it is not one."""
    kind = model.tex_type[texid]
    if kind == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
        return None
    width, height = int(model.tex_width[texid]), int(model.tex_height[texid])
    channels = int(model.tex_nchannel[texid])
    start = int(model.tex_adr[texid])
    flat = np.asarray(model.tex_data[start:start + width * height * channels],
                      dtype=np.uint8)
    pixels = flat.reshape(height, width, channels)[:, :, :3]
    if kind == mujoco.mjtTexture.mjTEXTURE_CUBE and height == 6 * width:
        # Six faces stacked into one column. The browser gets one flat
        # material per geom, so the first face stands in for the cube.
        pixels = pixels[:width]
    picture = Image.fromarray(pixels)
    picture.thumbnail((TEXTURE_MAX_PX, TEXTURE_MAX_PX))
    return picture


def _geom_texture(model: mujoco.MjModel, geom: int):
    """(texture id, repeat, uniform) for a geom's colour map, or None."""
    material = int(model.geom_matid[geom])
    if material < 0:
        return None
    slots = np.atleast_1d(np.asarray(model.mat_texid[material]))
    role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    texid = int(slots[role] if slots.size > role else slots[0])
    if texid < 0:
        return None
    return (texid,
            np.asarray(model.mat_texrepeat[material], dtype=float),
            bool(model.mat_texuniform[material]))


def _mesh_uv(model: mujoco.MjModel, geom: int, verts: np.ndarray, faces: np.ndarray):
    """Per-vertex texture coordinates, splitting the mesh apart if it needs it."""
    mesh = int(model.geom_dataid[geom])
    count = int(model.mesh_texcoordnum[mesh])
    if count == 0:
        return None, verts, faces
    start = int(model.mesh_texcoordadr[mesh])
    uv = np.asarray(model.mesh_texcoord[start:start + count],
                    dtype=np.float32).reshape(-1, 2)
    fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
    corners = np.asarray(model.mesh_facetexcoord[fa:fa + fn]).reshape(-1, 3)
    if uv.shape[0] != verts.shape[0] or not np.array_equal(corners, faces):
        # A vertex whose faces read different texcoords cannot carry a single
        # uv, so hand the browser independent triangles instead.
        verts = verts[faces.reshape(-1)]
        uv = uv[corners.reshape(-1)]
        faces = np.arange(verts.shape[0], dtype=np.uint32).reshape(-1, 3)
    return uv, verts, faces


def _panel_uv(across: np.ndarray, extent: float, repeat: float, uniform: bool):
    # texuniform means "repeats per metre"; otherwise it is repeats per side.
    if uniform:
        return across * repeat
    return (across / extent + 0.5) * repeat


def _box_uv(size: np.ndarray, repeat: np.ndarray, uniform: bool):
    """An unwelded box: one texture panel per face, so the six can differ."""
    verts, uvs, faces = [], [], []
    for axis in range(3):
        u_axis, v_axis = (axis + 1) % 3, (axis + 2) % 3
        for sign in (1.0, -1.0):
            quad = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
            if sign < 0:
                quad = quad[::-1]
            base = len(verts)
            for su, sv in quad:
                corner = np.zeros(3)
                corner[axis] = sign * size[axis]
                corner[u_axis] = su * size[u_axis]
                corner[v_axis] = sv * size[v_axis]
                verts.append(corner)
                uvs.append([
                    _panel_uv(corner[u_axis], 2.0 * size[u_axis], repeat[0], uniform),
                    _panel_uv(corner[v_axis], 2.0 * size[v_axis], repeat[1], uniform),
                ])
            faces += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    return (np.asarray(verts, dtype=np.float32),
            np.asarray(faces, dtype=np.uint32),
            np.asarray(uvs, dtype=np.float32))


def _plane_uv(size: np.ndarray, repeat: np.ndarray, uniform: bool):
    verts, faces = _plane_mesh(size)
    uv = np.column_stack([
        _panel_uv(verts[:, 0], 2.0 * (float(size[0]) or 8.0), repeat[0], uniform),
        _panel_uv(verts[:, 1], 2.0 * (float(size[1]) or 8.0), repeat[1], uniform),
    ]).astype(np.float32)
    return verts, faces, uv


def _textured_mesh(model: mujoco.MjModel, geom: int, images: dict):
    """A trimesh wearing the geom's texture, or None if it has none to wear."""
    texture = _geom_texture(model, geom)
    if texture is None:
        return None
    texid, repeat, uniform = texture
    if texid not in images:
        images[texid] = _texture_image(model, texid)
    picture = images[texid]
    if picture is None:
        return None

    kind = model.geom_type[geom]
    size = np.asarray(model.geom_size[geom], dtype=float)
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        verts, faces = _mesh_asset(model, geom)
        uv, verts, faces = _mesh_uv(model, geom, verts, faces)
        if uv is None:
            return None
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        verts, faces, uv = _box_uv(size, repeat, uniform)
    elif kind == mujoco.mjtGeom.mjGEOM_PLANE:
        verts, faces, uv = _plane_uv(size, repeat, uniform)
    else:
        return None

    # MuJoCo hands out texcoords in glTF's own top-left convention, but
    # trimesh flips V as it writes the glB. Pre-flip to cancel that out, or
    # every label arrives upside down.
    uv = np.column_stack([uv[:, 0], 1.0 - uv[:, 1]]).astype(np.float32)

    tint = _effective_rgba(model, geom)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=picture,
        baseColorFactor=tint,
        # glTF ignores alpha unless the material says to blend, so the smoked
        # glass on the jars would otherwise hide the labels behind it.
        alphaMode="BLEND" if tint[3] < 1.0 else "OPAQUE",
        # glTF also defaults to a fully metallic surface, which turns every one
        # of these scans into a near-black mirror under viser's lighting.
        metallicFactor=0.0,
        roughnessFactor=0.8,
    )
    return trimesh.Trimesh(
        vertices=verts, faces=faces, process=False,
        visual=trimesh.visual.TextureVisuals(uv=uv, material=material),
    )


def build_scene_handles(server: viser.ViserServer, model: mujoco.MjModel,
                        textured: bool = True) -> list:
    """Push every visual geom to the browser once. Returns (geom id, handle)."""
    handles = []
    images: dict[int, Image.Image | None] = {}
    for geom in range(model.ngeom):
        if model.geom_group[geom] not in VISUAL_GROUPS:
            continue
        rgba = _effective_rgba(model, geom)
        if rgba[3] == 0.0:
            continue  # invisible in MuJoCo too; drawing it only adds clutter
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or f"geom_{geom}"
        node = f"/geoms/{geom}_{name}"
        dressed = _textured_mesh(model, geom, images) if textured else None
        if dressed is not None:
            handles.append((geom, server.scene.add_mesh_trimesh(node, dressed)))
            continue
        mesh = _geom_mesh(model, geom)
        if mesh is None:
            continue
        verts, faces = mesh
        handle = server.scene.add_mesh_simple(
            node,
            vertices=verts,
            faces=faces,
            color=_geom_colour(model, geom),
            # The clear hull round a jar carries its transparency on the
            # material, not the geom; reading the geom alone drew it solid.
            opacity=float(rgba[3]),
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
        description="Serve the RoboCasa drawer scene to a browser via viser.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scene", type=Path, default=None,
                        help="Path to scene.xml; built on first run if absent.")
    parser.add_argument("--port", type=int, default=8087)
    parser.add_argument("--hold", action="store_true",
                        help="Start with the routine paused, arm at its home pose.")
    parser.add_argument("--no-open", dest="open_browser", action="store_false",
                        help="Do not open a browser; just serve and print the URL.")
    parser.add_argument("--flat", action="store_true",
                        help="Skip the textures and draw flat material colours.")
    parser.set_defaults(open_browser=True)
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
    started = time.time()
    handles = build_scene_handles(server, model, textured=not args.flat)
    sync_handles(handles, model, data)
    print(f"pushed {len(handles)} geoms in {time.time() - started:.1f} s"
          + (" (flat colours)" if args.flat else ""))

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # A fresh page otherwise opens metres out with the cabinet facing away.
        # Park every arriving browser on the scene's own "action" camera, the
        # one the recordings use, so the first frame is the one worth seeing.
        cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "action")
        if cam < 0:
            return
        pos = np.asarray(data.cam_xpos[cam], dtype=float)
        axes = np.asarray(data.cam_xmat[cam], dtype=float).reshape(3, 3)
        client.camera.fov = float(np.radians(model.cam_fovy[cam]))
        client.camera.position = tuple(float(v) for v in pos)
        client.camera.up_direction = tuple(float(v) for v in axes[:, 1])
        # A MuJoCo camera looks down its own -Z; viser wants a point to aim at.
        client.camera.look_at = tuple(float(v) for v in pos - 2.0 * axes[:, 2])

    with server.gui.add_folder("routine"):
        running = server.gui.add_checkbox("run routine", not args.hold)
        stage = server.gui.add_text("step", PHASES[0].label, disabled=True)
        opened = server.gui.add_text("drawer out", "0.000 m", disabled=True)
        holding = server.gui.add_text("holding handle", "no", disabled=True)
        carrying = server.gui.add_text("holding shaker", "no", disabled=True)
        tipped = server.gui.add_text("shaker tipped", "0 deg", disabled=True)
        restart = server.gui.add_button("restart routine")

    @restart.on_click
    def _(_event) -> None:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        routine.reset()

    print(f"scene: {scene_path}")
    print(f"slide travel: {routine.travel:.3f} m, drawers are passive "
          f"(no actuator) — only the grasp moves them")
    print(f"routine: {len(PHASES)} phases, {TOTAL_SECONDS:.0f} s per cycle")
    # Ask viser what it actually bound to. If the requested port is taken it
    # quietly serves on the next free one, and announcing the requested port
    # instead sends you to whatever else is already listening there.
    port = server.get_port()
    url = f"http://localhost:{port}"
    if port != args.port:
        print(f"port {args.port} was already in use; serving on {port} instead")
    print(f"drawer + seasoning scene: {url}")
    if args.open_browser:
        # This lane exists to be looked at, so opening the page is the
        # default; `--no-open` is there for headless and CI use.
        webbrowser.open(url)

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

        # Refresh the readouts once a frame, not once a physics tick — every
        # assignment is a message to the browser.
        label = phase.label if auto else "paused"
        if stage.value != label:
            stage.value = label
        opened.value = f"{routine.drawer_open:.3f} m"
        holding.value = "yes" if routine.gripping() else "no"
        carrying.value = "yes" if routine.holding() else "no"
        tipped.value = f"{np.degrees(routine.tip()):.0f} deg"

        sync_handles(handles, model, data)
        next_frame += 1.0 / RENDER_FPS
        time.sleep(max(0.0, next_frame - time.time()))


if __name__ == "__main__":
    main()

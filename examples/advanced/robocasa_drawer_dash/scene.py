"""Build the MuJoCo scene: a Panda arm facing a three-drawer dresser on a table.

The drawers are real RoboCasa `Drawer` fixtures: a procedurally sized carcass with
a prismatic (slide) joint, fronted by a scanned mesh door panel and a scanned mesh
handle from the RoboCasa / Lightwheel fixture library (free, Apache-2.0, downloaded
with `robocasa.scripts.download_kitchen_assets --type fixtures_lw`).

The dresser stands on a table, and a PandaOmron — the Franka Panda on an Omron
mobile base that RoboCasa's own kitchen tasks drive — faces it across the floor.

The drawer slide joints carry no actuator on purpose. They are passive and
damped, so the only way a drawer opens here is if the gripper takes hold of its
handle and pulls.

Running this module writes `scene.xml` next to it, self-contained apart from the
mesh files it references by absolute path inside the robocasa asset tree. The
written file carries a `home` keyframe holding both the arm's rest pose and the
matching actuator targets, so every consumer starts from the same configuration:

    mujoco.mj_resetDataKeyframe(model, data, 0)
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent

# One drawer per row, bottom to top. Sizes are (width, depth, height) in metres.
DRAWER_W = 0.50
DRAWER_D = 0.45
DRAWER_H = 0.22
NUM_DRAWERS = 3

# Mesh panel + handle picked from the RoboCasa fixture library.
PANEL_TYPE = "CabinetDoorPanel009"
HANDLE_TYPE = "CabinetHandle012"

TEXTURE = "textures/wood/light_wood_planks.png"

# Seasoning bottles, from the RoboCasa object library: scanned spice jars with
# their own textures and convex collision meshes. Each is a squat cylinder, so
# one laid on its side rolls the moment the surface under it accelerates.
def _object_root() -> Path:
    """Where the RoboCasa object library is installed, resolved on demand."""
    import robocasa

    return Path(robocasa.__file__).resolve().parent / "models" / "assets" / "objects"
BOTTLE_ASSETS = {
    "cinnamon": "lightwheel/cinnamon/Cinnamon001",     # brown "CINNAMON" jar
    "cayenne": "lightwheel/paprika/Paprika001",        # tall red cayenne jar
    "paprika": "lightwheel/cinnamon/Cinnamon003",       # amber cinnamon jar
}
# The library ships these with rolling friction of 0.1, which pins a bottle to
# the spot it lands on. Dropped to a tenth of that they actually roll.
BOTTLE_FRICTION = (0.7, 0.02, 0.001)
BOTTLE_DENSITY = 260.0   # a part-full glass spice jar, ~90 g
# Lying on its side, long axis along world x: a quarter turn about y.
ON_ITS_SIDE = (0.70710678, 0.0, 0.70710678, 0.0)

# The table the dresser stands on. Its top surface is TABLE_H above the floor,
# which sets every drawer handle into the Panda's comfortable working band.
TABLE_H = 0.40
TABLE_W = 1.30
TABLE_D = 0.86
TABLE_TOP_T = 0.025  # half-thickness of the top slab
LEG_R = 0.032
LEG_INSET = 0.07

# Mobile base, on the floor, set off to one side so it does not stand between
# the camera and the dresser. It is yawed to face the drawer fronts.
ROBOT_POS = (-0.35, -0.88, 0.0)
HANDLE_FACE = (0.0, -0.237)  # where the drawer handles sit, for aiming the base

# Rest pose, joint 1 through 7 — RoboCasa's own home pose for this robot.
def _arm_home() -> tuple[float, ...]:
    """RoboCasa's own rest pose for this robot, joint 1 through 7."""
    from robosuite.models.robots import PandaOmron

    return tuple(PandaOmron(idn=0).init_qpos)
GRIPPER_HOME = 0.04  # fingers open, in metres of travel per finger

# The two finger joints run in opposite directions: joint1 is [0, 0.04] and
# joint2 is [-0.04, 0]. Commanding both the same way drives them to the same
# side and the gripper never closes on anything.
FINGER_SIGNS = (1.0, -1.0)
# Finger gain: the fingers have to stall against a 18 mm handle bar and still
# hold it, so kp * (typical squeeze) has to come out around a real grasp force.
FINGER_KP = 8000.0
FINGER_KV = 200.0
FINGER_FORCE = 100.0

# Position-actuator gains per arm joint, and the torque each may command.
ARM_KP = (4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 500.0)
ARM_KV = (450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 50.0)
ARM_FORCE = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)

ARM_JOINTS = tuple(f"robot0_joint{i}" for i in range(1, 8))
FINGER_JOINTS = ("gripper0_finger_joint1", "gripper0_finger_joint2")
# The Omron base can drive and raise its torso. Nothing here does, so each of
# these is simply pinned at zero — without an actuator the base would drift.
BASE_JOINTS = (
    "mobilebase0_joint_mobile_forward",
    "mobilebase0_joint_mobile_side",
    "mobilebase0_joint_mobile_yaw",
    "mobilebase0_joint_torso_height",
)


def _look_at(eye: np.ndarray, target: np.ndarray) -> str:
    """MuJoCo camera `xyaxes` (right vector, then up vector) aiming eye at target."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return " ".join(f"{v:.4f}" for v in np.concatenate([right, up]))


def _drawer_z(index: int) -> float:
    """Centre height of drawer `index`, counting from 0 at the bottom of the stack."""
    return TABLE_H + DRAWER_H / 2 + index * DRAWER_H


def _add_table(worldbody: ET.Element) -> None:
    """A plain four-legged table for the dresser to stand on."""
    top_z = TABLE_H - TABLE_TOP_T
    ET.SubElement(
        worldbody, "geom", name="table_top", type="box",
        pos=f"0 0 {top_z:.4f}",
        size=f"{TABLE_W / 2:.4f} {TABLE_D / 2:.4f} {TABLE_TOP_T:.4f}",
        material="table_mat", condim="3", friction="1 0.02 0.001", group="1",
    )
    leg_h = (TABLE_H - 2 * TABLE_TOP_T) / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            x = sx * (TABLE_W / 2 - LEG_INSET)
            y = sy * (TABLE_D / 2 - LEG_INSET)
            ET.SubElement(
                worldbody, "geom", name=f"table_leg_{sx}_{sy}".replace("-", "n"),
                type="cylinder", pos=f"{x:.4f} {y:.4f} {leg_h:.4f}",
                size=f"{LEG_R} {leg_h:.4f}", material="leg_mat",
                condim="3", group="1",
            )


def _add_robot(world):
    """Merge the RoboCasa robot: a Panda on an Omron base with a parallel jaw."""
    from robosuite.models.bases import robot_base_factory
    from robosuite.models.grippers import gripper_factory
    from robosuite.models.robots import PandaOmron

    robot = PandaOmron(idn=0)
    robot.add_base(robot_base_factory("OmronMobileBase", idn=0))
    robot.add_gripper(gripper_factory("PandaGripper", idn=0))
    robot.set_base_xpos(list(ROBOT_POS))
    # The base's default heading looks along +x; yaw it to face the handles.
    yaw = np.arctan2(HANDLE_FACE[1] - ROBOT_POS[1], HANDLE_FACE[0] - ROBOT_POS[0])
    robot.set_base_ori([0.0, 0.0, float(yaw)])
    world.merge(robot)
    return robot


def _replace_actuators(root: ET.Element, slide_joints: list[str]) -> None:
    """Swap the robot's torque actuators for position actuators.

    robosuite ships the Panda with direct-torque motors because its controllers
    compute the torques. Nothing here runs a controller, so the arm would simply
    collapse; position actuators let the scene hold and command a pose instead.

    `slide_joints` is accepted and deliberately left unactuated — see below.
    """
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    for child in list(actuator):
        actuator.remove(child)

    for joint, kp, kv, force in zip(ARM_JOINTS, ARM_KP, ARM_KV, ARM_FORCE):
        ET.SubElement(actuator, "position", name=f"{joint}_act", joint=joint,
                      kp=f"{kp}", kv=f"{kv}",
                      forcerange=f"{-force} {force}")
    for joint, sign in zip(FINGER_JOINTS, FINGER_SIGNS):
        low, high = sorted((0.0, sign * 0.04))
        ET.SubElement(actuator, "position", name=f"{joint}_act", joint=joint,
                      kp=f"{FINGER_KP}", kv=f"{FINGER_KV}",
                      ctrlrange=f"{low} {high}",
                      forcerange=f"{-FINGER_FORCE} {FINGER_FORCE}")
    for joint in BASE_JOINTS:
        ET.SubElement(actuator, "position", name=f"{joint}_act", joint=joint,
                      kp="12000", kv="1200", forcerange="-4000 4000")
    # No actuator on the drawer slides. They are passive, damped joints: the
    # only thing that can open a drawer in this scene is something pulling it.


def build_world():
    from robocasa.models.fixtures.cabinets import Drawer
    from robocasa.models.objects.objects import MJCFObject
    from robosuite.models import MujocoWorldBase

    world = MujocoWorldBase()
    root = world.root

    # --- compiler / visual defaults -------------------------------------------
    compiler = root.find("compiler")
    compiler.set("angle", "radian")
    compiler.set("meshdir", "/")
    compiler.set("texturedir", "/")

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.002")
    option.set("integrator", "implicitfast")

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.35 0.35 0.35",
                  specular="0 0 0")
    ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")
    ET.SubElement(visual, "global", azimuth="140", elevation="-20", offwidth="1200",
                  offheight="800")

    asset = root.find("asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient",
                  rgb1="0.32 0.4 0.5", rgb2="0.08 0.1 0.13", width="512", height="3072")
    ET.SubElement(asset, "texture", type="2d", name="groundplane", builtin="checker",
                  mark="edge", rgb1="0.24 0.26 0.28", rgb2="0.30 0.32 0.34",
                  markrgb="0.75 0.75 0.75", width="300", height="300")
    ET.SubElement(asset, "material", name="groundplane", texture="groundplane",
                  texuniform="true", texrepeat="5 5", reflectance="0.12")
    ET.SubElement(asset, "material", name="table_mat", rgba="0.33 0.24 0.18 1",
                  specular="0.25", shininess="0.35")
    ET.SubElement(asset, "material", name="leg_mat", rgba="0.20 0.15 0.11 1",
                  specular="0.2", shininess="0.3")

    worldbody = root.find("worldbody")
    ET.SubElement(worldbody, "light", pos="0 0 3.0", dir="0 0 -1", directional="true",
                  diffuse="0.5 0.5 0.5")
    ET.SubElement(worldbody, "light", pos="1.2 -1.2 2.0", dir="-0.5 0.5 -1",
                  directional="false", diffuse="0.4 0.4 0.4")
    ET.SubElement(worldbody, "geom", name="floor", type="plane", size="4 4 0.05",
                  material="groundplane", condim="3", group="1")

    carcass_top = TABLE_H + NUM_DRAWERS * DRAWER_H
    focus = np.array([0.0, -0.35, carcass_top - 0.15])
    for cam_name, eye in {
        "front": (0.0, -2.75, 1.25),
        "threequarter": (1.95, -2.10, 1.65),
        "overhead": (0.75, -1.85, 2.45),
    }.items():
        ET.SubElement(worldbody, "camera", name=cam_name,
                      pos=" ".join(f"{v:.4f}" for v in eye),
                      xyaxes=_look_at(np.asarray(eye, dtype=float), focus))
    # Close in on the top drawer and the worktop, where the bottles are.
    bottle_eye = np.array([0.62, -1.05, 1.52])
    bottle_focus = np.array([0.0, -0.28, carcass_top + 0.02])
    ET.SubElement(worldbody, "camera", name="bottles",
                  pos=" ".join(f"{v:.4f}" for v in bottle_eye),
                  xyaxes=_look_at(bottle_eye, bottle_focus))

    _add_table(worldbody)
    _add_robot(world)

    # --- drawers ---------------------------------------------------------------
    slide_joints: list[str] = []
    for i in range(NUM_DRAWERS):
        drawer = Drawer(
            name=f"drawer{i}",
            size=[DRAWER_W, DRAWER_D, DRAWER_H],
            pos=[0.0, 0.0, _drawer_z(i)],
            panel_type=PANEL_TYPE,
            texture=TEXTURE,
            handle_type=HANDLE_TYPE,
        )
        world.merge_assets(drawer)
        worldbody.append(drawer.get_obj())
        slide_joints.append(f"{drawer.naming_prefix}slidejoint")

    # A worktop slab so the dresser reads as a real piece of furniture.
    ET.SubElement(
        worldbody, "geom", name="worktop", type="box",
        pos=f"0 0 {carcass_top + 0.015:.4f}",
        size=f"{DRAWER_W / 2 + 0.02:.4f} {DRAWER_D / 2 + 0.02:.4f} 0.015",
        rgba="0.86 0.84 0.80 1", condim="3", friction="1 0.02 0.001", group="1",
    )

    # --- seasoning bottles ------------------------------------------------------
    # One standing on the worktop, two lying on their sides inside the top
    # drawer. The pair in the drawer are the point: nothing holds them, so when
    # the arm yanks the drawer out they roll back against the rear wall, and
    # when it shoves it shut they roll forward again.
    def add_bottle(name: str, asset: str, xy, surface_z: float,
                   quat=None) -> None:
        obj = MJCFObject(name=name, mjcf_path=str(_object_root() / asset / "model.xml"),
                         density=BOTTLE_DENSITY, friction=BOTTLE_FRICTION)
        world.merge_assets(obj)
        body = obj.get_obj()
        width, _, height = obj.size
        if quat is None:
            lift = height / 2          # upright: half its height off the surface
        else:
            lift = width / 2           # on its side: half its width, i.e. the radius
        body.set("pos", f"{xy[0]:.4f} {xy[1]:.4f} {surface_z + lift + 0.002:.4f}")
        if quat is not None:
            body.set("quat", " ".join(f"{v:.6f}" for v in quat))
        worldbody.append(body)

    worktop_z = carcass_top + 0.030
    add_bottle("cinnamon", BOTTLE_ASSETS["cinnamon"], (0.13, -0.04), worktop_z)

    # Interior floor of the top drawer: the carcass panel the contents sit on.
    inner_z = _drawer_z(NUM_DRAWERS - 1) - DRAWER_H / 2 + 0.045
    add_bottle("cayenne", BOTTLE_ASSETS["cayenne"], (-0.09, 0.10), inner_z,
               ON_ITS_SIDE)
    add_bottle("paprika", BOTTLE_ASSETS["paprika"], (0.10, 0.02), inner_z,
               ON_ITS_SIDE)

    _replace_actuators(root, slide_joints)
    # robosuite ships visualisation sites — a green grip-axis cylinder, the
    # end-effector frame arrows, a red centre marker. Park them in a geom group
    # the viewers do not draw; lookups by name still work.
    for site in root.iter("site"):
        site.set("group", "4")
    return world


def _home_keyframe(xml: str) -> str:
    """Compile once, pose the arm, and bake the result back in as a keyframe.

    Writing the rest pose into the file means `verify.py`, `view.py` and
    `serve.py` all start identically, with one call to `mj_resetDataKeyframe`.
    """
    arm_home = _arm_home()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for joint, value in zip(ARM_JOINTS, arm_home):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        data.qpos[model.jnt_qposadr[jid]] = value
    for joint, sign in zip(FINGER_JOINTS, FINGER_SIGNS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        data.qpos[model.jnt_qposadr[jid]] = sign * GRIPPER_HOME
    mujoco.mj_forward(model, data)

    ctrl = np.zeros(model.nu)
    for joint, value in zip(ARM_JOINTS, arm_home):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint}_act")
        ctrl[aid] = value
    for joint, sign in zip(FINGER_JOINTS, FINGER_SIGNS):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint}_act")
        ctrl[aid] = sign * GRIPPER_HOME

    root = ET.fromstring(xml)
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", name="home",
                  qpos=" ".join(f"{v:.6f}" for v in data.qpos),
                  ctrl=" ".join(f"{v:.6f}" for v in ctrl))
    return ET.tostring(root, encoding="unicode")


def write_scene(path: Path | None = None) -> Path:
    path = path or (HERE / "scene.xml")
    world = build_world()
    xml = ET.tostring(world.root, encoding="unicode")
    path.write_text(_home_keyframe(xml))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = write_scene(args.out)
    model = mujoco.MjModel.from_xml_path(str(out))
    print(f"wrote {out}")
    print(f"{model.nbody} bodies, {model.ngeom} geoms, "
          f"{model.nu} actuators, table top at {TABLE_H:.2f} m")

"""Build the MuJoCo scene: a Panda arm, a three-drawer dresser, and a plated meal.

The drawers are real RoboCasa `Drawer` fixtures: a procedurally sized carcass with
a prismatic (slide) joint, fronted by a scanned mesh door panel and a scanned mesh
handle from the RoboCasa / Lightwheel fixture library (free, Apache-2.0, downloaded
with `robocasa.scripts.download_kitchen_assets --type fixtures_lw`; note that
repeating `--type` overwrites rather than appends, so several packs go after one
flag).

The dresser stands on a table. The top drawer is a spice drawer: four scanned
seasoning containers stand upright in it, the front-centre one a pepper shaker.
On the table in front of the dresser sits a plate with a steak and a head of
broccoli on it. A PandaOmron — the Franka Panda on an Omron mobile base that
RoboCasa's own kitchen tasks drive — faces the drawer fronts across the floor.

The drawer slide joints carry no actuator on purpose. They are passive and
damped, so the only way a drawer opens here is if the gripper takes hold of its
handle and pulls. Nothing holds the shaker in the drawer either: it is a free
body, so the only way it reaches the plate is if the gripper picks it up.

Running this module writes `scene.xml` next to it, self-contained apart from the
mesh files it references by absolute path inside the robocasa asset tree. The
written file carries a `home` keyframe holding both the arm's rest pose and the
matching actuator targets, so every consumer starts from the same configuration:

    mujoco.mj_resetDataKeyframe(model, data, 0)

RoboCasa and RoboSuite are imported where they are used rather than at module
scope, so importing this module costs nothing and `ensure_scene` can say what
is missing instead of failing on an import line.
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


def _object_root() -> Path:
    """Where the RoboCasa object library is installed, resolved on demand."""
    import robocasa

    return Path(robocasa.__file__).resolve().parent / "models" / "assets" / "objects"


# The spice drawer's contents, and the plated meal in front of it. Every one is
# a scanned mesh from the RoboCasa object library, with its own baked texture
# and a convex collision hull.
#
# Height matters for the drawer: the carcass panel above the top drawer sits
# 0.129 m above the drawer's own floor, so anything standing in it has to be
# shorter than that or it jams on the way back in. The pepper shaker is 0.101
# tall and the salt 0.095; the two spice jars are 0.084 and 0.111.
SHAKER_ASSET = "lightwheel/salt_and_pepper_shaker/PepperShaker007"
DRAWER_CONTENTS = {
    # name:            (asset,                                         x,     y)
    "pepper": (SHAKER_ASSET, 0.00, -0.11),   # the one the arm picks up
    "salt": ("lightwheel/salt_and_pepper_shaker/SaltShaker003", -0.15, -0.11),
    "paprika": ("lightwheel/paprika/Paprika002", -0.12, 0.10),
    "cinnamon": ("lightwheel/cinnamon/Cinnamon004", 0.13, 0.07),
}
WORKTOP_BOTTLE = ("lightwheel/paprika/Paprika001", -0.14, -0.02)

# The plate and its food come from the objaverse pack rather than the
# lightwheel one, so that pack has to be installed too — see the README.
PLATE_ASSET = "objaverse/plate/plate_0"
STEAK_ASSET = "objaverse/steak/steak_3"
BROCCOLI_ASSET = "objaverse/broccoli/broccoli_0"
# Where the plated meal stands on the table: on the near edge, squarely in
# front of the drawer and far enough forward that the open drawer stops short
# of it, so the arm can come straight down onto the plate without threading
# past the drawer it just pulled out.
PLATE_XY = (0.14, -0.64)

# The library ships spice containers with a rolling friction of 0.1, which pins
# one to the spot it lands on. Dropped to a hundredth of that they behave.
JAR_FRICTION = (0.9, 0.02, 0.001)
JAR_DENSITY = 320.0        # a part-full glass shaker, ~50 g
# The plate has to sit still while a shaker is waved over it, so it is heavy
# for its size and grippy against the table.
PLATE_DENSITY = 900.0
PLATE_FRICTION = (1.2, 0.05, 0.005)
FOOD_DENSITY = 700.0
FOOD_FRICTION = (1.0, 0.05, 0.005)

# The table the dresser stands on. Its top surface is TABLE_H above the floor.
# Chosen against the Panda's reach and the base's 0.34 m torso lift: with the
# torso up, the drawer handle sits a hand's breadth above the shoulder; with it
# down, the plate is a comfortable reach below. A fixed torso cannot have both —
# the arm can work the drawer or the plate, not the pair.
TABLE_H = 0.49
TABLE_W = 1.30
TABLE_D = 1.16
# The table is offset forward of the dresser rather than centred on it, so
# there is room in front of the open drawer for the plate.
TABLE_Y = -0.155
TABLE_TOP_T = 0.025  # half-thickness of the top slab
LEG_R = 0.032
LEG_INSET = 0.07

# Mobile base, on the floor, just off square in front of the drawer fronts and
# far enough back that its shell clears the table's near edge.
ROBOT_POS = (-0.24, -0.94, 0.0)
HANDLE_FACE = (0.0, -0.237)  # where the drawer handles sit, for aiming the base
# The Omron base carries the Panda on a 0.34 m lift, and the routine uses it:
# up to work the drawer, down again to work over the plate. `plan.py` drives
# it; this is only where the scene parks it to start with, and it has to match
# that module's TORSO_HIGH or the arm starts the routine at the wrong height.
TORSO_HEIGHT = 0.24


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
# The Omron base can drive and raise its torso. Only the torso is commanded by
# the routine; the other three are simply pinned at zero — without an actuator
# the base would drift.
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


# The interior floor of the top drawer — the panel its contents stand on. The
# `Drawer` fixture builds an inner tray whose bottom panel is centred 0.064 m
# below the drawer's mid-height and is 0.030 m thick.
TOP_DRAWER_FLOOR = _drawer_z(NUM_DRAWERS - 1) - 0.064 + 0.015
WORKTOP_Z = TABLE_H + NUM_DRAWERS * DRAWER_H + 0.030   # top face of the worktop
TABLE_TOP_Z = TABLE_H                                   # top face of the table


def _add_table(worldbody: ET.Element) -> None:
    """A plain four-legged table for the dresser and the plate to stand on."""
    top_z = TABLE_H - TABLE_TOP_T
    ET.SubElement(
        worldbody, "geom", name="table_top", type="box",
        pos=f"0 {TABLE_Y:.4f} {top_z:.4f}",
        size=f"{TABLE_W / 2:.4f} {TABLE_D / 2:.4f} {TABLE_TOP_T:.4f}",
        material="table_mat", condim="3", friction="1 0.02 0.001", group="1",
    )
    leg_h = (TABLE_H - 2 * TABLE_TOP_T) / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            x = sx * (TABLE_W / 2 - LEG_INSET)
            y = TABLE_Y + sy * (TABLE_D / 2 - LEG_INSET)
            ET.SubElement(
                worldbody, "geom", name=f"table_leg_{sx}_{sy}".replace("-", "n"),
                type="cylinder", pos=f"{x:.4f} {y:.4f} {leg_h:.4f}",
                size=f"{LEG_R} {leg_h:.4f}", material="leg_mat",
                condim="3", group="1",
            )


def _add_robot(world) -> None:
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


def _replace_actuators(root: ET.Element) -> None:
    """Swap the robot's torque actuators for position actuators.

    robosuite ships the Panda with direct-torque motors because its controllers
    compute the torques. Nothing here runs a controller, so the arm would simply
    collapse; position actuators let the scene hold and command a pose instead.

    The drawer slide joints are deliberately left out — see below.
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
    # No actuator on the drawer slides, and none on the seasoning: they are
    # passive, damped joints and free bodies. The only thing that can move a
    # drawer or a shaker in this scene is something taking hold of it.


def build_world():
    """Assemble the whole scene as a robosuite `MujocoWorldBase`."""
    from robocasa.models.fixtures.cabinets import Drawer
    from robocasa.models.objects.objects import MJCFObject
    from robosuite.models import MujocoWorldBase

    object_root = _object_root()
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

    # --- cameras ---------------------------------------------------------------
    # The routine spans the drawer (at the origin, about a metre up) and the
    # plate (out at +x, down on the table), so every camera has to hold both.
    carcass_top = TABLE_H + NUM_DRAWERS * DRAWER_H
    focus = np.array([0.05, -0.42, 0.88])
    for cam_name, eye in {
        # High enough to see into the open drawer, round enough to see the
        # plate: the routine spans both and no low camera holds the pair.
        "action": (0.90, -1.80, 1.95),
        "threequarter": (1.70, -1.90, 1.70),
        "front": (0.25, -2.60, 1.45),
        "overhead": (0.70, -1.45, 2.60),
    }.items():
        ET.SubElement(worldbody, "camera", name=cam_name,
                      pos=" ".join(f"{v:.4f}" for v in eye),
                      xyaxes=_look_at(np.asarray(eye, dtype=float), focus))
    # Down into the open top drawer, for watching the shaker come out.
    drawer_eye = np.array([0.28, -1.10, 1.90])
    drawer_focus = np.array([0.0, -0.34, 1.02])
    ET.SubElement(worldbody, "camera", name="drawer",
                  pos=" ".join(f"{v:.4f}" for v in drawer_eye),
                  xyaxes=_look_at(drawer_eye, drawer_focus))
    # Close on the plate, for watching the seasoning come down over the food.
    plate_eye = np.array([PLATE_XY[0] + 0.55, PLATE_XY[1] - 0.60, 1.15])
    plate_focus = np.array([PLATE_XY[0], PLATE_XY[1], TABLE_TOP_Z + 0.12])
    ET.SubElement(worldbody, "camera", name="plate",
                  pos=" ".join(f"{v:.4f}" for v in plate_eye),
                  xyaxes=_look_at(plate_eye, plate_focus))

    _add_table(worldbody)
    _add_robot(world)

    # --- drawers ---------------------------------------------------------------
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

    # A worktop slab so the dresser reads as a real piece of furniture.
    ET.SubElement(
        worldbody, "geom", name="worktop", type="box",
        pos=f"0 0 {carcass_top + 0.015:.4f}",
        size=f"{DRAWER_W / 2 + 0.02:.4f} {DRAWER_D / 2 + 0.02:.4f} 0.015",
        rgba="0.86 0.84 0.80 1", condim="3", friction="1 0.02 0.001", group="1",
    )

    # --- loose objects ----------------------------------------------------------
    def stand(name: str, asset_path: str, xy, surface_z: float,
              density: float = JAR_DENSITY, friction=JAR_FRICTION,
              clearance: float = 0.002):
        """Drop a library object onto `surface_z`, upright, standing on its base."""
        obj = MJCFObject(name=name,
                         mjcf_path=str(object_root / asset_path / "model.xml"),
                         density=density, friction=friction)
        world.merge_assets(obj)
        body = obj.get_obj()
        lift = float(-obj.bottom_offset[2])
        body.set("pos", f"{xy[0]:.4f} {xy[1]:.4f} "
                        f"{surface_z + lift + clearance:.4f}")
        worldbody.append(body)
        return obj

    # Four seasoning containers standing in the top drawer. Nothing holds any of
    # them: they are free bodies resting on the drawer floor, so the pepper
    # shaker only leaves the drawer if the gripper carries it out.
    for name, (asset_path, x, y) in DRAWER_CONTENTS.items():
        stand(name, asset_path, (x, y), TOP_DRAWER_FLOOR)

    # One tall jar on the worktop, well clear of everything the arm does.
    asset_path, x, y = WORKTOP_BOTTLE
    stand("worktop_jar", asset_path, (x, y), WORKTOP_Z)

    # The plated meal, on the table in front of the dresser. The food is dropped
    # a few millimetres above the plate's rim and settles into it.
    plate = stand("plate", PLATE_ASSET, PLATE_XY, TABLE_TOP_Z,
                  density=PLATE_DENSITY, friction=PLATE_FRICTION)
    plate_face = TABLE_TOP_Z + 2.0 * float(plate.top_offset[2]) - 0.008
    stand("steak", STEAK_ASSET, (PLATE_XY[0] - 0.005, PLATE_XY[1] + 0.012),
          plate_face, density=FOOD_DENSITY, friction=FOOD_FRICTION,
          clearance=0.004)
    stand("broccoli", BROCCOLI_ASSET, (PLATE_XY[0] + 0.055, PLATE_XY[1] - 0.045),
          plate_face, density=FOOD_DENSITY, friction=FOOD_FRICTION,
          clearance=0.004)

    _replace_actuators(root)
    # robosuite ships visualisation sites — a green grip-axis cylinder, the
    # end-effector frame arrows, a red centre marker. Park them in a geom group
    # the viewers do not draw; lookups by name still work.
    for site in root.iter("site"):
        site.set("group", "4")
    return world


def _home_keyframe(xml: str) -> str:
    """Compile once, pose the arm, and bake the result back in as a keyframe.

    Writing the rest pose into the file means `verify.py`, `viewer.py` and the
    Flow lane all start identically, with one `mj_resetDataKeyframe`.
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
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                              "mobilebase0_joint_torso_height")
    data.qpos[model.jnt_qposadr[torso]] = TORSO_HEIGHT
    mujoco.mj_forward(model, data)

    ctrl = np.zeros(model.nu)
    for joint, value in zip(ARM_JOINTS, arm_home):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint}_act")
        ctrl[aid] = value
    for joint, sign in zip(FINGER_JOINTS, FINGER_SIGNS):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint}_act")
        ctrl[aid] = sign * GRIPPER_HOME
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                            "mobilebase0_joint_torso_height_act")
    ctrl[aid] = TORSO_HEIGHT

    root = ET.fromstring(xml)
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", name="home",
                  qpos=" ".join(f"{v:.6f}" for v in data.qpos),
                  ctrl=" ".join(f"{v:.6f}" for v in ctrl))
    return ET.tostring(root, encoding="unicode")


def ensure_scene(path: Path | None = None) -> Path:
    """Return a usable `scene.xml`, building it first if it is not there yet.

    The file is generated rather than committed — it references meshes by
    absolute path inside whichever RoboCasa install produced it — so a fresh
    clone has no scene until something asks for one. The viewer and the MuJoCo
    lane both call this so that a clone plus the asset packs is enough to run
    them, with no separate build step to remember.
    """
    path = path or (HERE / "scene.xml")
    if path.exists():
        return path
    try:
        return write_scene(path)
    except ImportError as exc:
        raise RuntimeError(
            "building the scene needs RoboCasa and RoboSuite. Install them with "
            '`pixi run python -m pip install -e ".[robocasa_drawer]"` plus RoboCasa '
            "from source, then fetch the meshes with `pixi run demo-drawer-assets`. "
            "See this example's README."
        ) from exc


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
    print(f"top drawer floor at {TOP_DRAWER_FLOOR:.3f} m, "
          f"worktop at {WORKTOP_Z:.3f} m")

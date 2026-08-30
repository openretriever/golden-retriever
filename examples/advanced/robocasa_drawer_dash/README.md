# Drawer dash: an arm that opens a drawer by holding onto it

A MuJoCo scene built from RoboCasa parts — a three-drawer dresser standing on a
table, with a seasoning bottle on its worktop and two more lying loose inside
the top drawer — and a Panda on an Omron mobile base, the same robot RoboCasa's
own kitchen tasks drive, which takes hold of the top drawer's handle and pulls
it open.

The bottles are scanned spice jars from the RoboCasa object library, with their
own textures and convex collision meshes. Nothing holds the two in the drawer,
so they roll: back against the rear wall as the drawer is pulled out, forward
again as it is shoved shut.

![four stages: lining up on the handle, gripping it, the drawer pulled out with two spice jars rolling inside, and pushed shut again](../../../docs-site/src/assets/media/drawer-dash/drawer-dash-stages.jpg)

## 1. Mock-safe contract

Run this first. It needs no simulator, no assets, no GPU and no network:

```bash
pixi run demo-drawer-dash-mock
```

Expected output — the nine phases in order, the drawer travelling only while
the handle is held, and a successful finish:

```text
[mock step=0004] phase=line up          drawer=0.000m grasped=False progress=6.1% success=False
[mock step=0012] phase=grip handle      drawer=0.000m grasped=True progress=30.5% success=False
[mock step=0016] phase=pull drawer open drawer=0.144m grasped=True progress=42.7% success=False
[mock step=0020] phase=pull drawer open drawer=0.409m grasped=True progress=54.9% success=False
[mock step=0024] phase=push drawer shut drawer=0.115m grasped=True progress=67.1% success=False
[mock step=0036] phase=withdraw         drawer=0.000m grasped=False progress=100.0% success=True

routine complete: peak drawer travel 0.409 m (needs >= 0.35), shut to 0.000 m
(needs <= 0.02), 9 phases over 16.4 s -> success=True
```

The mock reproduces the timeline, the commanded travel and the pass thresholds.
It does not reproduce contact: only the MuJoCo lane can show that the *grasp* is
what moves the drawer, and `verify.py` is what asserts it.

## 2. The interactive viewer

Three commands from a fresh clone. The middle one is a ~2.8 GB download and is
only ever run once:

```bash
pixi run python -m pip install -e ".[drawer_dash]"   # + RoboCasa, from source
pixi run demo-drawer-dash-assets                     # fixtures_lw + objs_lw meshes
pixi run demo-drawer-dash-viewer                     # http://localhost:8087
```

The viewer builds `scene.xml` itself on first run, so there is no separate
build step to remember. It runs under plain `python` — no `mjpython`, no
native window — and streams the live simulation to the browser.

Its "grasp" panel is the point of the thing: **holding handle** reads `yes` or
`no` tick by tick, and when it reads `no` the drawer stops moving, because
nothing else in the scene can move it. **bottles rolled** counts how far the
two loose jars have spun, which separates rolling from being dragged along.
`restart routine` resets to the home keyframe; unticking `run routine` parks
the arm at its home pose.

RoboCasa is not published on PyPI — install it from source. Its assets are
fetched by the task above, which wraps
`robocasa.scripts.download_kitchen_assets`; this scene needs `fixtures_lw` for
the cabinet panels and handles, and `objs_lw` for the spice jars.

### Why not mjviser

`mjviser` is the obvious way to put a MuJoCo model in a viser page, and it is
what an earlier draft of this scene used. It requires `mujoco>=3.6`, but
RoboSuite's controllers do not survive the `mj_fullM` signature change in 3.10,
and `scene.py` needs RoboSuite to build the robot at all — so this example is
pinned to `mujoco==3.3.1` and talks to viser directly instead. `viewer.py`
pushes each visual geom once as a triangle mesh and then moves it per frame;
126 geoms and about 277k triangles for this scene.

## 3. The other simulator lanes

```bash
pixi run demo-drawer-dash          # the routine, through Retriever Flows
pixi run demo-drawer-dash-verify   # the assertions, headless
pixi run demo-drawer-dash-scene    # rebuild scene.xml explicitly
```

`scene.xml` is generated, not committed: it references the mesh and texture
files by absolute path inside your RoboCasa install, so it is only valid on the
machine that built it.

## What the robot actually does

It grasps the handle and pulls. **The drawer slide joints carry no actuator** —
they are passive and damped — so the drawer physically cannot open unless
something takes hold of it. The routine is: square up to the bar, advance until
the open fingers straddle it, close them, then drive the *hand* out along -y for
41 cm. The drawer comes because the fingers are on it.

That is also the check. `verify.py` asserts, first, that no actuator in the
model acts on a slide joint; then that a finger pad is in contact with the
handle for 100% of the pull and push. Take the grasp away and the drawer stays
shut and the run fails.

The handle is a cylinder lying along world x, standing about 8 mm proud of the
drawer front, so the fingers have to close *vertically* across it. That needs
orientation control, not just a reach: `arm_control.py` does one damped
least-squares step per tick against the full 6-row site Jacobian — position and
orientation together — integrated into position-actuator targets.

Three things had to be right before any of this worked:

- robosuite ships the Panda with direct-torque motors because its own
  controllers compute the torques. Nothing here runs a robosuite controller, so
  `scene.py` swaps them for position actuators; without that the arm collapses.
- The two finger joints run in **opposite** directions — joint1 is `[0, 0.04]`,
  joint2 is `[-0.04, 0]`. Commanding both the same way drives them to the same
  side and the gripper never closes on anything.
- The fingers have to stall against an 18 mm bar and still hold it, so their
  gain is 8000 with a 100 N force limit. At the default 120 the grip was worth
  about a newton and the handle slid straight out.

## Layout

| File | What it holds |
| --- | --- |
| `plan.py` | the nine-phase choreography as plain data — no MuJoCo, no RoboCasa, so the mock lane and the tests share the simulator's schedule |
| `app.py` | the Retriever Flows: a policy that walks the schedule, a simulator that runs it against MuJoCo or the mock, a printer |
| `scene.py` | builds the scene and writes `scene.xml` |
| `sequence.py` | executes a phase against a real model: phase to gripper pose |
| `arm_control.py` | damped least-squares Cartesian control for the Panda |
| `verify.py` | the headless assertions, plus video and contact-sheet capture |
| `viewer.py` | the browser viewer: MuJoCo geoms pushed to viser, and the grasp readouts |

`app.py` imports nothing heavier than the runtime, so the example stays
import-safe with no simulator installed; the MuJoCo modules are pulled in only
when `--mode mujoco` asks for them.

## Where the parts came from

| Piece | Source |
| --- | --- |
| Carcass + prismatic slide joint | `robocasa/models/fixtures/cabinets.py::Drawer` on `fixtures/cabinets/drawer.xml` |
| Door front | `CabinetDoorPanel009` — scanned mesh + 1024² oak albedo, RoboCasa / Lightwheel fixture library |
| Handle | `CabinetHandle012` — scanned mesh, same library |
| Carcass finish | `textures/wood/light_wood_planks.png` |
| Robot | `robosuite.models.robots.PandaOmron` + `OmronMobileBase` + `PandaGripper` |
| Table | built here from a slab and four legs |
| Seasoning bottles | `objects/lightwheel/cinnamon/Cinnamon001` and `Cinnamon003`, `objects/lightwheel/paprika/Paprika001` — scanned jars with baked labels, same free library |

## Recording it

`verify.py` renders while it asserts, so the video is a recording of the run
that passed rather than a separate take:

```bash
pixi run demo-drawer-dash-verify --video drawer-dash.mp4 --sheet drawer-dash.png
```

Cameras in the scene: `front`, `threequarter` (the default), `overhead`, and
`bottles`, which closes in on the top drawer and the worktop.

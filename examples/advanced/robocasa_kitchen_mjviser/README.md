# RoboCasa kitchens through mjviser — handover

**Branched from `examples/robocasa-retriever`, with `examples/robocasa-drawer` merged in**, so
the drawer-dash package and its history are intact and future work on that branch still merges
cleanly.

The short version: **mjviser *can* drive RoboCasa**, the version pin that says otherwise is
conservative, and the two bugs that made it look broken were a camera-framing miscalculation and a
wrong denominator in a speed readout. Neither was about geometry or physics.

```bash
pixi run demo-kitchen-mjviser      # a real RoboCasa kitchen, live at localhost:8092
pixi run demo-drawer-dash-mjviser  # the drawer-dash scene, live at localhost:8090
```

---

## What was actually wrong

### 1. mjviser and RoboCasa can share an environment

`robocasa/__init__.py` hard-asserts `mujoco == 3.3.1`; mjviser requires `mujoco >= 3.6` in **all 14
of its releases**. That is why `viewer.py` talks to viser directly and why
`demo-robocasa-web --visualize mjviser` cannot run as written.

Measured: RoboCasa builds and steps a full kitchen on **mujoco 3.9** — 288 bodies, 1522 geoms for
`OpenDrawer` — with no observed misbehaviour. The pin is conservative. `serve_kitchen.py`
neutralises the assert for the duration of the import and restores the real version immediately,
so nothing downstream is misled.

The catch that matters more: **robosuite must come from git master.** The PyPI build is missing
`robosuite.utils.mjcf_utils.get_elements`, which RoboCasa imports. That single missing symbol is
what makes a naive `pip install robosuite` fail.

### 2. The blank canvas was camera framing, not geometry

The kitchen rendered nothing — while the drawer-dash scene rendered fine through the identical code
path. The scene was there the whole time, as a speck.

MuJoCo derives `stat.extent` / `stat.center` from **every** geom in the model. RoboCasa kitchens
carry a lot that is never drawn: fully transparent placeholder fixtures, and `robot0_base` parked
out at (10, 10, 0). So the stats described a volume **29 m** across when the visible room is
**4.8 m**, centred outside it — and mjviser frames its default camera at `3 * extent`.

`frame_on_visible_geometry()` recomputes both over exactly the geoms mjviser keeps (a visible group,
non-zero alpha, not the ground plane), using each geom's true world AABB rather than `geom_rbound`
— the bounding sphere over-pads walls and floors enough to double the extent on its own.

### 3. The 0.07x was a wrong denominator, not slow physics

mjviser assumes one `step_fn` call advances the sim by exactly `model.opt.timestep`. `env.step`
advances a **control** timestep — 25 model steps at `control_freq=20` — so calling it from `step_fn`
overstated the required work 25-fold. mjviser asked for 500 env-steps a second, never got them, sat
permanently `[CAPPED]`, and divided the true rate by 25 in the readout: **0.12x displayed while the
sim was really running at 3.1x**.

`step_fn` now advances exactly one model timestep and runs the controller on the 20 Hz policy
boundary, which is what `env.step`'s inner loop does. The resulting trajectory is bit-identical
(verified: `max|dq| = 0.0` over 1.5 s). mjviser now reports the truth, throttles to 1.00x, and
spends the surplus on rendering: **1.00x at a steady 60 FPS**.

---

## Setup

Two environments, because the video path and the viewer path want different mujoco. Neither needs
the other.

| | mujoco | what it is for |
|---|---|---|
| `.venv-robocasa` | **3.3.1** | offscreen frames for video. No spoofing needed — mjviser is not involved. |
| `.venv-mjviser` | **>= 3.6** | the live browser viewer. The compat shim applies only here. |

```bash
# viewer env
python3.11 -m venv .venv-mjviser
.venv-mjviser/bin/pip install "mujoco>=3.6" "numpy==2.2.5" mjviser viser
git clone https://github.com/ARISE-Initiative/robosuite    # master, NOT PyPI
git clone https://github.com/robocasa/robocasa
.venv-mjviser/bin/pip install -e robosuite -e robocasa

# assets (~4 GB; the objaverse/aigen packs are another ~9 GB and are not needed)
cd robocasa && python -m robocasa.scripts.download_kitchen_assets \
  --type objs_lw --type fixtures_lw --type tex --type tex_generative
```

`obj_registries=("lightwheel",)` is required with only `objs_lw` installed: RoboCasa samples across
registries and divides by the total count, so an empty registry gives `Probabilities contain NaN`.

---

## Still open — where to take this next

**The kitchen has no motion.** `serve_kitchen.py` steps physics with a zero action, so it is a
beautiful static scene. This is the gap between what exists and a demo.

**The drawer-dash routine is the obvious donor, but it does not port directly.** `plan.py` and
`sequence.py` are written against `scene.py`'s own names — `drawer2_slidejoint`,
`drawer2_door_handle_g1`, `cinnamon_main` — and its geometry. A RoboCasa kitchen has real cabinets
and drawers with entirely different names and poses.

Worth considering before porting furniture: **RoboCasa already ships the task.** There are 374
registered envs, 13 of them drawer-related, including `PickPlaceCounterToDrawer`,
`PickPlaceDrawerToCounter` and `PlaceVeggiesInDrawer`. `PickPlaceCounterToDrawer` is the seasoning
errand already staged, in a real kitchen, with the props. So the valuable thing to carry across is
the **choreography**, not the dresser: retarget the phase table onto the kitchen's own drawer joint
and handle geom, rather than importing `scene.py`'s cabinet into the kitchen.

If the dresser *is* wanted in the kitchen, `scene.py` builds it procedurally from RoboCasa `Drawer`
fixtures, so it can be emitted into a kitchen arena — but the arm's reach envelope, the torso
heights (`TORSO_DOWN` / `TORSO_UP`) and the standoff geometry were all tuned against a table at
0.40 m, and would need re-tuning against counter height.

**One fix already made to the drawer-dash routine, worth folding upstream.** The stock slot sits
`HANDLE_TO_INTERIOR = 0.24` m behind the handle, close enough to the drawer front that the arm clips
it on the way down: measured, the drawer is shoved from 0.403 m back to 0.300 m across
"lower into drawer" and "let go of jar", and stays there for the rest of the cycle. Placing 0.30 m
behind the handle and releasing at `SLOT_PLACE_Z = 1.00` keeps it at 0.403 m untouched. Applied here
as a runtime override so `plan.py` stays as written — but it is a real defect in the numbers, not a
preference.

**Also:** the routine loops without resetting, so from the second cycle the arm reaches for a jar
that is already in the drawer and closes on empty air. `serve_drawer_dash.py` recycles the scene at
each cycle boundary.

**Untried:** bimanual. robosuite ships `TwoArmLift`, `TwoArmHandover`, `TwoArmTransport`, and
Baxter / GR1 / Tiago. Swapping the robot is one line; the choreography is written against one 7-DOF
arm's IK and would need rewriting.

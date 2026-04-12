# TAMP tabletop pick-place MVP

A deliberately small, **Golden-first** task-and-motion-planning example.

This is not a shared planning package and not a Predicators clone. It is a local example that demonstrates the narrow loop we want to validate first:

1. build a scripted tabletop scene
2. task-plan over a tiny symbolic domain
3. lazily refine **only the next symbolic action**
4. execute that refined step
5. repeat until the goal is satisfied

## Scope

Included in this v0 scaffold:
- one movable object: `red_block`
- one start region and one goal region
- two symbolic operators: `Pick` and `Place`
- local A* task planning
- local motion refinement with candidate filtering
- simple execution-state updates

Explicitly not included yet:
- perception
- learning / NSRTs
- generalized multi-object planning
- Dora / distributed runtime wiring
- full RoboPlan-backed trajectory execution

## Files

- `domain.py` — local operator logic plus the initial state and goal, all expressed on top of the shared `retriever_tamp.core.types` symbolic payloads
- `scene.py` — scripted tabletop geometry and motion candidates
- `task_planner.py` — tiny local A* over grounded operators
- `motion_refiner.py` — lazy next-step refinement
- `bridge.py` — narrow bridge into `retriever_tamp` execution/refinement interfaces without a second symbolic conversion layer
- `pybullet_sim.py` — lightweight tabletop simulator for execution playback with a real UR5 + suction visual
- `ur5_arm.py` — tiny UR5+suction arm wrapper using the local legacy asset files, without importing the old environment stack
- `../shared/pybullet.py` — tiny shared PyBullet viewer/bootstrap helper distilled from older env code, without importing the older environment stack
- `app.py` — end-to-end loop routed through `retriever_tamp.execution.TAMPController`

## Run

From the `GoldenRetriever/` repo root:

```bash
python examples/advanced/tamp_tabletop_pick_place/app.py
```

That default path keeps execution symbolic-only and does not require PyBullet.

To run the same controller loop against the lightweight tabletop simulator:

```bash
python examples/advanced/tamp_tabletop_pick_place/app.py --sim pybullet-direct
```

To open the PyBullet viewer:

```bash
python examples/advanced/tamp_tabletop_pick_place/app.py --sim pybullet-gui
```

The GUI path intentionally runs with a slower default step sleep so the motion is actually visible on desktop.
Add `--final-hold-seconds 8` if you want the viewer to stay open briefly after the final pose.

To use the repo-managed Pixi environment for the simulator-backed path:

```bash
pixi run -e tamp demo-tamp-tabletop
```

Useful variations:

```bash
python examples/advanced/tamp_tabletop_pick_place/app.py --no-obstacle
python examples/advanced/tamp_tabletop_pick_place/app.py --sim pybullet-direct --no-obstacle
python examples/advanced/tamp_tabletop_pick_place/app.py --sim pybullet-gui
pixi run -e tamp demo-tamp-tabletop-nosim
pixi run -e tamp demo-tamp-tabletop-gui
```

## Simulator modes

- `--sim none` keeps execution symbolic-only and is the fastest debug path
- `--sim pybullet-direct` runs the same loop in a headless PyBullet scene
- `--sim pybullet-gui` shows the tabletop animation in a PyBullet window

The current simulator is intentionally thin: it animates a real UR5 arm with the legacy suction geometry, but the pickup/place behavior is still a deterministic demo path rather than a full contact-rich robot execution backend.

## Why this shape

This example stays local on purpose. The current repo still has older planning/runtime surfaces mixed with newer package boundaries. The safest move is to validate the TAMP loop in one example directory and promote only the reusable seams.

The symbolic boundary is intentionally small:
- shared object-centric symbolic payloads live in `retriever_tamp.core.types`
- local files keep only example-specific planning logic, scene geometry, and simulator behavior
- the controller loop composes those stages structurally instead of wrapping each step in a new ad hoc payload class

## Broader subsystem direction

This local example is still the right concrete foothold, but the broader reusable direction is now documented in:

- `docs/tamp/2026-03-15_predicators_style_tamp_direction.md`
- `packages/retriever-tamp/`

The recommended path is to keep this demo runnable while gradually lifting reusable symbolic / refinement / execution-loop pieces into the standalone `retriever-tamp` package boundary. The current app already routes through `TAMPController`; the simulator and tabletop scene remain example-local.

## Next likely upgrade

Replace the current candidate-feasibility checks in `motion_refiner.py` with a light RoboPlan-backed refinement path for one symbolic step at a time:
- `Pick(obj)` -> pregrasp IK, grasp IK, short approach motion
- `Place(obj, region)` -> placement pose, IK, short approach + retreat

That would preserve the same high-level loop while making the motion side real.

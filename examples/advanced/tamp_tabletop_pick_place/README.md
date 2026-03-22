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

- `domain.py` — symbolic atoms, operators, initial state, goal
- `scene.py` — scripted tabletop geometry and motion candidates
- `task_planner.py` — tiny local A* over grounded operators
- `motion_refiner.py` — lazy next-step refinement
- `app.py` — end-to-end sequential loop

## Run

From the `GoldenRetriever/` repo root:

```bash
python3 examples/advanced/tamp_tabletop_pick_place/app.py
```

Or, after the added Pixi task lands:

```bash
pixi run demo-tamp-tabletop
```

To see the same flow without the obstacle affecting the first placement candidate:

```bash
python3 examples/advanced/tamp_tabletop_pick_place/app.py --no-obstacle
```

## Why this shape

This example stays local on purpose. The current repo still has symbolic import drift between older `retriever.types.*` references and newer `Retriever/retriever/core/symbolic_structs.py` surfaces. Until that is cleaned up, the safest move is to validate the TAMP loop in one example directory and only then decide what abstractions deserve promotion.

## Broader subsystem direction

This local example is still the right concrete foothold, but the broader reusable direction is now documented in:

- `docs/tamp/2026-03-15_predicators_style_tamp_direction.md`
- `packages/retriever-tamp/`

The recommended path is to keep this demo runnable while gradually lifting reusable symbolic / refinement / execution-loop pieces into the standalone `retriever-tamp` package boundary.

## Next likely upgrade

Replace the current candidate-feasibility checks in `motion_refiner.py` with a light RoboPlan-backed refinement path for one symbolic step at a time:
- `Pick(obj)` -> pregrasp IK, grasp IK, short approach motion
- `Place(obj, region)` -> placement pose, IK, short approach + retreat

That would preserve the same high-level loop while making the motion side real.

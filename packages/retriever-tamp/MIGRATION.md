# retriever-tamp migration plan

Status: package skeleton / migration-ready documentation pass

This file is the handoff-oriented companion to the lightweight Python scaffold.

## Current stance

Keep the concrete demo runnable in:

- `GoldenRetriever/examples/advanced/tamp_tabletop_pick_place/`

Promote code into `packages/retriever-tamp/` only when it improves reuse or package cleanliness.

That means this package should absorb:

- stable interfaces,
- reusable symbolic logic,
- reusable next-step refinement logic,
- reusable controller logic,
- bridge adapters.

It should **not** absorb example-only scene glue too early.

## Phase map

### Phase 0 — current repo foothold

Source of truth for the runnable loop:

- `GoldenRetriever/examples/advanced/tamp_tabletop_pick_place/`

Purpose:

- prove the loop cheaply,
- keep debugging friction low,
- avoid broad runtime coupling while the shape is still moving.

### Phase 1 — current package skeleton

Source of truth for reusable boundaries:

- `GoldenRetriever/packages/retriever-tamp/`

Purpose:

- define the package boundary now,
- make future subtree/repo extraction easier,
- record the intended landing zones before deeper ports begin.

### Phase 2 — first concrete ports

Good first ports:

1. local symbolic planner implementation
2. local tabletop candidate refiner
3. tabletop problem/world spec
4. example rewritten as a thin bridge over `TAMPController`

Do **not** start by pulling in broad GoldenRetriever planning/runtime internals.

### Phase 3 — GoldenRetriever bridge layer

Add adapters that translate:

- simulator/runtime observations -> `WorldSnapshot`
- domain/problem state -> `GoalSpec` and symbolic state
- symbolic action -> refinement request/provider
- execution result -> `ExecutionFeedback`

This is the right stage for RoboPlan-backed next-step refinement.

### Phase 4 — extraction choice

After the interfaces stop moving quickly, either:

- keep `packages/retriever-tamp/` vendored in the monorepo, or
- split it into a standalone `retriever-tamp` repo.

## File-by-file migration map

| Current example file | Package landing zone | Notes |
| --- | --- | --- |
| `examples/advanced/tamp_tabletop_pick_place/domain.py` | `src/retriever_tamp/symbolic/` + `src/retriever_tamp/problems/tabletop_pick_place/` | separate generic symbolic structs/operators from one concrete problem instance |
| `examples/advanced/tamp_tabletop_pick_place/task_planner.py` | `src/retriever_tamp/symbolic/planners/` | keep planner reusable across domains/problems |
| `examples/advanced/tamp_tabletop_pick_place/motion_refiner.py` | `src/retriever_tamp/refinement/providers/` | preserve lazy next-step refinement contract |
| `examples/advanced/tamp_tabletop_pick_place/scene.py` | `src/retriever_tamp/problems/tabletop_pick_place/` | world/task definition should not live in generic refinement code |
| `examples/advanced/tamp_tabletop_pick_place/app.py` | `src/retriever_tamp/execution/` + `src/retriever_tamp/bridges/` | controller stays reusable; demo stays thin |

## Promotion checklist

Promote code from the example into the package only if at least one is true:

- it is needed by more than one demo/problem,
- it expresses a stable boundary that is likely to survive extraction,
- it removes duplication without pulling runtime baggage into the kernel,
- it makes the example thinner without making the package more example-shaped.

Keep code in the example if it is still:

- scene-specific,
- geometry-hack-specific,
- debugging-oriented,
- likely to be replaced soon by RoboPlan/runtime-specific logic.

## What remains intentionally open

- whether the first concrete symbolic planner is A*, BFS, or a tiny wrapper around existing search utilities
- exactly how much tabletop scene geometry should live in the package versus the example
- how quickly RoboPlan-backed refinement should replace the mocked candidate filter
- whether later GoldenRetriever bridges should stay package-local or live outside as repo adapters

## Practical rule

The package should become the home of **reusable contracts and reusable logic**, not a dumping ground for the current example.

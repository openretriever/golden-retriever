# retriever-tamp

Predicators-style task-and-motion-planning kernel for GoldenRetriever.

## Intent

This package is deliberately placed behind its own package boundary inside the monorepo.

That gives GoldenRetriever a cleaner path to:

- keep the current example runnable,
- define a small stable TAMP API before deep integration,
- split this package into its own repo later,
- or manage it as a subtree-style reusable component.

The current runnable foothold remains:

- `examples/advanced/tamp_tabletop_pick_place/`

This package is the **migration target for reusable pieces**, not a replacement for the demo yet.

## What exists today

The scaffold is intentionally light, but it now has a clearer package shape:

```text
packages/retriever-tamp/
  README.md
  MIGRATION.md
  pyproject.toml
  src/retriever_tamp/
    core/
    perception/
    symbolic/
      planners/
    refinement/
      providers/
    execution/
    problems/
      tabletop_pick_place/
    bridges/
```

It currently defines:

- shared TAMP data types,
- goal and snapshot boundary objects,
- perception interfaces,
- symbolic planning interfaces,
- refinement interfaces,
- a small closed-loop controller surface,
- problem/world definition interfaces,
- a bridge note for the current tabletop MVP,
- placeholder package landing zones for the first concrete ports.

It does **not** yet provide:

- a real planner implementation,
- a RoboPlan bridge,
- GoldenRetriever runtime adapters,
- learned models or NSRT training,
- a concrete tabletop problem module inside the package.

## Boundary decisions

A few boundary choices are deliberate:

- `GoalSpec` is the task-facing goal surface, instead of passing raw goal atoms everywhere.
- `WorldSnapshot` is the concrete state handoff point between observation/perception and TAMP.
- `TAMPController` owns only the boring reusable loop: abstract -> plan -> refine next step -> execute -> decide whether to replan.
- GoldenRetriever-specific runtime wiring belongs in bridges/adapters, not in the kernel interfaces.

## Why a separate package?

GoldenRetriever’s current TAMP foothold lives in:

- `examples/advanced/tamp_tabletop_pick_place/`

That example is useful, but it is not yet the right public library shape.

Putting `retriever-tamp` behind its own package boundary makes the migration cleaner than immediately placing unfinished abstractions inside `src/golden_retriever/planning/`.

## Recommended migration path

1. keep the current example runnable,
2. port reusable symbolic/refinement pieces into this package,
3. wrap the example through `retriever_tamp.execution.TAMPController`,
4. add GoldenRetriever-specific bridges only after the core interfaces settle,
5. decide later whether `packages/retriever-tamp/` stays vendored or becomes its own repo.

For the concrete file-by-file migration plan, see:

- `packages/retriever-tamp/MIGRATION.md`
- `retriever_tamp.bridges.legacy_tabletop_pick_place`

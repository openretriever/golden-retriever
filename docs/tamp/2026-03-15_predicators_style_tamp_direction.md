# Predicators-style TAMP direction for GoldenRetriever

Status: proposed architecture + initial subtree-friendly scaffold

## Executive summary

The current `examples/advanced/tamp_tabletop_pick_place/` scaffold was the right **local MVP**: it validated a minimal loop with symbolic planning, lazy next-step refinement, and execution-state updates without forcing premature abstractions into the main package.

The next move should **not** be “promote those four files directly into `src/golden_retriever/planning/`”. That would lock GoldenRetriever into an example-shaped structure and make later extraction harder.

Instead, the recommended direction is:

1. keep the current example as the concrete foothold,
2. define a **small reusable TAMP kernel** with clear module boundaries,
3. place it in a **separate monorepo package boundary** now,
4. bridge GoldenRetriever into that package through adapters,
5. only fold it deeper into the main package after the interfaces stabilize.

Concretely, this note recommends a new monorepo package:

- `packages/retriever-tamp/`
- import surface: `retriever_tamp`
- future repo candidate: `retriever-tamp`

This follows the spirit of:

- **Predicators** for decomposition of perceiver / symbolic model / planner / refinement / execution monitor,
- **PRPL’s newer monorepo style** (`prpl-mono`) for keeping reusable packages separately installable inside a larger research repo.

## Why this is the right level-up

The current example is intentionally local and explicitly says so. That was useful because:

- symbolic surfaces in the repo are still in flux,
- GoldenRetriever’s main planning package already mixes newer and legacy planning ideas,
- the example is small enough to validate a concrete loop quickly.

But if we stop at the example-local shape, we get the wrong optimization target:

- `scene.py` conflates world definition, candidate generation, feasibility checking, and state mutation,
- `domain.py` mixes symbolic data types with one specific object set and one problem instance,
- `motion_refiner.py` is tied to one kind of candidate enumeration,
- `app.py` contains the coordinator logic that should eventually live behind a reusable execution loop.

The reusable subsystem should instead optimize for:

- **clear seams** between symbolic, refinement, execution, perception, and problem definition,
- **replaceable implementations** at each seam,
- **GoldenRetriever bridgeability** rather than GoldenRetriever entanglement,
- **eventual split-out** into a standalone repo or subtree without painful surgery.

## Design principles

### 1. Package boundary first, deep integration later

Do **not** start by placing the kernel under `src/golden_retriever/planning/tamp/`.

Use a separate package boundary first:

- it is easier to subtree-split later,
- it avoids accidental imports from unstable internal modules,
- it keeps dependency creep under control,
- it mirrors how PRPL’s monorepo keeps reusable packages installable as subdirectories.

### 2. Separate world/problem data from planning logic

A TAMP library should not assume one scene, one object set, or one robot.

Keep distinct:

- **world definitions**: geometry, movable/static entities, task-relevant constants,
- **problem definitions**: initial state, goal, task instance,
- **symbolic abstractions**: predicates, operators, grounding logic,
- **refinement**: turning a symbolic action into executable motion/skill candidates,
- **execution**: sending one refined step to a robot/sim and deciding whether to replan.

### 3. Use “provider / extractor / adapter” names instead of “oracle” at the public surface

Predicators and research code often use “oracle” in a useful way, but for a reusable subsystem it is a poor public API name because it bakes in an assumption about privileged access.

Recommended surface terms:

- `StateExtractor` instead of a hard-coded perceiver implementation,
- `SymbolicModel` for abstraction and goal/operator exposure,
- `RefinementProvider` instead of oracle motion model,
- `ExecutionAdapter` for robot/sim execution,
- `ExecutionMonitor` for replan decisions.

Internally, a specific implementation can still be labeled “oracle”.

### 4. Keep the coordinator small and boring

The most reusable part is the loop:

1. receive observation / snapshot,
2. abstract into symbolic state,
3. plan symbolically,
4. refine only the next action,
5. execute,
6. observe feedback,
7. replan when needed.

That loop should be stable even as:

- the planner changes,
- the refiner changes,
- perception gets smarter,
- execution moves from toy scene mutation to RoboPlan / robot skills.

## Proposed architecture

```text
observation stream
    |
    v
perception.receiver / state_extractor
    |
    v
WorldSnapshot
    |
    +--> symbolic.SymbolicModel.abstract(snapshot)
    |         |
    |         v
    |     SymbolicState + operators + goals
    |         |
    |         v
    |     symbolic.TaskPlanner.plan(problem)
    |         |
    |         v
    |     [GroundAction, ...]
    |
    +--> refinement.RefinementProvider.refine(next_action, snapshot)
              |
              v
          RefinementResult
              |
              v
        execution.ExecutionAdapter.execute(...)
              |
              v
         ExecutionFeedback
              |
              v
        execution.ExecutionMonitor.should_replan(...)
```

## Recommended public interfaces

### Perception layer

Purpose: isolate raw observation and belief/state extraction from the rest of the TAMP stack.

Recommended interfaces:

- `ObservationReceiver`
- `StateExtractor`
- `BeliefUpdater` (optional, when history matters)
- `WorldSnapshot`

GoldenRetriever-specific receivers can later wrap:

- Dora/flow observation streams,
- robot state streams,
- simulator state,
- vision-language or geometric predicate extraction.

### Symbolic layer

Purpose: define the task abstraction without committing to one environment.

Recommended interfaces:

- `GroundAtom`
- `GroundAction`
- `OperatorSchema`
- `TaskPlanningProblem`
- `SymbolicModel`
- `TaskPlanner`

This is where the current `domain.py` and `task_planner.py` logic should ultimately land, but broken apart so that:

- object sets are problem-specific,
- operators are reusable across problems,
- planner implementations are swappable.

### Refinement layer

Purpose: produce executable candidates for one symbolic step.

Recommended interfaces:

- `RefinementRequest`
- `ExecutionPrimitive`
- `RefinementCandidate`
- `RefinementResult`
- `RefinementProvider`

This layer is the natural home for:

- today’s candidate enumeration and collision checks,
- a later RoboPlan-backed refiner,
- future learned samplers or skill-conditioned motion generators.

### Execution layer

Purpose: own the closed-loop coordinator and runtime-facing adapters.

Recommended interfaces:

- `ExecutionAdapter`
- `ExecutionFeedback`
- `ExecutionMonitor`
- `TAMPController`
- `ReplanReason`

This is the reusable “CogMan-like” loop. It should stay agnostic to whether execution is:

- mutating a toy scene object,
- sending a motion plan to a robot,
- dispatching a GoldenRetriever skill flow.

### Problems / worlds layer

Purpose: separate reusable domain logic from concrete task instances.

Recommended interfaces:

- `WorldDefinition`
- `ProblemDefinition`
- `ProblemFactory`

This keeps environment/problem setup from getting embedded in symbolic or refinement modules.

## Proposed subtree-friendly directory layout

```text
packages/
  retriever-tamp/
    README.md
    pyproject.toml
    src/retriever_tamp/
      __init__.py
      api.py                         # optional later convenience exports
      core/
        types.py
      perception/
        base.py
        receivers/
          dora.py                    # later
          simulation.py              # later
        extractors/
          geometric.py               # later
          vlm.py                     # later
      symbolic/
        base.py
        planners/
          astar.py                   # later
          sesame.py                  # later
      refinement/
        base.py
        providers/
          tabletop_candidates.py     # first concrete port
          roboplan.py                # later
      execution/
        loop.py
        monitor.py                   # later if needed
      problems/
        base.py
        tabletop_pick_place/
          problem.py                 # later
          scene_spec.py              # later
      bridges/
        legacy_tabletop_pick_place.py
        golden_retriever.py          # later
      tests/
        test_controller.py           # later
        test_tabletop_refinement.py  # later
```

## Why this package boundary is better than `src/golden_retriever/planning/tamp`

### Better for migration

A standalone package directory with its own `pyproject.toml` can later become:

- a split-out repo with minimal rewriting,
- a git subtree import/export unit,
- a separately versioned dependency for GoldenRetriever.

### Better for dependency hygiene

The TAMP kernel should initially depend on almost nothing. That becomes much harder if it immediately starts importing broad GoldenRetriever planning/runtime internals.

### Better for experimentation

GoldenRetriever can carry multiple bridge implementations without forcing the kernel to absorb:

- flow runtime details,
- robot SDK details,
- environment-specific state formats.

## Concrete mapping from the current MVP example

Current file | Future home | Why
--- | --- | ---
`examples/advanced/tamp_tabletop_pick_place/domain.py` | `retriever_tamp/symbolic/base.py` + `retriever_tamp/problems/tabletop_pick_place/problem.py` | separate generic symbolic types from one tabletop task instance
`examples/advanced/tamp_tabletop_pick_place/task_planner.py` | `retriever_tamp/symbolic/planners/astar.py` | planner becomes reusable across problems
`examples/advanced/tamp_tabletop_pick_place/motion_refiner.py` | `retriever_tamp/refinement/providers/tabletop_candidates.py` | candidate generation/refinement becomes one provider implementation
`examples/advanced/tamp_tabletop_pick_place/scene.py` | `retriever_tamp/problems/tabletop_pick_place/scene_spec.py` | problem/world definition should not live in generic refinement code
`examples/advanced/tamp_tabletop_pick_place/app.py` | `retriever_tamp/execution/loop.py` + `retriever_tamp/bridges/legacy_tabletop_pick_place.py` | coordinator logic should be reusable; legacy example becomes a bridge/adapter

## Relationship to Predicators and closed-loop planning notes already in repo

The repo already has useful Predicators/CogMan notes in:

- `examples/experimental/closed_loop_planning/NOTES.md`
- `examples/experimental/closed_loop_planning/notes/predicators_integration.md`

Those notes already identify the right high-level split:

- perceiver,
- approach/planner,
- execution monitor,
- executor / policy loop.

The TAMP kernel proposed here is compatible with that split, but slightly refocused for GoldenRetriever:

- **Predicators-style decomposition** is retained,
- **GoldenRetriever integration details** are pushed to adapters,
- **problem/env definitions** are made explicit,
- **refinement** is promoted to a first-class layer instead of being hidden inside a local example.

## Initial migration plan

### Phase 0 — current state

Keep the existing example fully runnable and local.

### Phase 1 — package skeleton (this pass)

Add a subtree-friendly monorepo package:

- `packages/retriever-tamp/`
- only light interface modules and documentation,
- no deep coupling to current GoldenRetriever internals.

### Phase 2 — first concrete port

Port the MVP example into package-shaped modules while keeping the original example as a compatibility demo.

Suggested first moves:

1. move reusable symbolic data structures into `retriever_tamp.symbolic`,
2. move candidate refinement into `retriever_tamp.refinement.providers.tabletop_candidates`,
3. represent the tabletop task in `retriever_tamp.problems.tabletop_pick_place`,
4. rewrite the example as a thin adapter over `TAMPController`.

### Phase 3 — GoldenRetriever bridge

Introduce a bridge module that adapts GoldenRetriever runtime surfaces to the kernel:

- observation -> `WorldSnapshot`
- symbolic abstraction -> `SymbolicModel`
- symbolic step -> `RefinementProvider`
- execution feedback -> `ExecutionFeedback`

This is where RoboPlan and robot skill execution should land.

### Phase 4 — flow/runtime integration

Once the bridge is stable, connect the controller to the repo’s closed-loop flow setup.

That is the right time to reuse ideas from the experimental CogMan-style closed-loop planning example.

### Phase 5 — extraction or vendor decision

After the package stabilizes, choose one of two clean options:

1. **split repo**: move `packages/retriever-tamp/` into its own repository,
2. **subtree-managed package**: keep it in GoldenRetriever as a separately versioned package.

Do this **after** the interfaces stabilize, not before.

## What should *not* happen yet

- no immediate broad import of `src/golden_retriever/planning/*` into the kernel,
- no heavy learning/NSRT implementation wave yet,
- no large repo-wide symbolic migration to support TAMP before the package boundaries are tested,
- no deep robot-runtime entanglement in the public TAMP interfaces.

## Why this is the right direction for GoldenRetriever specifically

GoldenRetriever already appears to be moving toward clearer repo boundaries:

- the root `pyproject.toml` explicitly discusses future repo/package splitting,
- some existing docs already use subtree-managed language,
- the main planning package still contains legacy/archive material.

A separate `retriever-tamp` package fits that trajectory better than burying TAMP under the current main planning surface.

## Recommended next implementation slice

If the next pass is allowed to do more than scaffolding, the most leverage-per-line change is:

1. implement a tiny `AStarTaskPlanner` in `retriever_tamp.symbolic.planners.astar`,
2. port the current `motion_refiner.py` into `retriever_tamp.refinement.providers.tabletop_candidates`,
3. wrap the existing example through `TAMPController`,
4. leave all GoldenRetriever-specific execution outside the kernel.

That would prove the package boundary with the current example while keeping the migration path clean.

---
marp: true
theme: default
paginate: true
size: 16:9
title: Predicators-style TAMP direction for GoldenRetriever
style: |
  section { font-size: 28px; }
  h1 { font-size: 42px; }
  h2 { font-size: 28px; }
  code { font-size: 18px; }
  pre { font-size: 18px; }
---

# Predicators-style TAMP direction for GoldenRetriever

Move from a useful local MVP to a clean reusable subsystem.

- Repo: `GoldenRetriever`
- Foothold: `examples/advanced/tamp_tabletop_pick_place/`
- Recommendation: **separate monorepo package first**, deep integration later

---

## What the current MVP proved

The current example already validates the right narrow loop:

1. build a small symbolic problem
2. plan symbolically
3. refine **only the next** action
4. execute that step
5. replan if refinement fails

That is a strong foothold.

The mistake now would be promoting the example files directly into the main planning package.

---

## Why “just move the files into planning/” is the wrong next step

Current example files mix several concerns:

- `domain.py`
  - symbolic types
  - operator schemas
  - one object set
  - one problem instance
- `scene.py`
  - world definition
  - candidate generation
  - feasibility check
  - state mutation
- `app.py`
  - the reusable coordinator loop

That shape is great for an MVP, but too example-shaped for a library boundary.

---

## Recommended boundary

**Create a standalone monorepo package now:**

```text
packages/
  retriever-tamp/
    pyproject.toml
    src/retriever_tamp/
```

Why:

- easier future split-out or git-subtree extraction
- cleaner dependency boundary
- avoids entangling TAMP with unstable main-package planning surfaces
- matches the repo’s broader split-friendly direction

---

## Upstream inspiration

Two useful reference points:

- **Predicators**
  - perceiver / planner / execution monitor / execution loop split
- **PRPL monorepo style**
  - multiple separately installable packages inside one research repo

So the goal is **Predicators-style decomposition + PRPL-style package boundary**.

---

## Proposed architecture

```text
Observation
   -> StateExtractor / BeliefUpdater
   -> WorldSnapshot
   -> SymbolicModel.abstract(...)
   -> TaskPlanner.plan(...)
   -> [GroundAction, ...]
   -> RefinementProvider.refine(next_action, snapshot)
   -> RefinementResult
   -> ExecutionAdapter.execute(...)
   -> ExecutionFeedback
   -> ExecutionMonitor.should_replan(...)
```

Key point: the coordinator stays stable while planners, refiners, and adapters change.

---

## Public interface seams to stabilize

Use public names that survive migration:

- `ObservationReceiver`
- `StateExtractor`
- `WorldSnapshot`
- `SymbolicModel`
- `TaskPlanner`
- `RefinementProvider`
- `ExecutionAdapter`
- `ExecutionMonitor`
- `TAMPController`
- `ProblemDefinition`

Rename “oracle” to **provider** at the public API level.

---

## Proposed module layout

```text
packages/retriever-tamp/src/retriever_tamp/
  core/
  perception/
  symbolic/
  refinement/
  execution/
  problems/
  bridges/
```

Recommended ownership:

- `core/` → shared types
- `perception/` → observation -> snapshot extraction
- `symbolic/` → predicates, operators, planners
- `refinement/` → continuous grounding of one symbolic step
- `execution/` → closed-loop controller + runtime adapters
- `problems/` → world definitions and task instances
- `bridges/` → GoldenRetriever / legacy example glue

---

## How the current MVP maps into the new structure

```text
current example                          future home
--------------------------------------   ------------------------------------------
domain.py                                symbolic/ + problems/tabletop_pick_place/
task_planner.py                          symbolic/planners/astar.py
motion_refiner.py                        refinement/providers/tabletop_candidates.py
scene.py                                 problems/tabletop_pick_place/scene_spec.py
app.py                                   execution/loop.py + bridges/
```

The current example stays useful, but becomes a **bridge demo**, not the subsystem root.

---

## Migration plan

### Phase 1
- add `retriever-tamp` package skeleton
- keep dependencies minimal

### Phase 2
- port the current tabletop example into package-shaped modules
- keep legacy example runnable

### Phase 3
- add GoldenRetriever bridge
- connect observation, refinement, and execution adapters

### Phase 4
- wire into closed-loop planning flows

### Phase 5
- decide: split repo or subtree-managed package

---

## Why this is right for GoldenRetriever

It matches three realities in the repo:

- the current TAMP example is intentionally local
- the planning package still carries legacy/archive baggage
- the repo already hints at future package/repo splits

So the right move is:

**build a reusable TAMP kernel beside GoldenRetriever, not buried inside it**.

---

## What this pass adds

Artifacts:

- design / migration note
- package skeleton for `retriever-tamp`
- bridge note for the current MVP example
- this Marp deck + PDF export

This is enough to anchor the broader direction without overcommitting the implementation too early.

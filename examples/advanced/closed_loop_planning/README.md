# Closed-Loop Planning Patterns

This note preserves the useful design ideas from the earlier experimental closed-loop planning prototype without keeping the brittle prototype code in the public example path.

Closed-loop robot planning is useful when high-level goals, partial observability, and low-level execution feedback all matter at the same time. The important pattern is not one large script. It is a set of small flows that run on different clocks and exchange typed state.

## Pattern

A robust closed-loop planning graph usually separates these roles:

- **Environment or robot I/O**: emits observations and accepts low-level actions at the control or simulator rate.
- **Perception**: converts raw observations into task-relevant state, predicates, detections, or language-grounded facts.
- **Belief update**: maintains memory across partial observations and prevents known facts from accidentally regressing to unknown.
- **Planner**: produces a symbolic, language, or skill-level plan when the goal is not satisfied or a replan is requested.
- **Skill executor**: runs the current skill locally and reports completion, failure, or progress.
- **Execution monitor**: watches global progress and decides when expensive replanning is necessary.

The graph is cyclic on purpose: execution changes the world, perception updates belief, monitoring can request replanning, and the planner updates the executor.

## Belief Invariants

The most durable idea from the prototype was the belief layer for partially observable tasks:

- once a fact is known, it should not silently become unknown;
- unknown predicates should be explicit, not encoded as missing fields;
- information-gathering actions should be allowed to reduce uncertainty;
- planners and monitors should consume the same belief object instead of each reconstructing local state.

A compact flow pattern is:

```text
observation -> perception -> belief -> planner
                            -> executor
                            -> monitor -> replan request -> planner
```

## Bilevel Monitoring

Closed-loop planning benefits from two monitoring rates:

- **Fast local checks** in the executor: is the current skill still valid, terminal, or failed?
- **Slow global checks** in the monitor: has the robot deviated from the plan, or is replanning worth the cost?

This avoids calling an expensive planner or VLM on every control tick while still reacting quickly to local failures.

## Relation To Earlier Architecture

The old prototype mirrored a cognitive-manager style decomposition:

| Role | Retriever flow role |
| --- | --- |
| Perceiver | `PerceptionFlow` |
| Approach / planner | `PlannerFlow` |
| Execution monitor | `ExecutionMonitorFlow` |
| Policy execution | `SkillExecutorFlow` |
| Belief state | `BeliefUpdaterFlow` |

The public examples should now teach these ideas through smaller maintained ladders first: perception, memory, language grounding, state management, and then composition.

## Current Maintained Starting Points

Use these maintained examples before building a larger closed-loop planner:

- `examples/advanced/perception_examples/`: concise perception flows.
- `examples/advanced/memory_examples/`: belief and dropout-memory patterns.
- `examples/advanced/language_examples/`: captioning, grounding, and primitive plan text.
- `examples/advanced/state_management/`: reset behavior and older event-driven replanning examples.
- `examples/advanced/functional_wiring/`: fan-in/fan-out and surfaced composition.
- `examples/advanced/vlm_gridworld/`: a smaller visual closed-loop control demo with mock and VLM modes.

## What Was Not Kept

The removed experimental implementation included live webcam capture, web instruction serving, VLM calls, Spot-specific paths, Rerun side effects, and multiple direct script entrypoints. Those pieces made the prototype hard to maintain as a public example. Future runnable closed-loop planning examples should be rebuilt mock-first and promoted only after their tasks, docs, and dependencies are stable.

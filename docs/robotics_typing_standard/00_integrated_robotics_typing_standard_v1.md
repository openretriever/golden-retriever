# Robot Type-Pack Overview

Golden's robot type pack is a small applied profile on top of Retriever. It gives examples a shared language for robot state, beliefs, plans, commands, trajectories, and event streams without turning Golden into a second runtime.

## Why This Exists

Robot examples often pass values named `pose`, `velocity`, or `force`, but those values are unsafe to compose unless the boundary also says what frame, time, source, and units they belong to. The type pack makes those assumptions explicit at the reusable edges of a graph.

<div class="gr-fit-grid">
  <div class="gr-fit-card">
    <span>Readable</span>
    <strong>Keep examples ordinary</strong>
    <p>Flows still use normal Python payload classes. The extra structure lives at boundaries where ambiguity would hurt.</p>
  </div>
  <div class="gr-fit-card">
    <span>Composable</span>
    <strong>Avoid silent collisions</strong>
    <p>Composite inputs support deterministic qualified access, so two sources can both have a <code>pose</code> field without guessing.</p>
  </div>
  <div class="gr-fit-card">
    <span>Reusable</span>
    <strong>Prepare Hub packs</strong>
    <p>Examples can graduate into Hub-loadable packs only when their payloads, imports, and smoke tests are stable.</p>
  </div>
</div>

## Payload Families

| Family | Types | Use |
| --- | --- | --- |
| Spatial values | `Vector3`, `Quaternion`, `SE3Pose`, `Twist`, `Wrench`, `JointState` | Geometry, velocity, force, and joint-level state. |
| Stamped boundaries | `Header`, `PoseStamped`, `TwistStamped`, `WrenchStamped` | Carry frame, time, and source at graph boundaries. |
| Robot task state | `WorldState`, `RobotState`, `BeliefGraph` | Keep perception and memory examples on a shared schema. |
| Planning/execution | `Skill`, `Plan`, `StructuredPlan`, `TaskGoal`, `Trajectory`, `ExecutionStatus` | Connect language, planning, and control-facing examples. |
| Data streams | `Event`, `EventBuffer`, `MultiStreamBuffer`, manifests | Record, join, replay, and export multi-stream runs. |

## Authoring Pattern

```python
from retriever_typing import PoseStamped, get_type
from retriever_typing.robotics_types import WorldState, Plan

pose_type = get_type("PoseStamped")
```

Use stamped payloads at graph, process, dataset, or robot boundaries. Inside a small local Flow, it is fine to convert to leaner internal values once frame and time have already been resolved.

## Boundary Walkthrough

The promoted boundary demo follows this path:

1. perception emits a camera-frame `PoseStamped`,
2. normalization emits a base-frame `PoseStamped`,
3. control emits a typed command payload,
4. serialization round-trips the command with stable type identity.

```bash
pixi run demo-robotics-typing-boundary
```

Source: `examples/advanced/robotics_typing_standard/perception_to_control_boundary_demo.py`.
